"""
Tests for the geometric augmentation in code/data/generic.py.

Runnable without pytest:  python tests/test_geometric_augmentation.py

The augmentation exists because the store presents a fixed image *set* per game.
`ConceptDataset.__getitem__`'s permutation decides which agent sees which of the
20 positives, not which 20 images the game has, so a hundred epochs is a hundred
passes over the same pixels. What that produces is on record: the 2026-09-03
Conv4 baseline reached 0.880 on `train_acc_md_shape` and 0.484 on both eval
splits, with colour transferring and shape not at all.

Four properties have to hold, and each is checked here.

That the draw is per *image*. A single draw applied to the whole `(n, C, H, W)`
row would put one sampled transform on every referent in it, which
varies the epoch but not the game, and leaves the speaker's and listener's views
of a shared stored image under the same transform. Twenty identical inputs must
come back twenty different ways.

That it is off by default and a passthrough when off, so every run recorded
before the keys existed reproduces from its own config.

That nothing but the pixels moves: dtype, shape, and the caller's tensor, which
is a view onto the shared in-memory store and must not be written through.

And that the corners rotation leaves behind are the background. ShapeWorld
renders on black, so `padding_mode="zeros"` is the background rather than a
value appearing nowhere else in the dataset -- which is exactly the kind of
thing a model keys on when the task is hard and the artefact is easy.

What is *not* tested here is label preservation, because it is a property of the
transform's matrix rather than of the pixels it produces. Against this
dataset's five shapes -- circle, ellipse, rectangle, square, triangle -- shear
turns a rectangle into a parallelogram and anisotropic scaling maps circle to
ellipse and square to rectangle.
`test_the_transform_is_a_rotation_and_nothing_else` pins that on
`_rotation_theta` directly -- orthonormal, determinant +1, zero
translation -- so that letting one of them in has to be a deliberate act that
fails a test first.
"""

import inspect

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401

from data.generic import ConceptDataset, _rotation_theta


DEGREES = 10.0


class _Probe(ConceptDataset):
    """
    The augmentation alone, without a store, a vocabulary or a game behind it.

    `ConceptDataset.__init__` wants all three and none of them reach
    `_augment_geometry`, so constructing one here would test the loader rather
    than the transform.
    """

    def __init__(self, augment_flip=False, augment_affine_degrees=0.0):
        self.augment_flip = augment_flip
        self.augment_affine_degrees = augment_affine_degrees


def _referents(n=20):
    """
    `n` identical images of one off-centre bar.

    Off-centre so that a horizontal flip moves it, and asymmetric top to bottom
    so that a vertical flip does too. A centred square would be its own mirror
    and every test below would pass on a transform that did nothing.
    """
    imgs = torch.zeros(n, 3, 32, 32, dtype=torch.uint8)
    imgs[:, :, 6:20, 8:12] = 200
    return imgs


def test_the_draw_is_per_image_and_not_per_row():
    np.random.seed(0)
    out = _Probe(True, DEGREES)._augment_geometry(_referents())
    distinct = {image.numpy().tobytes() for image in out}
    # Twenty identical inputs, so any repeat is a shared draw. Two rows landing
    #     on the same transform by chance is possible but vanishingly unlikely
    #     against a continuous angle.
    assert len(distinct) == 20, f"{len(distinct)} distinct outputs from 20 rows"


def test_both_agents_draw_separately():
    """
    The listener's view of a stored image must not be the speaker's view of it.

    This is the property that makes the augmentation bite: the two agents split
    one game's images, so a transform shared between them would leave the pair
    looking at the same pixels it always has.
    """
    np.random.seed(0)
    probe = _Probe(True, DEGREES)
    referents = _referents()
    speaker = probe._augment_geometry(referents)
    listener = probe._augment_geometry(referents)
    assert not torch.equal(speaker, listener)


def test_it_is_a_passthrough_when_both_keys_are_off():
    referents = _referents()
    out = _Probe(False, 0.0)._augment_geometry(referents)
    assert torch.equal(out, referents)


def test_the_defaults_are_off():
    signature = inspect.signature(ConceptDataset.__init__)
    assert signature.parameters["augment_flip"].default is False
    assert signature.parameters["augment_affine_degrees"].default == 0.0


def test_the_callers_tensor_is_not_written_through():
    """
    `self.x[i]` is a view onto the shared store when it is held in memory, so an
        in-place transform would corrupt the dataset for every later epoch.
    """
    referents = _referents()
    before = referents.clone()
    np.random.seed(0)
    _Probe(True, DEGREES)._augment_geometry(referents)
    assert torch.equal(referents, before)


def test_shape_and_dtype_survive():
    referents = _referents()
    out = _Probe(True, DEGREES)._augment_geometry(referents)
    assert out.shape == referents.shape
    assert out.dtype == referents.dtype


def test_the_rotated_corners_are_background():
    np.random.seed(0)
    out = _Probe(False, DEGREES)._augment_geometry(_referents())
    corners = torch.stack(
        (out[:, :, 0, 0], out[:, :, 0, -1], out[:, :, -1, 0], out[:, :, -1, -1])
    )
    assert int(corners.max()) == 0


def test_flipping_alone_introduces_no_new_values():
    """
    A flip is a permutation of pixels, so the value set is closed under it.

    Rotation is not -- it interpolates -- which is why this is asserted with the
        angle off. It pins that `augment_flip` never reaches the affine branch.
    """
    referents = _referents()
    np.random.seed(0)
    out = _Probe(True, 0.0)._augment_geometry(referents)
    assert set(out.flatten().tolist()) == {0, 200}


def test_the_transform_is_a_rotation_and_nothing_else():
    """
    See this module's docstring: shear and anisotropic scale destroy labels in
        this dataset, so they have to be absent from the matrix.

    Stated as a fact about the transform rather than about how it was spelled.
        A rotation is exactly an orthonormal `(2, 2)` block of determinant +1 --
        orthonormality rules out shear and anisotropic scale, the determinant
        rules out a reflection folded in silently -- with a zero translation
        column. Anything that crept into `_rotation_theta` would have to break
        one of those.
    """
    angles = np.deg2rad(np.linspace(-180.0, 180.0, 37))
    theta = _rotation_theta(angles)
    assert theta.shape == (len(angles), 2, 3)

    block = theta[:, :, :2]
    identity = torch.eye(2).expand_as(block)
    assert torch.allclose(block @ block.transpose(1, 2), identity, atol=1e-6)
    assert torch.allclose(torch.linalg.det(block), torch.ones(len(angles)), atol=1e-6)

    # Exactly zero, not nearly: it is a literal in the matrix, not a product.
    assert torch.equal(theta[:, :, 2], torch.zeros(len(angles), 2))


def test_a_zero_angle_is_the_identity():
    """
    A zero angle must return the input pixels, not a half-pixel shift of them.

    `affine_grid` and `grid_sample` share an `align_corners` convention and a
        coordinate convention, and getting either wrong resamples every image
        by half a pixel. Every other test here passes under that -- the outputs
        still differ per image, the corners are still background, the dtype
        still survives -- so this is the one that catches it.
    """
    referents = _referents()
    theta = _rotation_theta(torch.zeros(referents.shape[0]))
    assert torch.equal(theta[:, :, :2], torch.eye(2).expand(referents.shape[0], 2, 2))

    grid = F.affine_grid(theta, list(referents.shape), align_corners=False)
    sampled = F.grid_sample(
        referents.to(torch.float32),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    assert torch.equal(sampled.round().to(referents.dtype), referents)


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"{name} ok")

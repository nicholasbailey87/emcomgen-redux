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

That the draw is per *image*. A single torchvision call on the whole
`(n, C, H, W)` row applies one sampled transform to every referent in it, which
varies the epoch but not the game, and leaves the speaker's and listener's views
of a shared stored image under the same transform. Twenty identical inputs must
come back twenty different ways.

That it is off by default and a passthrough when off, so every run recorded
before the keys existed reproduces from its own config.

That nothing but the pixels moves: dtype, shape, and the caller's tensor, which
is a view onto the shared in-memory store and must not be written through.

And that the corners rotation leaves behind are the background. ShapeWorld
renders on black, so `fill=0` is the background rather than a value appearing
nowhere else in the dataset -- which is exactly the kind of thing a model keys
on when the task is hard and the artefact is easy.

What is *not* tested here is label preservation, because it is a property of the
transform's arguments rather than of its code: `translate`, `scale` and `shear`
are pinned off at the call site. Against this dataset's five shapes -- circle,
ellipse, rectangle, square, triangle -- shear turns a rectangle into a
parallelogram and anisotropic scaling maps circle to ellipse and square to
rectangle. `test_the_unsafe_transform_arguments_stay_pinned` reads the source
for those three arguments instead, so that turning one into a config key has to
be a deliberate act that fails a test first.
"""

import inspect

import numpy as np
import torch

import _bootstrap  # noqa: F401

from data.generic import ConceptDataset


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


def test_the_unsafe_transform_arguments_stay_pinned():
    """
    See this module's docstring: shear and anisotropic scale destroy labels in
        this dataset, so they are arguments rather than knobs.
    """
    source = inspect.getsource(ConceptDataset._augment_geometry)
    assert "translate=[0, 0]" in source
    assert "scale=1.0" in source
    assert "shear=[0.0, 0.0]" in source
    assert "fill=0" in source


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"{name} ok")

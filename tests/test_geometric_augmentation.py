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

import parse_config
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
        self.flip = augment_flip
        self.degrees = augment_affine_degrees

    def augment(self, imgs):
        """
        `_augment_geometry` with this probe's settings.

        The two settings are arguments to the method rather than attributes of
            the dataset, because the sender and the receiver no longer share
            them; this wrapper is what keeps the tests below about the
            transform rather than about the calling convention.
        """
        return self._augment_geometry(imgs, self.flip, self.degrees)


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
    out = _Probe(True, DEGREES).augment(_referents())
    distinct = {image.numpy().tobytes() for image in out}
    # Twenty identical inputs, so any repeat is a shared draw. Two rows landing
    #     on the same transform by chance is possible but vanishingly unlikely
    #     against a continuous angle.
    assert len(distinct) == 20, f"{len(distinct)} distinct outputs from 20 rows"


def test_two_calls_draw_separately():
    """
    Two calls on the same pixels must not return the same pixels.

    Under the defaults only the listener is augmented, so this is no longer a
    statement about the pair -- it is what makes the transform a fresh draw
    every time it is asked for, which is what
    `test_the_sender_is_untouched_under_the_defaults` relies on to tell the two
    agents apart, and what a symmetric config would need to give the two views
    of a shared stored image different transforms.
    """
    np.random.seed(0)
    probe = _Probe(True, DEGREES)
    referents = _referents()
    first = probe.augment(referents)
    second = probe.augment(referents)
    assert not torch.equal(first, second)


def _game_dataset(**augment_keys):
    """
    A two-game store of identical off-centre bars, wired into a real dataset.

    The tests above reach `_augment_geometry` directly, which cannot say which
        agent's view it was called for. This one goes through `__getitem__`, so
        it reads the property the keys exist for: which agent gets augmented.

    Silhouetting and mixup are left off, so the only thing that can move a
        pixel here is the geometry.
    """
    x = np.zeros((2, 20, 3, 32, 32), dtype=np.uint8)
    x[:, :, :, 6:20, 8:12] = 200

    labels = np.zeros((2, 20), dtype=bool)
    labels[:, :10] = True

    data = {
        "x": x,
        "labels": labels,
        "langs": np.array([["red"], ["red"]], dtype=object),
        "metadata": np.zeros(2, dtype=int),
    }
    vocab = {
        "w2i": {"<PAD>": 0, "<s>": 1, "</s>": 2, "<UNK>": 3, "red": 4},
        "i2w": {0: "<PAD>", 1: "<s>", 2: "</s>", 3: "<UNK>", 4: "red"},
    }
    # `n_examples = 10` over a 20-image store: `split_spk_lis` gives each agent
    #     five positives and five negatives, which is the whole store and no
    #     overlap between the two.
    return ConceptDataset(data, vocab, n_examples=10, augment=True, **augment_keys)


def test_the_sender_is_untouched_under_the_defaults():
    """
    Receiver-only is the default, and this is where that is a fact about pixels.

    Every referent in the store is the same bar, so an un-augmented view is
        exactly that bar repeated and an augmented one is not. See DEFAULT.toml
        beside the keys for why the listener is the agent that gets it.
    """
    np.random.seed(0)
    dataset = _game_dataset(
        augment_flip_receiver=True, augment_affine_degrees_receiver=DEGREES
    )
    spk_inp, _, lis_inp, _, _, _, _ = dataset[0]

    bar = torch.from_numpy(np.zeros((3, 32, 32), dtype=np.uint8))
    bar[:, 6:20, 8:12] = 200
    assert all(torch.equal(view, bar) for view in spk_inp)
    assert not all(torch.equal(view, bar) for view in lis_inp)


def test_the_sender_keys_still_reach_the_sender():
    """The axis stays addressable: a config asking for it gets it."""
    np.random.seed(0)
    dataset = _game_dataset(
        augment_flip_sender=True, augment_affine_degrees_sender=DEGREES
    )
    spk_inp, _, lis_inp, _, _, _, _ = dataset[0]

    bar = torch.from_numpy(np.zeros((3, 32, 32), dtype=np.uint8))
    bar[:, 6:20, 8:12] = 200
    assert not all(torch.equal(view, bar) for view in spk_inp)
    assert all(torch.equal(view, bar) for view in lis_inp)


def test_it_is_a_passthrough_when_both_keys_are_off():
    referents = _referents()
    out = _Probe(False, 0.0).augment(referents)
    assert torch.equal(out, referents)


def test_the_defaults_are_off():
    signature = inspect.signature(ConceptDataset.__init__)
    for agent in ("sender", "receiver"):
        assert signature.parameters[f"augment_flip_{agent}"].default is False
        assert (
            signature.parameters[f"augment_affine_degrees_{agent}"].default
            == 0.0
        )


def test_the_callers_tensor_is_not_written_through():
    """
    `self.x[i]` is a view onto the shared store when it is held in memory, so an
        in-place transform would corrupt the dataset for every later epoch.
    """
    referents = _referents()
    before = referents.clone()
    np.random.seed(0)
    _Probe(True, DEGREES).augment(referents)
    assert torch.equal(referents, before)


def test_shape_and_dtype_survive():
    referents = _referents()
    out = _Probe(True, DEGREES).augment(referents)
    assert out.shape == referents.shape
    assert out.dtype == referents.dtype


def test_the_rotated_corners_are_background():
    np.random.seed(0)
    out = _Probe(False, DEGREES).augment(_referents())
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
    out = _Probe(True, 0.0).augment(referents)
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


def test_a_config_naming_a_retired_key_is_rejected():
    """
    The single-pair names are gone, and a config that still sets one must fail.

    A retired key is the silent failure: nothing reads it, so it merges under
        `DEFAULT.toml` and validates, and the run does the default thing under
        a config saying it does not. Here that default is *receiver-only*, so
        an un-migrated config asking for symmetric geometry would have run the
        asymmetric arm and reported it as the symmetric one -- which is the
        comparison this change exists to make.
    """
    for key, value in (("augment_flip", True), ("augment_affine_degrees", 10.0)):
        config = parse_config.get_config()
        config["cuda"] = False
        config["data"][key] = value
        try:
            parse_config.validate_config(config)
        except parse_config.InvalidConfig as error:
            assert "sender" in str(error) and "receiver" in str(error)
        else:
            raise AssertionError(f"`{key}` was accepted")


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"{name} ok")

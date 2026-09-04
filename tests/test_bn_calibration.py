"""
Tests for `train.calibrate_batch_norm` and the `train_clean` loader it reads.

Runnable without pytest:  python tests/test_bn_calibration.py

The problem the calibration pass exists for is a measurement one. Silhouetting
and the geometric augmentations are training-time only -- `shapeworld.load`
zeroes both silhouette rates and passes `augment=False` on every split but
`train` -- so every BatchNorm in the pair gathers its running mean and variance
over a silhouetted/augmented *mixture* and then applies them to clean images at
eval. docs/data.md works the consequence out for the receiver's input layer: a
fixed offset of `(mu_clean - mu_running) / sigma` on every eval activation, which
the learned affine cannot absorb because at train time that offset is zero. The
silhouette titration read shape transfer through exactly those statistics.

Four properties are checked of the pass itself, and one of the data it reads.

`n_batches = 0` must not touch a single running statistic, because that is the
default and every run on record was scored under it. A calibration that quietly
reset the statistics when it was switched off would change the meaning of every
config in the repository.

The momenta and the pair's training mode must come back, *including* when a
forward raises. The pass sets `momentum = None` on every BatchNorm and puts the
pair in train mode; leaking either into the eval passes that follow would turn a
diagnostic into a corruption of the run, and the failure would be silent.

The running mean must equal the true mean of the calibration batches. That is
what `momentum = None` buys -- PyTorch then accumulates a cumulative average
rather than an exponential one -- and it is what makes `n_batches` a readable
knob rather than a decay constant: the statistics are a function of how many
batches were seen and nothing else. It is exactly checkable, so it is checked
exactly.

The global RNG state must be unchanged. Dropout draws during calibration would
otherwise advance the run's own stochastic stream, so a run with the pass on
would diverge from the same run with it off for a reason that has nothing to do
with BatchNorm.

And `train_clean` must hand back the stored pixels: the pass is only worth
running if the images it gathers statistics from are the ones eval sees.
"""

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

import _bootstrap  # noqa: F401

import parse_config
import train
from data import shapeworld


CHANNELS = 3
SIZE = 8


class _Backbone(nn.Module):
    """
    A BatchNorm and a dropout: the two things about a backbone that this pass
        has to get right, and nothing else.

    The dropout is not decoration. It is what makes
        `test_the_global_rng_state_is_unchanged` a real test -- without a
        stochastic layer the pass would draw nothing and pass trivially -- and
        it is why `calibrate_batch_norm` forks the RNG at all.
    """

    def __init__(self, channels=CHANNELS, raises=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(0.5)
        self.raises = raises

    def forward(self, x):
        if self.raises:
            raise RuntimeError("backbone blew up")
        return self.dropout(self.bn(x)).mean((2, 3))


class _Agent(nn.Module):
    """One backbone under the attribute name its real agent uses."""

    def __init__(self, attribute, backbone):
        super().__init__()
        setattr(self, attribute, backbone)


class _Pair(nn.Module):
    def __init__(self, sender_backbone=None, receiver_backbone=None):
        super().__init__()
        self.sender = _Agent("feat_model", sender_backbone or _Backbone())
        self.receiver = _Agent(
            "feature_model", receiver_backbone or _Backbone()
        )


class _Dataset:
    """`prepare_batch` reads `name` and nothing else off the dataset."""

    def __init__(self, name="cub"):
        self.name = name


class _Loader:
    """
    A fixed list of batches, in a fixed order.

    Fixed rather than shuffled because the mean the calibration arrives at is
        being compared against the mean of a known set of images; a loader that
        drew a different subset each time could only be tested loosely.
    """

    def __init__(self, batches, name="cub"):
        self.batches = batches
        self.dataset = _Dataset(name)

    def __iter__(self):
        return iter(self.batches)


def _batch(images):
    """`(spk_inp, spk_y, lis_inp, lis_y)`, both agents seeing `images`."""
    labels = torch.zeros(images.shape[:2])
    return images, labels, images.clone(), labels.clone()


def _batches(n_batches, games=2, referents=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [
        _batch(
            torch.randn(
                games, referents, CHANNELS, SIZE, SIZE, generator=generator
            )
        )
        for _ in range(n_batches)
    ]


CONFIG = {'cuda': False}


def _stats(pair):
    return [
        (module.running_mean.clone(), module.running_var.clone())
        for module in pair.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]


def test_zero_batches_is_a_no_op():
    """
    The default, and what every run on record was scored under.

    Checked against statistics that are *not* the fresh ones, because
        `reset_running_stats` returns a BatchNorm to (0, 1) and a test starting
        from there could not tell a no-op from a reset.
    """
    pair = _Pair()
    pair.train()
    for _ in range(3):
        with torch.no_grad():
            train._flat_feature_forward(
                pair.sender.feat_model, _batches(1)[0][0]
            )

    before = _stats(pair)
    train.calibrate_batch_norm(pair, _Loader(_batches(4)), CONFIG, 0)

    for (mean, var), (was_mean, was_var) in zip(_stats(pair), before):
        assert torch.equal(mean, was_mean)
        assert torch.equal(var, was_var)


def test_the_running_mean_is_the_mean_of_the_calibration_batches():
    """
    Exactly, not approximately -- this is what `momentum = None` buys.

    Under the default momentum the running mean would be an exponential average
        that still carried whatever the train pass left behind, with weight
        `(1 - m)^n`. Then `bn_calibration_batches` would be a decay constant
        rather than a count, and the probe's question -- how many batches does
        the estimate need -- would have no answer.
    """
    batches = _batches(4, seed=1)
    pair = _Pair()
    train.calibrate_batch_norm(pair, _Loader(batches), CONFIG, len(batches))

    images = torch.cat([batch[0] for batch in batches])
    flat = images.reshape(-1, CHANNELS, SIZE, SIZE)
    expected = flat.mean((0, 2, 3))

    assert torch.allclose(pair.sender.feat_model.bn.running_mean, expected,
                          atol=1e-6)
    assert torch.allclose(pair.receiver.feature_model.bn.running_mean, expected,
                          atol=1e-6)


def test_only_the_requested_batches_are_seen():
    """
    `n_batches` bounds the pass. The loader here is longer than the request, so
        a mean over everything it holds would be a different number.
    """
    batches = _batches(6, seed=2)
    pair = _Pair()
    train.calibrate_batch_norm(pair, _Loader(batches), CONFIG, 2)

    flat = torch.cat([batch[0] for batch in batches[:2]]).reshape(
        -1, CHANNELS, SIZE, SIZE
    )
    assert torch.allclose(
        pair.sender.feat_model.bn.running_mean, flat.mean((0, 2, 3)), atol=1e-6
    )


def test_the_momenta_and_the_training_mode_are_restored():
    pair = _Pair()
    pair.eval()
    momenta = {
        id(module): module.momentum
        for module in pair.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    }

    train.calibrate_batch_norm(pair, _Loader(_batches(2)), CONFIG, 2)

    assert not pair.training
    for module in pair.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert module.momentum == momenta[id(module)]


def test_the_momenta_and_the_training_mode_are_restored_when_a_forward_raises():
    """
    The `finally` path, which is the one that matters: a pass that died with
        `momentum = None` still set would leave the *next* epoch's train pass
        accumulating a cumulative average over the whole run, and a pass that
        died in train mode would score the eval splits with dropout on. Neither
        would raise anything.
    """
    pair = _Pair(receiver_backbone=_Backbone(raises=True))
    pair.eval()

    try:
        train.calibrate_batch_norm(pair, _Loader(_batches(2)), CONFIG, 2)
    except RuntimeError as error:
        assert "backbone blew up" in str(error)
    else:
        raise AssertionError("the forward was supposed to raise")

    assert not pair.training
    for module in pair.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert module.momentum == 0.1


def test_the_global_rng_state_is_unchanged():
    """
    The pass must not move the run it is measuring. Two dropouts per batch draw
        from the global generator, so without `fork_rng` this fails.
    """
    torch.manual_seed(7)
    before = torch.get_rng_state()

    pair = _Pair()
    train.calibrate_batch_norm(pair, _Loader(_batches(3)), CONFIG, 3)

    assert torch.equal(torch.get_rng_state(), before)


def test_a_pair_with_no_batch_norm_is_left_alone():
    """An architecture with nothing to calibrate must not cost a forward."""

    class _NoNorm(nn.Module):
        def forward(self, x):
            raise AssertionError("nothing should have been forwarded")

    pair = _Pair(sender_backbone=_NoNorm(), receiver_backbone=_NoNorm())
    train.calibrate_batch_norm(pair, _Loader(_batches(2)), CONFIG, 2)


# ---------------------------------------------------------------------------
# The data the pass reads
# ---------------------------------------------------------------------------

N_GAMES = 4
N_IMG = 8
N_EXAMPLES = 4
# `shapeworld.load` pins the input at 64px and interpolates anything else, which
#     would put resampled pixels on both sides of the comparison below.
SIZE_IMG = 64


def _write_split(directory, split, seed):
    """A ShapeWorld-shaped `.npz`: coloured squares, one concept per game."""
    rng = np.random.default_rng(seed)
    imgs = np.zeros((N_GAMES, N_IMG, 3, SIZE_IMG, SIZE_IMG), dtype=np.uint8)
    for game in range(N_GAMES):
        for i in range(N_IMG):
            colour = rng.integers(64, 256, size=3, dtype=np.uint8)
            imgs[game, i, :, 2:6, 2:6] = colour.reshape(3, 1, 1)

    labels = np.zeros((N_GAMES, N_IMG), dtype=bool)
    labels[:, : N_IMG // 2] = True

    np.savez(
        os.path.join(directory, f"{split}.npz"),
        imgs=imgs,
        labels=labels,
        langs=np.array(["red"] * N_GAMES),
    )
    return imgs


def _shapeworld_datasets(directory, **data_overrides):
    config = parse_config.get_config()
    config['reference_game'] = False
    config['data'].update(
        dataset=directory,
        n_examples=N_EXAMPLES,
        load_shapeworld_into_memory=True,
        percent_novel=1.0,
        **data_overrides,
    )
    return shapeworld.load(config)


def test_train_clean_returns_the_stored_images_and_train_does_not():
    """
    The whole value of the calibration pass is that these are the pixels eval
        sees, so `train_clean` is pinned against the store itself rather than
        against `train` with the augmentations turned off -- which would pass
        even if both loaders were silhouetting.

    `train` is checked in the same breath, at the same seed, because a
        `train_clean` that matched the store while `train` did too would mean
        the augmentations were off for an unrelated reason and the test was
        measuring nothing.
    """
    with tempfile.TemporaryDirectory() as directory:
        imgs = _write_split(directory, "train", seed=0)
        _write_split(directory, "test", seed=1)

        datasets = _shapeworld_datasets(
            directory,
            silhouette_p_sender=1.0,
            silhouette_p_receiver=1.0,
            augment_flip=True,
            augment_affine_degrees=15.0,
            mixup_alpha=1.0,
        )

        assert "train_clean" in datasets

        # `split_spk_lis` at `n_examples = 4` over 8 stored images: the speaker
        #     takes 0, 1 (positive) and 4, 5 (negative), the listener 2, 3 and
        #     6, 7. With no shuffle and no reference game that mapping is fixed.
        expected_spk = torch.from_numpy(imgs[0][[0, 1, 4, 5]])
        expected_lis = torch.from_numpy(imgs[0][[2, 3, 6, 7]])

        np.random.seed(0)
        spk_inp, _, lis_inp, _, _, _, _ = datasets["train_clean"][0]
        assert torch.equal(spk_inp, expected_spk)
        assert torch.equal(lis_inp, expected_lis)

        np.random.seed(0)
        train_spk, _, train_lis, _, _, _, _ = datasets["train"][0]
        assert not torch.equal(train_spk, expected_spk)
        assert not torch.equal(train_lis, expected_lis)


def test_train_clean_shares_the_train_store():
    """
    One more DataLoader's workers, and no extra copy of the ~9.8 GB in-memory
        ShapeWorld store. If this ever stops holding, the calibration pass costs
        a second store on a 24 GB allocation.
    """
    with tempfile.TemporaryDirectory() as directory:
        _write_split(directory, "train", seed=0)
        _write_split(directory, "test", seed=1)

        datasets = _shapeworld_datasets(directory)
        assert datasets["train_clean"].x is datasets["train"].x


def test_the_eval_splits_get_no_clean_twin():
    """`train_clean` is the train games only; eval is already clean."""
    with tempfile.TemporaryDirectory() as directory:
        _write_split(directory, "train", seed=0)
        _write_split(directory, "test", seed=1)

        datasets = _shapeworld_datasets(directory)
        assert set(datasets) == {"train", "test", "train_clean"}


if __name__ == "__main__":
    for name, test in sorted(list(globals().items())):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")

"""
Tests for `Sender._record_referent_spread` in code/models/sender.py.

Runnable without pytest:  python tests/test_referent_spread.py

The speaker's referents can stop differing from each other, and when they do
every downstream column goes quiet at once: `pool_score_sd` falls because there
is nothing left to score, `contrast_within_share` falls because the attention
returns the same vector for every query, and the message collapses to one for
every game. That happened twice on `shapeworld-post-silhouette-update.csv`, at
epochs 15 and 21, and neither of those columns can say *why*.

Two causes, opposite fixes. Either the backbone's outputs collapsed in
direction, or the single scoring vector rotated somewhere the examples do not
vary. This pair of columns settles it, because it measures the referents and
never touches the scoring vector: a fall here is the backbone, and a healthy
reading here alongside a flat `pool_score_sd` is the pool.

Three properties make it readable, and each is checked below.

**It is a ratio, so the units cancel.** Rescaling the whole embedding leaves it
alone, which is what lets a Conv4 speaker and a post-norm ViT be compared at all
-- the same defect `AttentionPrototyper`'s scoring `LayerNorm` exists to remove.

**It is taken within a polarity.** Positives differ from negatives by
construction and that is signal, not spread. The decomposition is
`ExampleContrast._record_diagnostics`', so the two columns are read on one
basis.

**It falls when a common vector grows.** That is not a side effect, it is the
second thing worth catching: a contrast branch that emits one vector for the
whole game shows up as a large `contrast_share` while doing nothing, and taking
the reading either side of that stage says whether the branch is the thing doing
the homogenising.
"""

import math

import pytest
import torch
import torch.nn as nn

import _bootstrap  # noqa: F401

import models.sender as S


FEAT = 64
N_EXAMPLES = 20  # ten positive, ten negative, as the speaker is handed them
BATCH = 8


class _StubBackbone(nn.Module):
    """A deterministic image encoder, so the spread is the test's to control."""

    def __init__(self, feat_size=FEAT):
        super().__init__()
        self.final_feat_dim = feat_size
        self.projection = nn.Linear(feat_size, feat_size)

    def forward(self, images):
        return self.projection(images)


class _StubLanguageModel(nn.Module):
    """`Sender.__init__` requires one; `get_prototypes` never calls it."""

    def forward(self, *args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("get_prototypes must not speak")


def _sender(contrast=None, feat_size=FEAT, seed=0):
    torch.manual_seed(seed)
    return S.Sender(
        _StubBackbone(feat_size),
        S.AveragePrototyper(),
        _StubLanguageModel(),
        contrast=contrast,
        # Off, so `get_prototypes` is a deterministic function of its input and
        # the two readings are not separated by two different dropout masks.
        vision_dropout=0.0,
        prototype_dropout=0.0,
    )


def _examples(batch=BATCH, seed=0, scale=1.0, feat_size=FEAT):
    generator = torch.Generator().manual_seed(seed)
    return scale * torch.randn(batch, N_EXAMPLES, feat_size, generator=generator)


def _labels(batch=BATCH):
    """1.0 for the first half, 0.0 for the rest, as `Sender` requires."""
    labels = torch.zeros(batch, N_EXAMPLES)
    labels[:, : N_EXAMPLES // 2] = 1.0
    return labels


def _spread(embedded, sender=None):
    """Read the column off a throwaway attribute, so the test names it once."""
    sender = sender or _sender()
    sender._record_referent_spread(embedded, "referent_spread")
    return sender.referent_spread


# --------------------------------------------------- 1. what it measures --

def test_identical_referents_read_zero():
    """
    The collapse the column exists to name. Every example the same vector means
    nothing to pool, nothing to contrast and nothing to say.
    """
    one = torch.randn(1, 1, FEAT)
    collapsed = _spread(one.expand(BATCH, N_EXAMPLES, FEAT).contiguous())

    # Not bit-exact: subtracting a mean from the values it was computed over
    # leaves float residue, and it is ~1e-8 against live readings of order 1.
    assert collapsed == pytest.approx(0.0, abs=1e-6)


def test_it_rises_with_the_spread_it_is_named_for():
    """It has to *move*, or it says nothing about a run."""
    common = torch.randn(BATCH, 1, FEAT).expand(BATCH, N_EXAMPLES, FEAT)

    readings = [
        _spread(common + noise * _examples(seed=1))
        for noise in (0.01, 0.1, 1.0)
    ]

    assert readings == sorted(readings)
    assert readings[2] > 10.0 * readings[0]


def test_a_global_rescale_leaves_it_alone():
    """
    A ratio, so the backbone's own magnitude cancels -- which is what lets the
    column be compared across arms whose feature scales differ by fifty.
    """
    embedded = _examples(seed=2)
    reference = _spread(embedded)

    for scale in (1e-3, 7.0, 1e3):
        assert _spread(scale * embedded) == pytest.approx(reference, rel=1e-4)


def test_it_is_the_within_polarity_residual_over_the_polarity_means():
    """
    The formula, against an independently computed expectation, because every
    other test here checks a property the wrong formula could also have.

    Pins the polarity handling in particular: positives differ from negatives by
    construction and that is the concept, not a failure to collapse, so anything
    constant *within* a polarity has to leave the numerator. The prototyper pools
    each half separately and never sees the gap between them.
    """
    embedded = _examples(seed=3)
    half = N_EXAMPLES // 2

    positive, negative = embedded[:, :half], embedded[:, half:]
    means = torch.cat(
        (
            positive.mean(1, keepdim=True).expand_as(positive),
            negative.mean(1, keepdim=True).expand_as(negative),
        ),
        dim=1,
    )
    expected = (
        (embedded - means).pow(2).mean().sqrt() / means.pow(2).mean().sqrt()
    ).item()

    assert _spread(embedded) == pytest.approx(expected, rel=1e-5)

    # And the numerator itself is blind to a per-polarity offset: shifting one
    # half wholesale leaves `embedded - means` untouched, so the reading moves
    # only through the denominator it also grows.
    shifted = embedded.clone()
    shifted[:, :half] += 5.0 * torch.randn(1, 1, FEAT)

    shifted_means = torch.cat(
        (
            shifted[:, :half].mean(1, keepdim=True).expand(-1, half, -1),
            shifted[:, half:].mean(1, keepdim=True).expand(-1, half, -1),
        ),
        dim=1,
    )
    assert torch.allclose(
        shifted - shifted_means, embedded - means, atol=1e-5
    )


def test_a_common_vector_lowers_it():
    """
    The second reading, and the reason the pair brackets the contrast stage: a
    branch that emits one vector for the whole game raises `contrast_share`
    while adding nothing, and it shows up here as the referents being drowned in
    what they share.
    """
    embedded = _examples(seed=4)
    common = torch.randn(1, 1, FEAT)

    quiet = _spread(embedded + 20.0 * common)
    loud = _spread(embedded)

    assert quiet < loud / 10.0


# ------------------------------------------- 2. what reaches metrics.csv --

def test_both_columns_are_written_on_the_train_pass():
    """
    NaN until a train pass, like every other speaker diagnostic, and never
    written on eval -- which samples nothing and so measures nothing.
    """
    sender = _sender()

    assert math.isnan(sender.referent_spread)
    assert math.isnan(sender.referent_spread_backbone)

    sender.train()
    sender.get_prototypes(_examples(seed=5), _labels())

    assert sender.referent_spread > 0.0
    assert sender.referent_spread_backbone > 0.0

    sender.eval()
    trained = (sender.referent_spread, sender.referent_spread_backbone)
    with torch.no_grad():
        sender.get_prototypes(_examples(seed=6), _labels())

    assert (sender.referent_spread, sender.referent_spread_backbone) == trained


def test_the_two_columns_agree_without_the_contrast_stage():
    """
    Equal rather than one of them NaN, because equality is the informative
    reading: it is what says the gap on a contrast rung belongs to the stage.
    """
    sender = _sender().train()
    sender.get_prototypes(_examples(seed=7), _labels())

    assert sender.referent_spread == pytest.approx(
        sender.referent_spread_backbone, rel=1e-6
    )


def test_the_contrast_stage_separates_them():
    """
    With the stage on and its gate driven open onto a branch, the post-stage
    reading has to move away from the backbone's. Otherwise the pair could not
    attribute a collapse to the stage, which is the whole point of taking two.
    """
    contrast = S.ExampleContrast(FEAT, d_model=32, heads=4, self_attention_dropout=0.0)
    sender = _sender(contrast=contrast).train()

    with torch.no_grad():
        contrast.contrast_gate.fill_(50.0)

    sender.get_prototypes(_examples(seed=8), _labels())

    assert sender.referent_spread != pytest.approx(
        sender.referent_spread_backbone, rel=1e-3
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

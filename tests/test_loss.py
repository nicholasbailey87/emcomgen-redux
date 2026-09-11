"""
`train.hinge_loss` and the top-level `loss` sentinel `parse_config` validates.

The key exists because `BCEWithLogitsLoss` has a trivial optimum that a listener
with nothing to read descends into. With an uninformative message, scoring every
candidate at zero is BCE's *strict* minimum, so the volume in front of the score
has a downhill path to `ln 2` and takes it: the 2026-09-10 ShapeWorld ViT run
slid `train_score_scale` 0.998 -> 0.199 over thirty epochs while
`bilinear_weight_norm` moved 18.5 -> 18.2, which is one scalar carrying the whole
collapse and the objective asking it to.

A hinge has no such basin, and that is the property this file pins:

    * **Squashing the scores cannot reduce the loss.** Inside the margin the
      loss is `margin - mean(t * score)`, exactly `margin` whenever `t` and
      `score` are uncorrelated *and independent of their scale*. Going quiet is
      flat rather than rewarded.
    * **A candidate already right by a full margin contributes exactly zero.**
      This is the speculative half -- the hope that once the colour-solvable
      games are solved the gradient concentrates on the games colour cannot do
      -- and it is also the failure to watch for, since a comfortable listener
      then sends the speaker nothing.

`test_squashing_the_scores_does_not_reduce_the_hinge` is the one that matters,
and it is written against BCE in the same breath so the contrast is in the
assertion rather than in this docstring.

The decision threshold does not move: `train.py` decides on `lis_scores > 0`
under either objective, so `per_game_accuracy` is untouched and `train_acc` and
topsim stay comparable across the two. `train_loss` does not -- it reads against
`ln 2` = 0.6931 on one arm and against 1.0 on the other.

Runnable without pytest:  python tests/test_loss.py
"""

import _bootstrap  # noqa: F401

import pytest
import torch
import torch.nn.functional as F

import parse_config
import train

from _bootstrap import all_rungs, rung

RUNGS = all_rungs()

LN_2 = 0.6931471805599453


def _scores_and_labels(n=8):
    """Ten-candidate games, positives first, as `split_spk_lis` builds them."""
    labels = torch.zeros(n, 10)
    labels[:, :5] = 1.0
    return torch.zeros(n, 10, requires_grad=True), labels


# --------------------------------------------------------------------------
# What the objective does
# --------------------------------------------------------------------------


def test_the_margin_is_one():
    """
    Fixed rather than configurable, and 1.0 against the listener's calibrated
        opening score of `1 / sqrt(3)` = 0.577, so every candidate opens inside
        the margin and returns gradient at epoch 0.
    """
    assert train.HINGE_MARGIN == 1.0
    assert 3 ** -0.5 < train.HINGE_MARGIN


def test_silence_costs_the_margin_rather_than_ln_2():
    scores, labels = _scores_and_labels()
    scores = scores.detach()
    assert float(train.hinge_loss(scores, labels)) == pytest.approx(
        train.HINGE_MARGIN
    )
    # The number the same listener pays under BCE, for the contrast.
    assert float(
        F.binary_cross_entropy_with_logits(scores, labels)
    ) == pytest.approx(LN_2)


def test_squashing_the_scores_does_not_reduce_the_hinge():
    """
    The property the key exists for. Scores uncorrelated with the labels, scaled
        down by successive factors of ten: BCE falls towards `ln 2` every time
        and a hinge does not move at all.
    """
    torch.manual_seed(0)
    _, labels = _scores_and_labels()
    scores = torch.randn(8, 10) * 2.0

    hinges = []
    bces = []
    for volume in (1.0, 0.1, 0.01, 0.001):
        hinges.append(float(train.hinge_loss(scores * volume, labels)))
        bces.append(
            float(F.binary_cross_entropy_with_logits(scores * volume, labels))
        )

    # Every candidate is inside the margin from the second entry on, where the
    #     loss is `margin - mean(t * score)` and the mean is shrinking towards
    #     zero -- so the hinge converges *up* to the margin, never below it.
    assert all(h >= train.HINGE_MARGIN - 1e-6 for h in hinges[1:]), hinges
    assert hinges[-1] == pytest.approx(train.HINGE_MARGIN, abs=1e-3)

    # BCE, on the same tensors, is monotonically rewarded for the same move.
    assert bces == sorted(bces, reverse=True), bces
    assert bces[-1] == pytest.approx(LN_2, abs=1e-3)
    assert bces[0] > bces[-1] + 0.1


def test_a_candidate_past_the_margin_contributes_no_gradient():
    scores = torch.tensor([[2.0, -2.0, 0.5]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 1.0]])

    train.hinge_loss(scores, labels).backward()

    # The first two are right by more than the margin; the third is right but
    #     only by 0.5, so it is still inside it and still being pushed.
    assert float(scores.grad[0, 0]) == 0.0
    assert float(scores.grad[0, 1]) == 0.0
    assert float(scores.grad[0, 2]) < 0.0


def test_the_gradient_at_zero_is_constant_and_signed_by_the_label():
    """
    Constant magnitude, unlike BCE's `sigmoid(s) - y`, which is what keeps a
        starved speaker getting the same push from every game.
    """
    scores, labels = _scores_and_labels()
    train.hinge_loss(scores, labels).backward()

    size = 1.0 / scores.numel()
    assert torch.allclose(
        scores.grad, torch.where(labels > 0.5, -size, size), atol=1e-7
    )


def test_a_wrong_candidate_costs_more_than_a_silent_one():
    """
    So there is no shortcut in confidence either: the loss grows without bound
        on the wrong side, which is what stops the scores running away.
    """
    labels = torch.tensor([[1.0]])
    silent = float(train.hinge_loss(torch.tensor([[0.0]]), labels))
    wrong = float(train.hinge_loss(torch.tensor([[-3.0]]), labels))
    right = float(train.hinge_loss(torch.tensor([[3.0]]), labels))

    assert right == 0.0
    assert wrong > silent > right


def test_it_agrees_with_the_decision_threshold_at_zero():
    """
    `per_game_accuracy` thresholds at `lis_scores > 0` under either objective,
        so the loss must be scoring the same side of the same origin.
    """
    scores = torch.tensor([[0.2, -0.2]])
    labels = torch.tensor([[1.0, 0.0]])

    assert train.per_game_accuracy(scores, labels, False) == 1.0
    # Both are on the right side of the origin and inside the margin, so the
    #     loss is below the margin and above zero.
    loss = float(train.hinge_loss(scores, labels))
    assert 0.0 < loss < train.HINGE_MARGIN


# --------------------------------------------------------------------------
# What the config is allowed to say
# --------------------------------------------------------------------------


def _validate(loss="hinge", **data):
    """`validate_config` over a whole config, with `loss` and `[data]` set."""
    config = parse_config.get_config(rung(RUNGS[0]))
    config["loss"] = loss
    config["data"].update(data)
    return parse_config.validate_config(config)


@pytest.mark.parametrize("loss", ["hinge_loss", "bce_with_logits", "", None, True])
def test_an_unknown_objective_is_rejected(loss):
    with pytest.raises(parse_config.InvalidConfig):
        _validate(loss=loss)


@pytest.mark.parametrize("loss", sorted(parse_config.LOSSES))
def test_every_sentinel_in_the_table_is_accepted(loss):
    _validate(loss=loss, mixup_blends_classes=False)


def test_a_hinge_beside_class_blending_mixup_is_rejected():
    """
    Rejected rather than ignored, and this is the coupling that matters: a
        continuous target is readable by BCE and by nothing else. A hinge has a
        direction and a minimum magnitude and no target score at all, so there
        is no answer for a candidate labelled 0.7 -- weighting by the label's
        confidence would push it *less hard to the same full-confidence target*,
        which is arithmetic rather than meaning.
    """
    with pytest.raises(parse_config.InvalidConfig):
        _validate(loss="hinge", mixup_alpha=1.0, mixup_blends_classes=True)


def test_a_hinge_beside_within_polarity_mixup_is_accepted():
    """The point of `mixup_blends_classes`: the labels stay hard."""
    _validate(loss="hinge", mixup_alpha=1.0, mixup_blends_classes=False)


def test_a_hinge_beside_mixup_switched_off_is_accepted():
    _validate(loss="hinge", mixup_alpha=0.0, mixup_blends_classes=True)


def test_bce_is_free_to_blend_classes():
    _validate(loss="bce", mixup_alpha=1.0, mixup_blends_classes=True)


def test_a_hinge_beside_the_cross_entropy_branch_is_rejected():
    """
    The xent branch scores a single target and never reaches the per-candidate
        criterion `loss` selects, so the key would sit there unread.
    """
    config = parse_config.get_config(rung(RUNGS[0]))
    config["loss"] = "hinge"
    config["reference_game"] = True
    config["reference_game_xent"] = True

    with pytest.raises(parse_config.InvalidConfig):
        parse_config.validate_config(config)


@pytest.mark.parametrize("blends", ["false", 0, None])
def test_a_non_boolean_blend_flag_is_rejected(blends):
    with pytest.raises(parse_config.InvalidConfig):
        _validate(loss="bce", mixup_blends_classes=blends)


# --------------------------------------------------------------------------
# What the defaults say
# --------------------------------------------------------------------------


def test_the_default_is_hinge_and_within_polarity_mixup():
    """
    Hinge since 2026-09-11, on `experiments/hinge_vs_bce/`:
        `train_acc_md_shape` 0.726 against BCE's 0.525 at an identical colour
        accuracy, where 0.525 is the colour-only minimum every ViT arm of
        `lr_sweep_2_sender_vit` sat in. Within-polarity mixup because the hinge
        cannot read a continuous target, which is the coupling checked above.
    """
    config = parse_config.get_config()
    assert config["loss"] == "hinge"
    assert config["data"]["mixup_blends_classes"] is False


@pytest.mark.parametrize("config_file", RUNGS)
def test_every_rung_names_an_objective_in_the_table(config_file):
    assert parse_config.get_config(rung(config_file))["loss"] in parse_config.LOSSES


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))

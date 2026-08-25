"""
Tests for the listener's score scale in code/models/receiver.py.

Runnable without pytest:  python tests/test_score_scale.py

The listener is `ReceiverGRULM + BilinearDiscriminator` or
`ReceiverCrossAttentionLM + AttentionDiscriminator`; this file calls those the
bilinear arm and the attention arm, and follows each end to end because volume
is a property of the whole path. The two modules they were split out of are
named below where the history is theirs.

Both used to let the architecture set how loudly the listener stated a
conclusion, and both have now been stopped from doing it. That is the same
defect `e3fcabd` fixed in three places on the speaker, arriving one module at a
time: whatever multiplies the score also multiplies every gradient in the pair,
so a listener that quietens starves the machinery that would make it worth
listening to.

`BilinearGRUComparer`, now the bilinear arm, scored a referent by a raw dot
product on unnormalised
backbone output, so on ViT2 the size was set by an `nn.BatchNorm1d` at the end
of broccoli's classification head and on ResNet18 by the trunk's own
normalisation -- per batch, and differently at eval.

`TransformerCrossAttentionComparer`, now the attention arm, read its score off a
bare
`nn.Linear(d_model, 1)`, which made one vector both the *direction* the head
reads out and the *volume* it reads at. BCE reduces a loss it cannot otherwise
reduce by becoming less confident, and that pressure is first-order where
learning a useful direction is not. On CUB the volume collapsed: scores from sd
0.42 to sd 0.016 inside one epoch, `train_loss` pinned at `ln 2 + 2e-5` for
thirty.

The first answer was the same on both: normalise everything that could set the
score's magnitude -- both operands of the dot product on one, the readout
direction and its input on the other -- and leave a single `log_score_scale`
opening at 1.0.

That held on the bilinear arm, which still has it, and failed on the other
one. `issue.csv` is the second round: rungs 11 and 12 sat at `train_loss` = ln 2
and `train_acc` = 0.4998 for thirty epochs while `score_scale` slid 0.914 ->
0.273, monotone, never recovering. Making the collapse legible had not made it
stop. Nor could accuracy see it -- `train.py` reads the decision as
`scores > 0`, and a strictly positive scale leaves `s * (u + b) > 0` equivalent
to `u + b > 0`, so the loss walked to ln 2 without a single prediction changing.

The third answer was to remove the volume parameter altogether: `decision`
called directly, its output standardised by a `BatchNorm1d(1, affine=False)`
over the flattened batch, a fixed `decision_gain` setting the volume once. It
closed the collapse exactly as designed, and it stopped every rung carrying this
comparer -- 11, 12, 13 and 14 -- from learning at all, at 0.5000 accuracy for
thirty epochs apiece.

So the readout is a plain `nn.Linear(d_model, 1)` again, on a layer-normed
input, with the collapse route open and watched rather than closed.
`diagnostics/bootstrap_probe.py` is why: the whole pair with only the vision
models stubbed out, where at the config's own 1e-4 the bilinear baseline reaches
accuracy 1.000, the standardised readout 0.606, and the plain readout 0.863 --
the last taking off by the same route the baseline does, `polarity_separation`
crossing 6-8 and the speaker's logit scale traversing behind it. A listener that
cannot go quiet while the message is still noise never lets the speaker learn to
send one.

The fourth answer put the volume in a named scalar on each arm --
`log_score_scale` on the bilinear one, `log_mix_scale` downstream of the
attention arm's standardised mix -- both on an elevated learning rate so that a
lone scalar could move fast enough to matter.

**The fifth, which this file now tests, took both scalars away again and gave
the volume back to the weight matrices.** Not by pinning anything: the reason
is that the listener spent that elevated mobility squashing its own logits.
`score_scale` fell 0.9021 -> 0.3731 on rung 09 and 0.9377 -> 0.4072 on rung 11
across thirty epochs, monotone, never returning -- correct behaviour on a
message carrying nothing, since BCE's minimiser is p=0.5 everywhere, but the
scalar factors to the front of the score, so shrinking it multiplies down every
gradient going back through the message and into the speaker. A fast scalar is
a cheap single-knob route to going quiet.

`bilinear`'s `message_layer_norm` therefore moved to its *input*, and
`standardise` left the attention arm's forward path for its telemetry block. So
the volume is `bilinear.weight` and `decision.weight`: matrices, learning
direction and magnitude together, which is what jayelm's unnormalised `compare`
has and what has no equivalently cheap collapse. The listener can still go
quiet -- that freedom is what the third answer proved is required -- it just
has to move a whole matrix to do it.

The tests below split accordingly: shared properties stay parametrised over both
arms, the bilinear weight's new reach has a section of its own, and so does the
attention readout. What is new and worth watching is the property standardising
used to guarantee and no longer does -- a branch *can* now go quiet alone --
which is why `mix_share` is reported beside `mix_alpha`.
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401

import models.builder
import models.receiver as R
import parse_config

from _bootstrap import CONFIG_DIR, build_listener, config_section, rung

REFERENT_DIM = 512
BATCH, N_OBJ, SEQ = 8, 20, 7


def _comparer(referent_dim=REFERENT_DIM, **overrides):
    """
    The bilinear arm: `ReceiverGRULM` feeding `BilinearDiscriminator`, composed
        the way `Receiver` composes them. Overrides go to the language model,
        which is where every key this arm reads lives -- `BilinearDiscriminator`
        takes nothing from its own config table.
    """
    return build_listener(
        "ReceiverGRULM",
        "BilinearDiscriminator",
        referent_dim,
        language_model_overrides=overrides or None,
    )


CROSS_RUNG = "15_shapeworld_receiver_cross_attention_lm.toml"

# The keys that belong to the discriminator's table rather than the language
#     model's, so `_cross_comparer` can take one flat kwargs like the builder it
#     replaced. `layers`, `alpha` and `beta` are deliberately absent: both
#     tables carry them and mean different stacks, so a test that wants one has
#     to say which. See test_residual_scaling.py.
_DISCRIMINATOR_KEYS = frozenset(
    {
        "d_model", "heads", "ff_inner_size", "stochastic_depth",
        "self_attention_dropout", "cross_attention_dropout",
        "ff_inner_dropout", "ff_outer_dropout", "activation",
        "pre_norm", "post_norm", "knocking_heads",
        "depthwise_linear_stochastic_depth",
        "mix_floor", "mix_logit_init",
    }
)


def _cross_comparer(referent_dim=REFERENT_DIM, dropout=0.0, **overrides):
    """
    The attention arm: `ReceiverCrossAttentionLM` feeding
        `AttentionDiscriminator`.

    Built from rung 11 rather than from DEFAULT, which cannot construct the
        encoder: DEFAULT's `[receiver_language_model] d_model = 1024` is the
        GRU's width and does not divide its `heads = 5`. See the note beside
        `d_model` in DEFAULT.toml.

    `dropout` is `[receiver] dropout` and defaults to off, because these are
        tests of a deterministic property and a resampled mask between two
        calls would be measuring dropout.
    """
    discriminator_overrides = {
        key: value for key, value in overrides.items()
        if key in _DISCRIMINATOR_KEYS
    }
    return build_listener(
        "ReceiverCrossAttentionLM",
        "AttentionDiscriminator",
        referent_dim,
        config_file=rung(CROSS_RUNG),
        dropout=dropout,
        language_model_overrides=overrides or None,
        discriminator_overrides=discriminator_overrides or None,
    )


# Every property below the first section holds of both arms, and is
#     parametrised over them rather than written twice. Where a mechanism
#     differs the test is in the arm's own section further down.
BOTH = pytest.mark.parametrize(
    "build", [_comparer, _cross_comparer], ids=["bilinear", "cross_attention"]
)


def _inputs(listener, referent_scale=1.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    referents = referent_scale * torch.randn(
        BATCH, N_OBJ, listener.referent_embedding_size, generator=generator
    )
    messages = torch.randn(
        BATCH,
        getattr(listener, "message_length", SEQ),
        listener.token_embedding_size,
        generator=generator,
    )
    return referents, messages


def _labels():
    labels = torch.zeros(BATCH, N_OBJ)
    labels[:, : N_OBJ // 2] = 1.0
    return labels


# --------------------------------------------------------------------------
# The norms, and which of them carry an affine.
# --------------------------------------------------------------------------

def test_neither_bilinear_operand_norm_has_an_affine():
    discriminator = _comparer().discriminator
    assert discriminator.referent_layer_norm.weight is None
    assert discriminator.referent_layer_norm.bias is None
    assert discriminator.message_layer_norm.weight is None
    assert discriminator.message_layer_norm.bias is None


def test_the_cross_attention_norms_that_must_be_affine_free_are():
    """
    `referent_layer_norm` has no affine because one there is a second route to
        score magnitude, and because the adapter above it is `bias=False`
        precisely so this norm can divide the backbone's scale out exactly.

    The last thing before the readout is instead the referent stack's own
        post-norm, an `RMSNorm` carrying broccoli's default learnable gain --
        not ours to turn off, since the same class is used by every stack in the
        repo. That gain is a route to global score magnitude and it is
        deliberately open: `decision_spread` watches it, and two attempts to
        close it are in docs/anecdotes.md. What the RMSNorm still does
        structurally is equalise the candidates against *each other*, which is
        the part that has to hold, so its affine is asserted here as present
        rather than absent.
    """
    listener = _cross_comparer()

    assert listener.language_model.referent_layer_norm.weight is None
    assert listener.discriminator.referent_layer_norm.weight is None
    assert listener.discriminator.memory_layer_norm.weight is None
    assert (
        listener.discriminator.referent_decoder.blocks[-1].post_mlp_norm.weight
        is not None
    )


def test_the_referent_adapter_has_no_bias():
    """
    Load-bearing for the invariance below, not tidiness: the norm after it can
        only remove the vision model's scale exactly if what reaches it is
        homogeneous in the input. `W(cx) = cW(x)` gives `LN(W(cx)) = LN(W(x))`;
        `W(cx) + b` does not.
    """
    listener = _cross_comparer()
    # One per slot, because each owns its own projection; see
    #     test_receiver_slots.py for why they are not shared.
    assert listener.language_model.referent_adapter.bias is None
    assert listener.discriminator.referent_adapter.bias is None


# --------------------------------------------------------------------------
# What the score may not depend on.
# --------------------------------------------------------------------------

# Seven orders of magnitude, and the span is chosen from the measured floor
#     below rather than picked round: see
#     `test_the_layer_norm_epsilon_floor_sits_below_anything_a_backbone_emits`.
@BOTH
@pytest.mark.parametrize("referent_scale", [1e-3, 0.01, 1.0, 100.0, 1e4])
def test_scores_are_independent_of_the_referent_magnitude(build, referent_scale):
    """
    A backbone emitting features a hundred times larger must not thereby make
        its listener a hundred times more confident. This is the property the
        whole change exists for.
    """
    comparer = build().eval()
    referents, messages = _inputs(comparer, referent_scale=referent_scale)
    with torch.no_grad():
        scores = comparer(referents, messages)

    reference = build().eval()
    with torch.no_grad():
        expected = reference(*_inputs(reference, referent_scale=1.0))

    # Absolute as well as relative, because the cross-attention comparer's
    #     scores are deliberately near unit variance, so some of them are small
    #     and a purely relative bound would be measuring float32 noise on those.
    #     3e-6 is the worst seen across this span; a real failure is 4.5%.
    assert torch.allclose(scores, expected, rtol=1e-3, atol=1e-5)


@BOTH
def test_the_layer_norm_epsilon_floor_sits_below_anything_a_backbone_emits(build):
    """
    `F.layer_norm` divides by `sqrt(var + eps)`, so scale invariance holds only
        while the incoming variance is large against `LAYER_NORM_EPS`. Below
        that the normaliser quietly stops normalising and the score's magnitude
        goes back into the backbone's hands, which is the trap
        `receiver.LAYER_NORM_EPS` documents.

    Both comparers give out at the same place -- around referent RMS 1e-3,
        where the variance is 1e-6 -- and at 1e-4 both are 1.5e-4 adrift. That
        is not float32 rounding: it reproduces identically in float64.

    ViT2 emits RMS 0.23 and ResNet18 is the same order, so the floor is two and
        a half orders below anything real and this test exists to say so with a
        number rather than to guard a live risk. If a backbone ever did emit
        features that small, the fix is a smaller epsilon, not a wider
        tolerance.
    """
    comparer = build().eval()
    reference = build().eval()
    with torch.no_grad():
        expected = reference(*_inputs(reference, referent_scale=1.0))
        intact = comparer(*_inputs(comparer, referent_scale=1e-3))
        given_out = comparer(*_inputs(comparer, referent_scale=1e-5))

    assert (intact - expected).abs().max().item() < 1e-5
    assert (given_out - expected).abs().max().item() > 1e-3


@BOTH
def test_the_referent_norm_is_not_a_global_rescale(build):
    """
    It normalises each candidate separately, so it can and must change which
        object wins. Enlarging one candidate alone must not promote it.
    """
    comparer = build().eval()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        before = comparer(referents, messages)

    inflated = referents.clone()
    inflated[:, 3, :] *= 50.0
    with torch.no_grad():
        after = comparer(inflated, messages)

    assert torch.allclose(before, after, atol=1e-4)


def test_an_unnormalised_referent_would_have_been_promoted():
    """
    The counterfactual the test above is worth checking against: the same
        inflation, scored the way `BilinearGRUComparer` scored it before this
        change.
    """
    listener = _comparer().eval()
    referents, messages = _inputs(listener)
    inflated = referents.clone()
    inflated[:, 3, :] *= 50.0

    with torch.no_grad():
        message_repr = listener.language_model(messages, referents)
        projected = listener.discriminator.bilinear(message_repr.mean(1))
        raw = torch.einsum("ijh,ih->ij", (inflated, projected))

    assert raw[:, 3].abs().mean() > 10.0 * raw.abs().mean()


def test_an_unnormalised_referent_would_have_hijacked_the_value_mixture():
    """
    The cross-attention counterfactual, and a different mechanism from the one
        above. broccoli RMS-normalises Q and K per head (`project_qkv`) and the
        attention *output* (`out_norm`), so the logits and a uniformly louder
        backbone are both already handled -- but V is normalised nowhere, and at
        the first stage the referents are K *and* V. The output is a
        magnitude-weighted mixture, so one outsized candidate captures it for
        every message token, and no downstream norm can undo an average that
        has already been taken.
    """
    listener = _cross_comparer(referent_dim=320).eval()
    language_model = listener.language_model
    referents, messages = _inputs(listener)
    inflated = referents.clone()
    inflated[:, 3, :] *= 50.0

    with torch.no_grad():
        adapted = language_model.referent_adapter(referents)
        adapted_inflated = language_model.referent_adapter(inflated)
        encoded = language_model.message_adapter(messages)

        def stage_one(values):
            return language_model.message_decoder.blocks[0].cross_attention(
                encoded, values, values
            )

        raw = stage_one(adapted)
        raw_inflated = stage_one(adapted_inflated)
        normed = stage_one(language_model.referent_layer_norm(adapted))
        normed_inflated = stage_one(
            language_model.referent_layer_norm(adapted_inflated)
        )

    moved_raw = ((raw_inflated - raw).norm(dim=-1) / raw.norm(dim=-1)).mean()
    moved_normed = (
        (normed_inflated - normed).norm(dim=-1) / normed.norm(dim=-1)
    ).mean()

    assert moved_raw > 0.5           # measured 1.16 -- the mixture is captured
    assert moved_normed < 1e-4       # measured 0.0


# --------------------------------------------------------------------------
# The unit, and where the scale opens.
# --------------------------------------------------------------------------

def test_the_bilinear_score_opens_at_one_over_root_three():
    """
    The opening is now a consequence rather than a setting. With
        `message_layer_norm` ahead of `bilinear`, the default `nn.Linear` init
        -- `U(+/- 1/sqrt(fan_in))`, variance `1/(3 * fan_in)` -- takes a
        unit-variance input to an output of variance `1/3` whatever `fan_in`
        is. The referent operand is normalised, and `/sqrt(referent_dim)`
        cancels the dot product's width, so the score opens at `1/sqrt(3)`.

    0.577 rather than the 1.0 `log_score_scale` used to set. The constant is
        cosmetic; what the calibration was for is that it does not move with
        either width, which `test_the_untrained_score_opens_at_a_width_...`
        below pins across both.
    """
    listener = _comparer().eval()
    with torch.no_grad():
        scores = listener(*_inputs(listener))

    assert scores.std().item() == pytest.approx(1.0 / math.sqrt(3.0), abs=0.08)


@BOTH
@pytest.mark.parametrize("referent_dim", [320, 512])
def test_the_untrained_score_opens_at_a_width_independent_magnitude(
    build, referent_dim
):
    """
    The property that survives both designs: the opening confidence is the same
        whichever backbone the rung mounts, rather than growing with its width.

    On `BilinearGRUComparer` that is what `1/sqrt(referent_embedding_size)`
        buys against a scale opening at 1.0. On the cross-attention comparer it
        is the referent stack's last post-norm, which puts every candidate at
        unit RMS whatever the backbone emitted, so `decision`'s default init
        sets the opening and the referent width does not -- see
        `test_the_readout_opens_below_a_confident_wrong_answer`.
    """
    listener = build(referent_dim=referent_dim).eval()
    with torch.no_grad():
        scores = listener(*_inputs(listener))

    assert 0.3 < scores.std().item() < 3.0


@BOTH
@pytest.mark.parametrize("referent_dim", [320, 512])
def test_untrained_bce_opens_within_reach_of_ln_2(build, referent_dim):
    """
    The reason the opening confidence matters. A listener that opens by
        shouting wrong answers makes muting the fast descent direction, which
        is the state `e3fcabd` was written about.

    Both classes now open close to ln 2 from either side -- bilinear just
        under, cross-attention at ~0.73 just over. They briefly did not: a fixed
        `decision_gain` of 2.0 opened the cross-attention comparer at 1.07,
        deliberately worse than chance on the argument that sitting at ln 2
        should never be free. That argument is what
        `test_the_readout_opens_below_a_confident_wrong_answer` records the
        refutation of.
    """
    listener = build(referent_dim=referent_dim).eval()
    with torch.no_grad():
        scores = listener(*_inputs(listener))

    loss = F.binary_cross_entropy_with_logits(scores, _labels()).item()

    assert loss < 2.0 * math.log(2.0)


# --------------------------------------------------------------------------
# What carries the bilinear volume now. `bilinear.weight` does, on both arms:
#     the class no longer takes a `score_scale` argument, and inside
#     `AttentionDiscriminator` its output is no longer standardised, so the
#     same weight reaches the mixed score too.
# --------------------------------------------------------------------------

def test_no_scalar_volume_survives_anywhere_on_the_listener():
    """
    The parameter is gone rather than frozen, on both arms. A frozen one would
        still be matched by `split_out_parameter`'s suffix and still report a
        column, which is the state this was in when it was built with
        `score_scale=False`.
    """
    for build in (_comparer, _cross_comparer):
        named = dict(build().named_parameters())
        assert not [
            name for name in named
            if "log_score_scale" in name or "log_mix_scale" in name
        ], sorted(named)


def test_the_bilinear_weight_now_reaches_the_score_magnitude():
    """
    The inversion. `message_layer_norm` used to sit on `bilinear`'s *output*,
        so growing `W` turned the comparison without making it louder and
        `log_score_scale` was the only route to volume. The norm is now on
        `bilinear`'s input, so `W` sets direction and volume together -- which
        is what jayelm's unnormalised `compare` gets for free, and what the
        listener has no single cheap way to collapse.
    """
    listener = _comparer().eval()
    referents, messages = _inputs(listener)
    with torch.no_grad():
        before = listener(referents, messages)
        listener.discriminator.bilinear.weight.mul_(37.0)
        after = listener(referents, messages)

    assert torch.allclose(after, 37.0 * before, atol=1e-4)


def test_scaling_the_bilinear_weight_cannot_change_the_decision():
    """
    `train.py` reads the decision as `scores > 0` and the reference-game branch
        as an argmax, and neither may move with the listener's confidence. A
        positive rescale multiplies an operand shared across the objects of a
        game, so it cannot change which object wins -- only how loudly the
        listener says so.

    Which is also why a volume collapse is invisible in `train_acc`, whether it
        runs through a scalar or through a matrix. It is why the volume needs a
        column of its own; `train_bilinear_weight_norm` is now that column.
    """
    listener = _comparer().eval()
    referents, messages = _inputs(listener)
    with torch.no_grad():
        quiet = listener(referents, messages)
        listener.discriminator.bilinear.weight.mul_(37.0)
        loud = listener(referents, messages)

    assert torch.equal(quiet > 0, loud > 0)
    assert torch.equal(quiet.argmax(1), loud.argmax(1))


def test_scaling_the_bilinear_weight_does_change_the_loss():
    """
    Which is what makes it a volume: BCE is not scale-invariant, so this is the
        listener's control over its own confidence. It is also the exposure --
        the same asymmetry that makes quietening a first-order descent
        direction, only now it costs a whole matrix to move rather than one
        scalar on an elevated learning rate.
    """
    listener = _comparer().eval()
    referents, messages = _inputs(listener)
    labels = _labels()

    with torch.no_grad():
        quiet = F.binary_cross_entropy_with_logits(
            listener(referents, messages), labels
        ).item()
        listener.discriminator.bilinear.weight.mul_(37.0)
        loud = F.binary_cross_entropy_with_logits(
            listener(referents, messages), labels
        ).item()

    assert loud > quiet


def test_the_bilinear_weight_receives_gradient():
    listener = _comparer()
    referents, messages = _inputs(listener)

    F.binary_cross_entropy_with_logits(
        listener(referents, messages), _labels()
    ).backward()

    weight = listener.discriminator.bilinear.weight
    assert weight.grad is not None
    assert weight.grad.abs().sum().item() > 0.0


def test_reset_parameters_returns_the_volume_to_its_opening():
    discriminator = _comparer().discriminator
    opening = discriminator.bilinear.weight.norm().item()
    with torch.no_grad():
        discriminator.bilinear.weight.mul_(37.0)
    discriminator.reset_parameters()

    # Not the same draw, so the norm rather than the tensor: what has to come
    #     back is the scale of the opening, which is what `fan_in` fixes.
    assert discriminator.bilinear.weight.norm().item() == pytest.approx(
        opening, rel=0.1
    )


# --------------------------------------------------------------------------
# The readout. `AttentionDiscriminator` only.
#
# `decision` is a plain `nn.Linear(d_model, 1)` on a layer-normed input, which
#     is what it was before two attempts to take the volume out of the
#     listener's hands -- `log_score_scale`, then a fixed-gain BatchNorm. The
#     second closed the collapse it was written for and stopped four rungs
#     learning at all.
#
# What follows it is the mix, and the mix standardises this readout. That is
#     not the fixed gain coming back: the volume moved to `log_mix_scale`
#     downstream of both paths rather than being pinned, and the bilinear path
#     carries the decision through the opening so nothing has to be confident
#     early. What is closed is only the escape of turning the attention path
#     down instead of learning it.
#
# Tests here run in train mode because everything downstream of the readout
#     (dropout, stochastic depth) is mode-dependent, and because that is the
#     mode the collapse happens in.
# --------------------------------------------------------------------------

def _quiet_cross_comparer(**overrides):
    """
    The attention arm with every dropout off, so that two calls on the same
        input differ only through what the test changed. Without this the
        train-mode masks are resampled between calls and an exact-invariance
        assertion is measuring dropout.

    Note the keys reach both slots: `_cross_comparer` routes them by name, and
        the attention dropouts are in both tables under the same names.
    """
    return _cross_comparer(
        referent_dim=320,
        dropout=0.0,
        cross_attention_dropout=0.0,
        self_attention_dropout=0.0,
        ff_inner_dropout=0.0,
        ff_outer_dropout=0.0,
        stochastic_depth=0.0,
        **overrides,
    ).train()


def test_the_readout_is_a_plain_linear_layer():
    """
    The design, stated as what it is not.

    No `log_score_scale`: that was attempt one at holding the volume, and it
        collapsed on rungs 11 and 12 exactly as it did on the bare layer. No
        `score_norm` and no `decision_gain`: that was attempt two, which closed
        the collapse and cost the run its ability to bootstrap -- with the
        readout standardised, rung 12 sits at accuracy 0.606 in
        `diagnostics/bootstrap_probe.py` where the same module with a plain
        readout reaches 0.863 and the bilinear baseline reaches 1.000.

    The bias has been live, then dead, then live again, and it is live now. It
        was dropped once because a mean subtraction downstream absorbed any
        constant it could add, restored when nothing subtracted a mean, and
        dead again while `standardise` ran in the forward path. `standardise`
        has now left the forward path for the telemetry block, so it adds a
        real constant to the score once more -- redundantly with `mix_bias`,
        which is the other one, and harmlessly, because a constant common to
        every candidate in a game cannot change which one wins.
    """
    discriminator = _cross_comparer(referent_dim=320).discriminator

    assert not hasattr(discriminator, "score_norm")
    assert not hasattr(discriminator, "decision_gain")
    assert isinstance(discriminator.decision, torch.nn.Linear)
    assert discriminator.decision.out_features == 1
    assert discriminator.decision.bias is not None

    # No scalar volume on either path or downstream of them: `decision.weight`
    #     and the bilinear path's own weight are what set the magnitude now.
    assert not hasattr(discriminator.bilinear, "log_score_scale")
    assert not hasattr(discriminator, "log_mix_scale")
    assert not hasattr(discriminator, "mix_scale")


def test_the_readout_opens_below_a_confident_wrong_answer():
    """
    What the opening magnitude has to be, now that nothing sets it by
        construction.

    The standardised readout opened at exactly `decision_gain`, and at 2.0 that
        put untrained BCE at 1.07 -- deliberately worse than chance, on the
        argument that there should be no setting at which sitting at ln 2 is
        free. That argument cost the run its bootstrap: a listener that must
        commit through a fixed volume from step zero is committing before the
        message carries anything.

    A plain readout opens where its initialisation puts it, which on rung 11's
        width is sd ~0.59 and BCE ~0.73 -- just above ln 2, which is the
        bilinear comparer's regime and the one that bootstraps. Bounded rather
        than pinned, because the number is now an emergent property of
        `nn.Linear`'s default init and the referent stack's post-norm output
        scale, and pinning it would make this a change-detector for PyTorch.
    """
    listener = _quiet_cross_comparer()
    with torch.no_grad():
        scores = listener(*_inputs(listener))

    loss = F.binary_cross_entropy_with_logits(scores, _labels()).item()

    assert 0.2 < scores.std().item() < 1.5
    assert math.log(2.0) < loss < 1.0


def test_the_decision_head_now_reaches_the_score_magnitude():
    """
    Where the volume lives now. `standardise` used to run on `decision`'s
        output, so scaling its weight scaled the pre-standardisation logits,
        their mean and their spread alike, and the quotient did not move -- the
        head could turn but neither shout nor fall silent, and the volume sat
        in `log_mix_scale` downstream of both paths.

    The branches are mixed unstandardised now, so `decision.weight` sets its
        branch's volume as well as its direction, exactly as
        `bilinear.weight` does on the other one. The score moves by the
        attention path's *share* of it rather than by the full factor, which is
        what `mix_share` reports.
    """
    listener = _quiet_cross_comparer()
    discriminator = listener.discriminator
    referents, messages = _inputs(listener)

    with torch.no_grad():
        before = listener(referents, messages)
        opening_spread = discriminator.decision_spread
        opening_share = discriminator.mix_share

        discriminator.decision.weight.mul_(10.0)
        discriminator.decision.bias.mul_(10.0)
        after = listener(referents, messages)

    assert not torch.allclose(after, before, rtol=1e-3, atol=1e-3)
    assert discriminator.decision_spread > opening_spread
    # The branch got ten times louder, so it now makes up more of the score
    #     without `mix_logit` having moved at all. That gap is the whole reason
    #     `mix_share` is reported next to `mix_alpha`.
    assert discriminator.mix_share > opening_share
    assert discriminator.mix_alpha == pytest.approx(
        discriminator.mix_weight.item()
    )


def test_the_listener_can_still_go_quiet():
    """
    The freedom the bootstrap needs, and the one thing that must survive every
        rearrangement of this readout. A pair forced to commit through a fixed
        volume from step zero is committing before the message carries
        anything, which is what took four rungs down -- see this module's
        docstring and docs/anecdotes.md.

    It used to be one scalar. It is now both branch weights, which is strictly
        more freedom and not less; what it costs is that a branch can also go
        quiet *alone*, which standardising had closed. `mix_share` against
        `mix_alpha` is what watches that.
    """
    listener = _quiet_cross_comparer()
    discriminator = listener.discriminator
    referents, messages = _inputs(listener)

    with torch.no_grad():
        before = listener(referents, messages)
        opening_spread = discriminator.decision_spread

        discriminator.decision.weight.mul_(1e-3)
        discriminator.decision.bias.mul_(1e-3)
        discriminator.bilinear.bilinear.weight.mul_(1e-3)
        quiet = listener(referents, messages)

    assert quiet.std().item() < 0.01 * before.std().item()
    assert discriminator.decision_spread < 0.01 * opening_spread


def test_a_constant_attention_readout_leaves_the_bilinear_path_deciding():
    """
    The attention path going flat is not the end state of a collapse: it is one
        path saying nothing while the other still decides. A constant is the
        same value for every candidate in a game, so it shifts the whole game's
        scores together and cannot change which candidate wins -- the bilinear
        path at `1 - mix_weight` keeps discriminating.

    Note what changed and what did not. `standardise` used to send that constant
        to exactly zero in the forward path; it now only does so in the
        telemetry, so `path_agreement` still reads 0.0 while the score itself
        carries the offset. Either way the offset is common to every candidate
        in a game, so it cannot decide anything -- what is asserted is the
        within-game spread, not that the winners match `live`. They need not:
        silencing the attention path removes a real contribution to the score,
        and would have under the old arrangement too.
    """
    listener = _quiet_cross_comparer()
    discriminator = listener.discriminator

    class _Constant(torch.nn.Module):
        def forward(self, x):
            return torch.full((*x.shape[:-1], 1), 3.7)

    referents, messages = _inputs(listener)
    with torch.no_grad():
        live = listener(referents, messages)
        discriminator.decision = _Constant()
        flattened = listener(referents, messages)

    # Within a game the constant is common to every candidate, so the spread
    #     over candidates -- which is what decides -- survives it.
    within_game = flattened - flattened.mean(1, keepdim=True)
    assert within_game.std().item() > 0.1 * live.std().item()
    assert discriminator.path_agreement == pytest.approx(0.0, abs=1e-6)


def test_a_listener_with_no_spread_at_all_is_reported_rather_than_hidden():
    """
    The end state of a collapse, and what the columns say when it arrives. It
        takes *both* branch weights going to zero now that there is no scalar
        downstream of them -- but when it happens the reporting is unchanged:
        zero spread, and a kurtosis of NaN rather than a 0.0 that would read as
        "Gaussian, nothing to see".
    """
    listener = _quiet_cross_comparer()
    discriminator = listener.discriminator

    with torch.no_grad():
        discriminator.decision.weight.zero_()
        discriminator.decision.bias.zero_()
        discriminator.bilinear.bilinear.weight.zero_()
        scores = listener(*_inputs(listener))

    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-12)
    assert discriminator.decision_spread == pytest.approx(0.0, abs=1e-6)
    assert math.isnan(discriminator.decision_kurtosis)


def test_the_kurtosis_column_separates_the_shapes_the_spread_column_cannot():
    """
    Size and shape are different questions and the readout reports both,
        because on the runs that mattered only one of them answered.

    The column arrived with the standardised readout, where the escape was
        specifically through the fourth moment -- under a pinned variance a
        handful of outliers absorb the budget cheaply while the bulk sits at
        sigmoid 0.5. That arbitrage went with the pin. What the column reads did
        not: bimodal scores are what a discriminating listener produces and
        floor at -2, heavy-tailed ones are what a listener with nothing to say
        produces, and no amount of reading `decision_spread` distinguishes them.

    Driven here with the two distributions directly rather than through
        training, and `decision_spread` is asserted *identical* across both --
        that is the point of the test. On the real runs the two conditions
        overlapped on it (1.4-2.1 against 2.7-5.1) while kurtosis separated them
        by sign.

    `mix_logit` is pushed to saturation so the mix is the attention path
        alone, which is what lets a shape injected at `decision` reach the
        columns unchanged. Standardising does not disturb the measurement: both
        columns are read off the final scores, kurtosis is invariant to a
        positive affine, and `decision_spread` is asserted equal across the two
        conditions rather than at a particular value.
    """
    listener = _quiet_cross_comparer()
    discriminator = listener.discriminator
    with torch.no_grad():
        discriminator.mix_logit.fill_(50.0)
    assert discriminator.mix_weight.item() == pytest.approx(1.0)

    referents, messages = _inputs(listener)

    class _Shape(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = values

        def forward(self, x):
            return self.values.expand(*x.shape[:-1]).reshape(*x.shape[:-1], 1)

    # Built *per game* rather than over the flattened batch, because
    #     `standardise` works per game: a shape that lived only in the first and
    #     last games would leave the rest flat and the spread column would then
    #     be reading how many games had any variation at all.
    bimodal = torch.where(torch.arange(N_OBJ) % 2 == 0, 1.0, -1.0)

    # One candidate a game at each of +-7 and the rest at zero, matched to
    #     `bimodal`'s standard deviation so only the shape differs.
    heavy = torch.zeros(N_OBJ)
    heavy[0] = 7.0
    heavy[-1] = -7.0
    heavy = heavy * (bimodal.std() / heavy.std())

    readings = {}
    for name, values in (("bimodal", bimodal), ("heavy", heavy)):
        discriminator.decision = _Shape(values)
        with torch.no_grad():
            listener(referents, messages)
        readings[name] = (
            discriminator.decision_spread, discriminator.decision_kurtosis
        )

    assert readings["bimodal"][1] == pytest.approx(-2.0, abs=0.05)
    assert readings["heavy"][1] > 5.0
    assert readings["bimodal"][0] == pytest.approx(readings["heavy"][0], rel=1e-3)


def test_the_readout_still_carries_gradient_to_the_message():
    """
    The failure mode this whole change is aimed at is a listener that stops
        passing anything back. Normalising the readout must not be a way of
        doing that quietly.
    """
    listener = _quiet_cross_comparer()
    referents, messages = _inputs(listener)
    messages = messages.clone().requires_grad_(True)

    F.binary_cross_entropy_with_logits(
        listener(referents, messages), _labels()
    ).backward()

    assert messages.grad is not None
    assert messages.grad.norm().item() > 0.0


# --------------------------------------------------------------------------
# The optimiser wiring.
# --------------------------------------------------------------------------

def _pair_and_optimiser(config_file):
    config = parse_config.get_config(os.path.join(CONFIG_DIR, config_file))
    config["cuda"] = False

    class _Dataset:
        n_feats = (3, 224, 224)
        name = "cub"

    class _Loader:
        dataset = _Dataset()

    built = models.builder.build_models({"train": _Loader()}, config)
    return config, built["pair"], built["optimiser"]


def test_the_listener_asks_for_no_elevated_volume_rate():
    """
    The keys are gone, not merely unused. `score_scale_lr` and `mix_scale_lr`
        moved two volumes on an elevated rate, and the listener spent that
        mobility turning itself down -- so the volumes moved onto the weight
        matrices that produce the scores, where the base rate applies and
        `is_mup_exempt` does not claim them either.

    A leftover key would be worse than a leftover parameter here:
        `split_out_parameter` raises when nothing matches its suffix,
        deliberately, so a stale key takes every rung down at construction.
    """
    config, pair, optimiser = _pair_and_optimiser("02_birds_baseline.toml")

    assert "score_scale_lr" not in config["optimiser"]
    assert "mix_scale_lr" not in config["optimiser"]

    base = config["optimiser"]["lr"]
    weight = pair.receiver.discriminator.bilinear.weight
    holding = [
        group for group in optimiser.param_groups
        if any(p is weight for p in group["params"])
    ]

    assert len(holding) == 1
    assert holding[0]["lr"] == base


def test_an_attention_rung_asks_for_the_mix_weight_rate_and_nothing_else():
    """
    Which parameters are left in an elevated group, on the rung that has the
        most of them. The groups cannot be told apart by their rate --
        DEFAULT.toml opens all four remaining keys at 2e-3 -- so the assertion
        is on membership.

    This rung's `SenderTransformerLM` earns the speaker's two and its contrast
        stage a third. The listener contributes the mixing *weight* and nothing
        else: `mix_logit` is a composition knob, not a volume, so the argument
        that removed the other two does not reach it.
    """
    config, pair, optimiser = _pair_and_optimiser(
        "16_birds_receiver_cross_attention_lm.toml"
    )
    wanted = config["optimiser"]["mix_logit_lr"]
    assert wanted != config["optimiser"]["lr"]

    assert not any(
        name.endswith("log_score_scale") or name.endswith("log_mix_scale")
        for name, _ in pair.receiver.named_parameters()
    )

    named = {id(p): name for name, p in pair.named_parameters()}
    elevated = {
        named[id(p)]
        for group in optimiser.param_groups if group["lr"] == wanted
        for p in group["params"]
    }

    assert elevated == {
        "sender.language_model.log_logit_scale",
        "sender.language_model.polarity_embedding",
        "sender.contrast.contrast_gate",
        "receiver.discriminator.mix_logit",
    }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

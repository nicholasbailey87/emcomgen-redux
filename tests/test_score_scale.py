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

The fifth took both scalars away again and gave the volume to the weight
matrices. The reason was that the listener spent the elevated mobility squashing
its own logits: `score_scale` fell 0.9021 -> 0.3731 on rung 09 and 0.9377 ->
0.4072 on rung 11 across thirty epochs, monotone, never returning -- correct
behaviour on a message carrying nothing, since BCE's minimiser is p=0.5
everywhere, but the scalar factors to the front of the score, so shrinking it
multiplies down every gradient going back through the message and into the
speaker. A fast scalar is a cheap single-knob route to going quiet.

That stopped the volume moving. `bilinear_weight_norm` travelled 13.055 ->
12.889 on rung 09 and 13.049 -> 12.968 on rung 10 over thirty epochs -- 1.3% and
0.6%, against the 59% the scalar managed -- because a 320x320 matrix under Adam
spends its step turning and only a fraction of it radially, and because
`score_scale_lr = 2e-3` went with the scalar so the matrix also dropped to the
1e-4 base. Rung 10, the one rung on this ladder that has ever ignited, then sat
at `train_loss` 0.7298 -> 0.7006 for a whole run: *above* ln 2 throughout, with
`realised_survival` collapsing 0.545 -> 0.190. Freedom that cannot be exercised
at the available learning rate is the third answer's fixed gain wearing a
different hat.

The sixth kept the scalar and tried to remove the coupling that made it look
dangerous: `ScoreVolume` standardised the score per game and applied one
`log_score_scale` through `model_util.scale_without_attenuating`, so the forward
was `s * standardise(u)` and the backward into the message did not carry `s`.

**The seventh, which this file now tests, is that the coupling was never
reaching the optimiser and the standardise was never needed.**

The coupling first. AdamW updates by `m / sqrt(v)`, and a uniform factor on a
parameter's gradient scales the numerator and the denominator alike, so it
cancels. `train.py`'s `clip_gradients` is per-submodule and renormalises each
module to `clip_grad_norm` whenever it binds, which at the ablation's recorded
speaker norms of ~10 against a ceiling of 1.0 it does. Five rounds of design
went into a factor that two lines of the training loop were already removing.
What the helper did add was an inconsistent gradient -- `dL/ds = x` is the true
partial while `dL/du = J` is not, the truth being `s * J` -- so the machinery
behind the scale was shaped for a volume of 1 whatever the forward used.

The standardise second. Both of `BilinearDiscriminator`'s operands are already
layer-normed, so the score is already backbone-independent; normalising it again
downstream bought nothing and cost the exact `/sqrt(referent_dim)` calibration,
trading an analytic opening for an empirical one. It also divided each game by
its own margin, which is the wrong way round -- but measurably too small to be
the whole story, and absent at initialisation where the spread is 0.567 whether
the message carries signal or noise. That is recorded in docs/anecdotes.md
rather than claimed here.

So the readout was `s * u` on a score whose inputs are normalised, and the
opening is `1/sqrt(3)` at every width and under every backbone -- the same
number the first round calibrated for, arrived at by keeping the operands
normalised rather than by normalising the result.

Round eight finished that. Removing the centring made the decision threshold a
fixed origin -- `train.py` reads `lis_scores > 0` -- and only four of the
sixteen rungs could place their scores against one. `AttentionDiscriminator`
had a `mix_bias`; `BilinearDiscriminator`, and so rungs 1-12, had no bias
anywhere, `bilinear` being built `bias=False`. So `ScoreVolume` gained a
`score_bias` beside its volume, applied after it, and `mix_bias` retired into
it: same position, same arithmetic, both arms, and now with a config key at
`score_bias_lr` and a metrics column. The readout is `s * u + b`.

It opens at zero and is expected to stay near it, because the games are balanced
and the loss-optimal global offset is therefore about zero. It is insurance
against a systematic offset, and it cannot reach a per-game one.

Consequences the tests below follow. `bilinear.weight` carries volume as well as
direction again, so `bilinear_weight_norm` is not the drift column it briefly
was. `decision` still has no bias, now because `score_bias` is the module's one
constant across candidates rather than because a centring would annihilate it --
a bias there is worth `score_scale * mix_weight * b` on the score, which is what
`score_bias` says directly. And neither branch is standardised, which is what
keeps a branch able to go quiet alone and `mix_share` worth reporting beside
`mix_alpha`.
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

from _bootstrap import build_listener, config_section, rung

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


def _bilinear_weight(discriminator):
    """
    The bilinear comparison's matrix, wherever the arm keeps it. On the
        attention arm the whole `BilinearDiscriminator` is composed, so it is a
        level deeper.
    """
    if isinstance(discriminator, R.AttentionDiscriminator):
        return discriminator.bilinear.bilinear.weight
    return discriminator.bilinear.weight


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

def test_the_score_opens_at_one_over_root_three():
    """
    The calibration, and it is exact rather than cosmetic because both operands
        of the bilinear form arrive layer-normed. Each is at per-element unit
        variance, so `|r| = |m| = sqrt(d)`; `nn.Linear`'s default init is
        uniform on `+/- 1/sqrt(fan_in)` with standard deviation `1/sqrt(3d)`,
        which puts the raw score's standard deviation at `sigma_w * d =
        sqrt(d/3)`; and `/sqrt(referent_dim)` leaves `1/sqrt(3)` = 0.577.

    `log_score_scale` opens at 0, so that is where the readout opens too. BCE
        on a random map at that spread is 0.725 against `ln 2` = 0.693, where
        unit spread would give 0.804 -- the gentler opening is why the scalar is
        left at 1.0 rather than calibrated to `sqrt(3)`.

    Pooled rather than per game: nothing centres the score any more, so a
        per-game spread would miss exactly the between-game variation this
        arrangement leaves free.
    """
    listener = _comparer().eval()
    with torch.no_grad():
        scores = listener(*_inputs(listener))

    opening = scores.std().item()
    assert opening == pytest.approx(3.0 ** -0.5, rel=0.12)

    # And the scalar is exactly proportional on top of it: the readout is a
    #     plain product, so turning the listener down by 4x turns the score
    #     down by 4x and nothing renormalises it back.
    with torch.no_grad():
        listener.discriminator.log_score_scale.fill_(math.log(0.25))
        quiet = listener(*_inputs(listener))

    assert quiet.std().item() == pytest.approx(0.25 * opening, rel=1e-4)


@BOTH
@pytest.mark.parametrize("referent_dim", [320, 512])
def test_the_untrained_score_opens_at_a_width_independent_magnitude(
    build, referent_dim
):
    """
    The property that survives every design this file records: the opening
        confidence is the same whichever backbone the rung mounts, rather than
        growing with its width.

    Bought differently on each arm, and that is fine: `/sqrt(referent_dim)`
        over two layer-normed operands on the bilinear one, the referent
        stack's last post-norm putting every candidate at unit RMS on the
        other. What matters is that neither inherits the backbone's magnitude.
        The band below is left wide rather than tightened to the bilinear arm's
        exact `1/sqrt(3)`, so that it keeps testing the width and not the
        readout; `test_the_score_opens_at_one_over_root_three` is the tight one.
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
# What carries the volume now: exactly one `log_score_scale` per discriminator,
#     from the `ScoreVolume` mixin, downstream of a per-game `standardise`. The
#     weight matrices carry direction. See the sixth round in the preamble.
# --------------------------------------------------------------------------

def test_each_discriminator_owns_exactly_one_volume_and_one_offset():
    """
    One of each per arm, under one name, so one config key and one suffix reach
        both. Neither `log_mix_scale` nor `mix_bias` comes back:
        `AttentionDiscriminator` uses the same `ScoreVolume` the bilinear arm
        does, for the offset as well as the volume.

    The offset matters more on the bilinear arm than on the arm it came from.
        `mix_bias` existed only on `AttentionDiscriminator`, so rungs 1-12 had
        no bias anywhere -- `bilinear` is built `bias=False` -- and could not
        place their scores against `train.py`'s fixed `lis_scores > 0` at all.
    """
    for build in (_comparer, _cross_comparer):
        named = dict(build().named_parameters())
        volumes = [name for name in named if name.endswith("log_score_scale")]
        offsets = [name for name in named if name.endswith("score_bias")]

        assert len(volumes) == 1, sorted(named)
        assert len(offsets) == 1, sorted(named)
        assert not [name for name in named if "log_mix_scale" in name]
        assert not [name for name in named if "mix_bias" in name]


def test_the_composed_bilinear_path_has_neither_of_its_own():
    """
    `AttentionDiscriminator` builds its bilinear path with `score_scale=False`,
        and that flag gates both of `ScoreVolume`'s scalars.

    The reason is degeneracy, for each of them. The composed path is one of two
        branches multiplied by `1 - mix_weight` and read out through the outer
        module downstream, so a scale on it says what `mix_logit` already says,
        and a constant across candidates on it says what the outer `score_bias`
        already says. Either would still match its `SPLIT_LEARNING_RATES` suffix
        and still report a value.

    Absent rather than frozen, so `split_out_parameter`'s suffix match sees the
        truth.
    """
    attention = _cross_comparer().discriminator

    assert attention.learns_score_scale
    assert not attention.bilinear.learns_score_scale
    assert not hasattr(attention.bilinear, "log_score_scale")
    assert not hasattr(attention.bilinear, "score_bias")


def test_both_the_scale_and_the_weight_reach_the_score_magnitude():
    """
    `bilinear_weight_norm` is a volume column again. Nothing downstream divides
        a rescaling of `bilinear.weight` back out -- the two operands are
        normalised going *in*, not the score coming out -- so scaling `W` by 37
        scales the score by 37, exactly as the scalar does.

    Sharing the volume between a matrix and a scalar is not the ambiguity two
        scalars would be. A 320x320 matrix under Adam spends its step turning
        and only a fraction of it radially, which is the measurement that
        killed the round where `W` held the volume alone: 1.3% of its norm in
        thirty epochs against the scalar's 59%. The scalar is the fast path and
        the matrix is not competing for the job.
    """
    listener = _comparer().eval()
    referents, messages = _inputs(listener)

    with torch.no_grad():
        before = listener(referents, messages)
        listener.discriminator.bilinear.weight.mul_(37.0)
        after = listener(referents, messages)

    assert torch.allclose(after, 37.0 * before, atol=1e-3)

    with torch.no_grad():
        listener.discriminator.bilinear.weight.div_(37.0)
        listener.discriminator.log_score_scale.fill_(math.log(37.0))
        loud = listener(referents, messages)

    assert torch.allclose(loud, 37.0 * before, atol=1e-3)


def test_scaling_the_volume_cannot_change_the_decision():
    """
    `train.py` reads the decision as `scores > 0` and the reference-game branch
        as an argmax, and neither may move with the listener's confidence. A
        positive rescale multiplies an operand shared across the objects of a
        game, so it cannot change which object wins -- only how loudly the
        listener says so.

    Which is why a volume collapse is invisible in `train_acc`, and why the
        volume needs a column of its own. `train_score_scale` is that column.
    """
    listener = _comparer().eval()
    referents, messages = _inputs(listener)
    with torch.no_grad():
        quiet = listener(referents, messages)
        listener.discriminator.log_score_scale.fill_(math.log(37.0))
        loud = listener(referents, messages)

    assert torch.equal(quiet > 0, loud > 0)
    assert torch.equal(quiet.argmax(1), loud.argmax(1))


def test_the_offset_is_downstream_of_the_volume():
    """
    Why `readout` is `score_scale * scores + score_bias` in that order, and the
        property that made `mix_bias`'s position load-bearing.

    An offset applied *before* the volume would be multiplied by it, so the
        threshold would slide every time the listener changed how loudly it
        spoke -- and `score_scale` moves fast, at `score_scale_lr` = 2e-3.
        Downstream, a bias of `b` moves the score by exactly `b` whatever the
        volume is doing, so the two parameters say independent things.
    """
    listener = _comparer().eval()
    referents, messages = _inputs(listener)

    with torch.no_grad():
        listener.discriminator.log_score_scale.fill_(math.log(0.25))
        before = listener(referents, messages)
        listener.discriminator.score_bias.fill_(1.0)
        after = listener(referents, messages)

    # Exactly 1.0, not 0.25. Upstream of the volume it would have been 0.25.
    assert torch.allclose(after - before, torch.ones_like(before), atol=1e-5)


def test_moving_the_offset_does_change_the_decision():
    """
    The whole point of it, and the one thing `score_scale` cannot do -- pair
        this with `test_scaling_the_volume_cannot_change_the_decision` above.

    `train.py` decides on `lis_scores > 0`, a fixed origin since the readout
        stopped centring each game on its own mean. A positive rescale moves
        every candidate towards or away from zero without crossing it; an
        offset is what actually moves the threshold. Before `score_bias`,
        nothing on the bilinear arm could.
    """
    listener = _comparer().eval()
    referents, messages = _inputs(listener)

    with torch.no_grad():
        before = listener(referents, messages)
        # Comfortably past the opening spread of `1/sqrt(3)` = 0.577, so every
        #     candidate in every game ends up on the positive side.
        listener.discriminator.score_bias.fill_(50.0)
        after = listener(referents, messages)

    assert not torch.equal(before > 0, after > 0)
    assert bool((after > 0).all())
    # And the *ordering* is untouched, which is what makes it a threshold
    #     rather than a re-ranking: it adds the same constant to every candidate.
    assert torch.equal(before.argmax(1), after.argmax(1))


def test_scaling_the_volume_does_change_the_loss():
    """
    Which is what makes it a volume: BCE is not scale-invariant, so this is the
        listener's control over its own confidence, and the exposure that makes
        quietening a first-order descent direction. What has changed is not
        that -- the listener still gets to go quiet, and BCE still rewards it
        for doing so on a message carrying nothing. What going quiet costs
        upstream is a uniform factor the optimiser divides out. See
        `test_the_gradient_reaching_the_message_tracks_the_volume`.
    """
    listener = _comparer().eval()
    referents, messages = _inputs(listener)
    labels = _labels()

    with torch.no_grad():
        quiet = F.binary_cross_entropy_with_logits(
            listener(referents, messages), labels
        ).item()
        listener.discriminator.log_score_scale.fill_(math.log(37.0))
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
    """
    Both halves: the scalar that is now the volume, and the matrix that is now
        the direction. A reset must not leave a trained confidence, or a trained
        comparison, behind a fresh listener.
    """
    for build in (_comparer, _cross_comparer):
        discriminator = build().discriminator
        opening = _bilinear_weight(discriminator).norm().item()

        with torch.no_grad():
            discriminator.log_score_scale.fill_(math.log(37.0))
            _bilinear_weight(discriminator).mul_(37.0)
        discriminator.reset_parameters()

        assert discriminator.score_scale.item() == pytest.approx(1.0)

        # Not the same draw, so the norm rather than the tensor: what has to
        #     come back is the scale of the opening, which `fan_in` fixes.
        assert _bilinear_weight(discriminator).norm().item() == pytest.approx(
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
# What follows it is the mix, and `ScoreVolume.readout` standardises the mix.
#     That is not the fixed gain coming back: the volume is a learned scalar
#     downstream of the standardise rather than a pinned constant, and the
#     bilinear path carries the decision through the opening so nothing has to
#     be confident early. Standardising the *mix* rather than each branch is
#     what leaves the escape of turning one path down open -- `mix_share`
#     against `mix_alpha` is what watches it.
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

    The bias stays off, and since `485b38e` for a different reason than it was
        turned off for. It used to be *dead*: the readout centred each game, and
        a constant added to every candidate is exactly what a centring removes,
        so it would have taken identically zero gradient. The centring is gone,
        so it is now merely redundant -- it adds the same constant across
        candidates that `score_bias` adds, one degree of freedom expressed by two
        parameters free to drift against each other. One offset per module.
    """
    discriminator = _cross_comparer(referent_dim=320).discriminator

    assert not hasattr(discriminator, "score_norm")
    assert not hasattr(discriminator, "decision_gain")
    assert isinstance(discriminator.decision, torch.nn.Linear)
    assert discriminator.decision.out_features == 1
    assert discriminator.decision.bias is None

    # One scalar volume, downstream of the mix, and none on either branch.
    assert discriminator.learns_score_scale
    assert not hasattr(discriminator.bilinear, "log_score_scale")
    assert not hasattr(discriminator, "log_mix_scale")
    assert not hasattr(discriminator, "mix_scale")


def test_a_bias_on_the_decision_head_is_redundant_with_the_offset():
    """
    Why `decision` has none. The measurement behind the docstring above, and it
        changed with `485b38e`.

    It used to be that a bias there could not move the score at all -- the
        readout centred each game and a constant across candidates is exactly
        what a centring removes. Now it moves the score by precisely the amount
        `score_bias` would have to be moved to match it: the attention branch is
        multiplied by `mix_weight` and read out through `score_scale`, so a bias
        of `b` on the head is worth `score_scale * mix_weight * b` on the score,
        the same constant for every candidate in the game. Two parameters, one
        degree of freedom.
    """
    listener = _quiet_cross_comparer()
    referents, messages = _inputs(listener)
    discriminator = listener.discriminator
    decision = discriminator.decision

    with torch.no_grad():
        before = listener(referents, messages)

    revived = torch.nn.Linear(
        decision.in_features, decision.out_features, bias=True
    )
    with torch.no_grad():
        revived.weight.copy_(decision.weight)
        revived.bias.fill_(11.0)
    discriminator.decision = revived

    with torch.no_grad():
        after = listener(referents, messages)
        expected = (
            discriminator.score_scale * discriminator.mix_weight * 11.0
        ).item()

    shift = after - before
    # A constant across the game, and the one `score_bias` also expresses.
    assert shift.std(1, unbiased=False).max().item() < 1e-4
    assert shift.mean().item() == pytest.approx(expected, rel=1e-3)


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


def test_the_decision_head_reaches_both_its_branch_share_and_the_volume():
    """
    What the branch weights do, which since `485b38e` is both things.

    They set their branch's magnitude, and the branches are mixed
        unstandardised, so making one ten times louder changes what the score is
        *made of* without `mix_logit` moving at all. That gap is the whole
        reason `mix_share` is reported next to `mix_alpha`.

    They now also set the volume, which they did not while the readout
        standardised the mix -- this test asserted `decision_spread` held to
        within 2% of its opening until that centring was removed. It moves now,
        and that is the intended arrangement: nothing downstream divides a
        rescale of a branch weight back out, so `decision_weight_norm` and
        `bilinear_weight_norm` mean magnitude as well as direction and
        `score_scale` is one voice among several rather than the only one.
    """
    listener = _quiet_cross_comparer()
    discriminator = listener.discriminator
    referents, messages = _inputs(listener)

    with torch.no_grad():
        before = listener(referents, messages)
        opening_spread = discriminator.decision_spread
        opening_share = discriminator.mix_share

        discriminator.decision.weight.mul_(10.0)
        after = listener(referents, messages)

    assert not torch.allclose(after, before, rtol=1e-3, atol=1e-3)
    assert discriminator.mix_share > opening_share
    # Well short of 10x, and it should be: `mix_logit_init` opens `mix_weight`
    #     at ~0.116, so a 10x on the attention branch alone reaches the mix
    #     diluted. Measured at 1.63x. What matters is that it moves at all.
    assert discriminator.decision_spread > 1.5 * opening_spread
    # `mix_alpha` is the parameter's own reading, so it is the one thing here
    #     that must *not* move: it is what `mix_share` is compared against.
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

    It has been one scalar, then two matrices, and is one scalar again. What is
        new is that going quiet is now free of consequence upstream: see
        `test_the_gradient_reaching_the_message_does_not_see_the_volume`. So
        this is the same freedom, with the reason it was taken away removed.
    """
    listener = _quiet_cross_comparer()
    discriminator = listener.discriminator
    referents, messages = _inputs(listener)

    with torch.no_grad():
        before = listener(referents, messages)
        opening_spread = discriminator.decision_spread

        discriminator.log_score_scale.fill_(math.log(1e-3))
        quiet = listener(referents, messages)

    assert quiet.std().item() < 0.01 * before.std().item()
    assert discriminator.decision_spread < 0.01 * opening_spread


def test_silencing_both_branches_silences_the_score():
    """
    The other half of the above, and why `bilinear_weight_norm` and
        `decision_weight_norm` *are* volume columns on this arm.

    This asserted the opposite between `7b10d47` and `485b38e`: with the readout
        standardising the mix, turning both branches down to a thousandth left
        the score coming back out at `score_scale` regardless, because
        `standardise` divides by whatever spread survives and 1e-3 is nowhere
        near its 1e-6 clamp. That is what made the branch norms unreadable as
        volume, and it is gone.
    """
    listener = _quiet_cross_comparer()
    discriminator = listener.discriminator
    referents, messages = _inputs(listener)

    with torch.no_grad():
        before = listener(referents, messages)
        discriminator.decision.weight.mul_(1e-3)
        discriminator.bilinear.bilinear.weight.mul_(1e-3)
        still_loud = listener(referents, messages)

    assert still_loud.std(1, unbiased=False).mean().item() == pytest.approx(
        1e-3 * before.std(1, unbiased=False).mean().item(), rel=0.05
    )


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
    The end state of a collapse, and what the columns say when it arrives.

    The route there is the scalar again: with the readout standardising, zeroing
        both branch weights leaves a mix that is identically zero, and
        `standardise` divides it by its clamped 1e-6 floor rather than by
        nothing -- so the mix stays at zero and `score_scale` is what decides
        how loud zero is. Reporting is unchanged: zero spread, and a kurtosis of
        NaN rather than a 0.0 that would read as "Gaussian, nothing to see".
    """
    listener = _quiet_cross_comparer()
    discriminator = listener.discriminator

    with torch.no_grad():
        discriminator.log_score_scale.fill_(math.log(1e-12))
        scores = listener(*_inputs(listener))

    assert discriminator.decision_spread == pytest.approx(0.0, abs=1e-6)
    assert math.isnan(discriminator.decision_kurtosis)


def test_a_mix_with_no_spread_survives_the_readout_without_dividing_by_zero():
    """
    The clamp in `standardise`, reached from the forward path rather than
        constructed by hand. Both branch weights at zero make every candidate in
        a game score identically, which is the 0/0 the clamp exists for.
    """
    listener = _quiet_cross_comparer()
    discriminator = listener.discriminator

    with torch.no_grad():
        discriminator.decision.weight.zero_()
        discriminator.bilinear.bilinear.weight.zero_()
        scores = listener(*_inputs(listener))

    assert torch.isfinite(scores).all()
    assert scores.std(dim=1, unbiased=False).max().item() == pytest.approx(
        0.0, abs=1e-6
    )


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
# The property the sixth round exists for: the volume is absent from the
#     backward pass into everything upstream of it. See
#     `model_util.scale_without_attenuating`.
# --------------------------------------------------------------------------

def _message_gradient(listener, scale):
    """The norm of dL/dmessages at a given `score_scale`."""
    referents, messages = _inputs(listener)
    messages = messages.clone().requires_grad_(True)

    with torch.no_grad():
        listener.discriminator.log_score_scale.fill_(math.log(scale))

    F.binary_cross_entropy_with_logits(
        listener(referents, messages), _labels()
    ).backward()

    return messages.grad.norm().item()


@BOTH
def test_the_gradient_reaching_the_message_tracks_the_volume(build):
    """
    The behaviour five rounds of design tried to remove, asserted as the live
        one because removing it was a mistake.

    A scalar at the front of the score multiplies every gradient behind it, so
        a listener turning itself down scales down what reaches the speaker.
        That was read as a self-sealing loop -- quiet listener, starved
        speaker, nothing worth listening to, quieter listener -- and answered
        twice, by deleting the scalar (`a9a6a9c`) and by hiding it from the
        backward pass (`7b10d47`).

    Neither was necessary. AdamW updates by `m / sqrt(v)`, so a uniform factor
        on a parameter's gradient cancels before it becomes a step, and
        `train.py`'s per-submodule `clip_gradients` renormalises whatever
        survives that. See the preamble.

    Asserted low rather than across four orders of magnitude: at large `s` the
        loss's own saturation takes over -- `sigma(s*z) - y` flattens -- so the
        product stops being monotone in `s`. That saturation belongs to BCE and
        is deliberately kept.
    """
    listener = build().eval()
    at_one = _message_gradient(listener, 1.0)

    assert _message_gradient(listener, 1e-2) < 0.1 * at_one
    assert _message_gradient(listener, 1e-1) < 0.5 * at_one


def test_the_readout_does_not_reweight_games_by_their_own_margin():
    """
    What removing `standardise` from the readout actually bought, measured
        rather than argued, because the argument overshot.

    `standardise` divides each game by the spread of its own candidate scores.
        That spread is the margin -- how well the listener is separating that
        game -- so dividing by it hands every game the same magnitude whatever
        its message carried, damping the informative games relative to the
        uninformative ones. Which is backwards, since at bootstrap the games
        that accidentally do better are the only signal there is.

    The size of it is the part worth pinning. Measured on this module:

      * With `bilinear.weight` at its random init -- the bootstrap regime --
        the score's spread is 0.567 whether the message carries the separating
        direction or pure noise, because the listener cannot read the message
        yet. There is no margin to divide by, so `standardise` is a *uniform*
        factor there and AdamW cancels it.
      * With a listener that can read the message the spreads come apart, 4.20
        against 0.98, and `standardise` then damps the informative games by
        about 1.4x relative to the noise ones.

    So the reweighting is real, points the way the argument said, and is far
        too small on its own to explain the 382-445x slowdown in the sender's
        pre-channel parameters that the 2026-08-26 rung 9 and 10 runs showed.
        `standardise` is present in exactly the frozen runs and absent from
        every run that was merely dead, and that correlation is **not
        explained** by this. See docs/anecdotes.md.

    This test pins the first bullet, which is the one that makes the readout's
        behaviour at initialisation independent of what it cannot yet read.
    """
    listener = _comparer().eval()
    referents, messages = _inputs(listener)
    discriminator = listener.discriminator

    with torch.no_grad():
        scores = listener(referents, messages)
        spreads = scores.std(dim=1, unbiased=False)

    # A readout that reweighted by the margin would have to see the margins
    #     differ; at the opening they do not, which is why the effect the
    #     removal was argued from is absent exactly where it was wanted.
    assert spreads.std().item() < 0.25 * spreads.mean().item()

    # And a game the listener does separate arrives louder, because nothing
    #     renormalises it -- the property `standardise` removed. Built with a
    #     listener that can read: the identity weight makes the message's own
    #     direction the scoring one, so a message pointed at the gap between
    #     the two halves separates the candidates and a random one does not.
    referent_dim = discriminator.referent_embedding_size
    reader = R.BilinearDiscriminator(referent_dim, referent_dim).eval()
    with torch.no_grad():
        reader.bilinear.weight.copy_(torch.eye(referent_dim))

    torch.manual_seed(11)
    candidates = torch.randn(32, N_OBJ, referent_dim)
    half = N_OBJ // 2
    separating = (
        candidates[:, :half].mean(1) - candidates[:, half:].mean(1)
    ).unsqueeze(1)

    with torch.no_grad():
        informative = reader(candidates, separating).std(dim=1).mean().item()
        noise = reader(
            candidates, torch.randn(32, 1, referent_dim)
        ).std(dim=1).mean().item()

    assert informative > 2.0 * noise


def test_the_volume_still_learns_and_still_wants_to_be_quiet_on_noise():
    """
    The other half, and unchanged by any of this: `dz/ds` is the score itself,
        so `dL/ds` reads whether being louder would have helped.

    Sign, on a message carrying nothing: BCE's minimiser is p = 0.5 everywhere,
        so the gradient asks for a smaller scale. `log_score_scale` is the log,
        and a *positive* gradient there means descent shrinks it.
    """
    listener = _comparer()
    referents, messages = _inputs(listener)

    F.binary_cross_entropy_with_logits(
        listener(referents, messages), _labels()
    ).backward()

    gradient = listener.discriminator.log_score_scale.grad
    assert gradient is not None
    assert gradient.item() > 0.0


# --------------------------------------------------------------------------
# The optimiser wiring.
# --------------------------------------------------------------------------

def _pair_and_optimiser(config_file):
    config = parse_config.get_config(rung(config_file))
    config["cuda"] = False

    class _Dataset:
        n_feats = (3, 224, 224)
        name = "cub"

    class _Loader:
        dataset = _Dataset()

    built = models.builder.build_models({"train": _Loader()}, config)
    return config, built["pair"], built["optimiser"]


def test_the_readout_scalars_are_elevated_and_the_weight_that_turns_is_not():
    """
    The split the rates encode. `score_scale_lr` and `score_bias_lr` each move a
        lone scalar, which Adam takes about `lr` per step whatever the gradient,
        so its whole travel over a run is bounded by `lr * steps` and it needs
        the elevated rate to be able to calibrate inside one. `bilinear.weight`
        learns a direction and stays at the base rate.

    `score_bias` is the one this arm never had. Its predecessor `mix_bias`
        existed only on `AttentionDiscriminator` and had no config key at all,
        so it sat at the base 1e-4: at birds' 194 steps an epoch that bounded
        its entire thirty-epoch travel at 0.58, against a score whose opening
        spread is 0.577.

    `mix_scale_lr` has no successor: one `ScoreVolume` per discriminator means
        one key. A leftover key would be worse than a leftover parameter here --
        `split_out_parameter` raises when nothing matches its suffix,
        deliberately, so a stale key takes every rung down at construction.
    """
    config, pair, optimiser = _pair_and_optimiser("02_birds_baseline.toml")

    assert "mix_scale_lr" not in config["optimiser"]
    elevated = config["optimiser"]["score_scale_lr"]
    base = config["optimiser"]["lr"]
    assert elevated != base
    assert config["optimiser"]["score_bias_lr"] == elevated

    discriminator = pair.receiver.discriminator

    def _group_lr(parameter):
        holding = [
            group for group in optimiser.param_groups
            if any(p is parameter for p in group["params"])
        ]
        assert len(holding) == 1
        return holding[0]["lr"]

    assert _group_lr(discriminator.log_score_scale) == elevated
    assert _group_lr(discriminator.score_bias) == elevated
    assert _group_lr(discriminator.bilinear.weight) == base

    # Separate groups, not one shared one. They open at the same rate, so the
    #     only way to see the difference is by identity -- and a single group
    #     would make one key silently move both.
    def _group(parameter):
        return next(
            group for group in optimiser.param_groups
            if any(p is parameter for p in group["params"])
        )

    assert _group(discriminator.log_score_scale) is not _group(
        discriminator.score_bias
    )


def test_an_attention_rung_asks_for_the_mix_weight_rate_and_nothing_else():
    """
    Which parameters are left in an elevated group, on the rung that has the
        most of them. Four of the five remaining keys cannot be told apart by
        their rate -- DEFAULT.toml opens the scaling scalars and `score_bias`
        together at 6e-3 -- so the assertion is on membership.

    `polarity_embedding` is the fifth and is deliberately *not* with them. It is
        a 2-d tag rather than a scalar, and it has no traverse to cover since it
        opens as an antipodal draw at `referent_layer_norm`'s own scale, so the
        argument that put the others at 6e-3 does not reach it. Since
        `2026-08-29` it takes the speaker's module rate, so it is asserted on
        group membership below rather than by looking a group up by its rate.

    This rung's `SenderTransformerLM` earns the speaker's two and its contrast
        stage a third. The listener contributes three: its volume, its offset,
        and the mixing *weight*. They are separate keys because they are
        separate things -- `mix_logit` says what the score is made of,
        `log_score_scale` how loudly it is stated, `score_bias` where it sits
        against `train.py`'s fixed `lis_scores > 0` -- and only one of them was
        ever accused of starving the speaker.

    Exactly one `log_score_scale` and one `score_bias` on the whole listener,
        and no `log_mix_scale` or `mix_bias`: `AttentionDiscriminator` composes
        a bilinear path built with `score_scale=False`, so the composed module
        contributes neither a second volume nor a second offset, and the offset
        it used to own itself now comes from `ScoreVolume` like the volume does.
    """
    config, pair, optimiser = _pair_and_optimiser(
        "16_birds_receiver_cross_attention_lm.toml"
    )
    wanted = config["optimiser"]["mix_logit_lr"]
    assert wanted == config["optimiser"]["score_scale_lr"]
    assert wanted != config["optimiser"]["lr"]

    volumes = [
        name for name, _ in pair.receiver.named_parameters()
        if name.endswith("log_score_scale")
    ]
    assert volumes == ["discriminator.log_score_scale"]

    offsets = [
        name for name, _ in pair.receiver.named_parameters()
        if name.endswith("score_bias")
    ]
    assert offsets == ["discriminator.score_bias"]

    assert not any(
        "log_mix_scale" in name or "mix_bias" in name
        for name, _ in pair.receiver.named_parameters()
    )

    named = {id(p): name for name, p in pair.named_parameters()}
    elevated = {
        named[id(p)]
        for group in optimiser.param_groups if group["lr"] == wanted
        for p in group["params"]
    }

    assert elevated == {
        "sender.contrast.contrast_gate",
        "receiver.discriminator.mix_logit",
        "receiver.discriminator.log_score_scale",
        "receiver.discriminator.score_bias",
    }

    # The tag keeps a group of its own, which is the whole point of it having a
    #     separate key: turning the listener's volume up must not retune a
    #     parameter that lives on one speaker only. That
    #     is a claim about the partition and not about the number, so it is
    #     asserted by finding the group that holds the tag rather than by
    #     collecting every group at the tag's rate. The two were the same
    #     question only while the tag was elevated to a value nothing else took;
    #     at the speaker's module rate they are not, and the rate-based form
    #     would sweep up the whole language model.
    tag = config["optimiser"]["polarity_embedding_lr"]
    assert tag != wanted

    holding_the_tag = [
        group for group in optimiser.param_groups
        if any(
            named[id(p)] == "sender.language_model.polarity_embedding"
            for p in group["params"]
        )
    ]

    assert len(holding_the_tag) == 1
    assert {named[id(p)] for p in holding_the_tag[0]["params"]} == {
        "sender.language_model.polarity_embedding"
    }
    assert holding_the_tag[0]["lr"] == tag


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""
Tests for the listener's score scale in code/models/receiver.py.

Runnable without pytest:  python tests/test_score_scale.py

The listener is `ReceiverGRULM + BilinearDiscriminator` or
`ReceiverCrossAttentionLM + BilinearDiscriminator`; this file calls those the
bilinear arm and the cross-attention arm, and follows each end to end because
volume is a property of the whole path. The two modules they were split out of
are named below where the history is theirs. Much of the history is an attention
discriminator's, which was removed on 2026-10-01; its tests went with it.

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
fourteen rungs could place their scores against one: the attention
discriminator had an offset of its own, and `BilinearDiscriminator`, and so
rungs 1-10, had no bias anywhere, `bilinear` being built `bias=False`. So
`ScoreVolume` gained a `score_bias` beside its volume, applied after it, with a
config key at `score_bias_lr` and a metrics column. The readout is `s * u + b`.

It opens at zero and is expected to stay near it, because the games are balanced
and the loss-optimal global offset is therefore about zero. It is insurance
against a systematic offset, and it cannot reach a per-game one.

Round nine put the volume back through
`model_util.scale_without_attenuating`, reversing round seven on the one ground
round seven's argument does not cover. AdamW and `clip_gradients` both act
*after* the backward pass, and `train.py` runs the forward under `autocast`, so
under float16 a gradient the volume has divided down can underflow to zero
before either of them sees it -- and no optimiser recovers a zero. That is the
skipped-step failure docs/anecdotes.md records. Inert under bfloat16, which has
float32's exponent range, so check the dtype before reading a result as evidence
either way.

**And on 2026-08-31 the speaker's scale came back to meet it.** `44767b2` had
deleted `log_logit_scale` in favour of a constant, on the grounds that a learned
scale climbs until the straight-through estimator is shut; that is a property of
the Jacobian `diag(p) - p pT` and it is real, but `MAX_LOGIT_SCALE` bounds it,
so the parameter can learn under a ceiling instead of being solved away. So both
ends of the channel are learned lone
scalars again, both go through `scale_without_attenuating`, and both take the
same rate -- `logit_scale_lr` and `score_scale_lr`, 6e-3 apiece. They are not
symmetric in one respect: the speaker's is bounded above at 2.0 by a projection
after the optimiser step, because a channel scale has a natural ceiling where a
volume does not. See docs/channel.md and tests/test_exploration.py.

**Round eleven, 2026-09-27, took the helper out of both ends for good.** The
speaker's scale climbs to its 2.0 ceiling in every run and stays there, so the
factor the helper hid lay in [1, 2]; the listener's volume is off by default
under the hinge; Hyperion runs bfloat16, and `GradScaler` answers underflow on
the float16 fallback. Round seven's argument stands, and the test below asserts
it again. See docs/anecdotes.md.

Consequences the tests below follow. `bilinear.weight` carries volume as well as
direction again, so `bilinear_weight_norm` is not the drift column it briefly
was.
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


# Both readout scalars, asked for explicitly rather than inherited.
#     DEFAULT.toml turned them off on 2026-09-11, when `loss = "hinge"` became
#     the default and a volume in front of a fixed margin became degenerate
#     with it. This file is *about* those two parameters, so it builds the
#     configuration that has them; `test_both_gates_off_is_the_identity` below
#     is where the new default's arrangement is pinned.
READOUT_ON = {"scale_score": True, "bias_score": True}


def _comparer(referent_dim=REFERENT_DIM, **overrides):
    """
    The bilinear arm: `ReceiverGRULM` feeding `BilinearDiscriminator`, composed
        the way `Receiver` composes them. Overrides go to the language model,
        which is where every key this arm reads lives -- `BilinearDiscriminator`
        takes nothing from its own config table -- apart from the two readout
        gates, which are `READOUT_ON` here for the reason stated beside it.
    """
    return build_listener(
        "ReceiverGRULM",
        "BilinearDiscriminator",
        referent_dim,
        language_model_overrides=overrides or None,
        discriminator_overrides=dict(READOUT_ON),
    )


CROSS_RUNG = "17_shapeworld_receiver_cross_attention_lm.toml"


def _cross_comparer(referent_dim=REFERENT_DIM, dropout=0.0, **overrides):
    """
    The cross-attention arm: `ReceiverCrossAttentionLM` feeding
        `BilinearDiscriminator`. Overrides go to the language model.

    Built from rung 17 rather than from DEFAULT, which cannot construct the
        encoder: DEFAULT's `[receiver_language_model] d_model = 1024` is the
        GRU's width and does not divide its `heads = 5`. See the note beside
        `d_model` in DEFAULT.toml.

    `dropout` is `[receiver] dropout` and defaults to off, because these are
        tests of a deterministic property and a resampled mask between two
        calls would be measuring dropout.
    """
    return build_listener(
        "ReceiverCrossAttentionLM",
        "BilinearDiscriminator",
        referent_dim,
        config_file=rung(CROSS_RUNG),
        dropout=dropout,
        language_model_overrides=overrides or None,
        discriminator_overrides=dict(READOUT_ON),
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
        BATCH, N_OBJ, listener.feature_size, generator=generator
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

def test_no_interface_norm_has_an_affine():
    """
    Every norm on the listener's input path is `Receiver`'s now, and none of
        them carries a gain. An affine there is a second route to score
        magnitude, and on a referent interface it would also break the exact
        cancellation the `bias=False` below is for.

    Both arms, in one test, because after the hoist there is one kind of object
        to check rather than two slots' worth of privately-owned norms.
    """
    for listener in (_comparer(), _cross_comparer()):
        assert listener.interfaces, "the listener declared no interfaces at all"

        for name, interface in listener.interfaces.items():
            assert interface.norm.weight is None, name
            assert interface.norm.bias is None, name


def test_the_referent_interfaces_have_no_bias_and_the_message_ones_do():
    """
    Load-bearing for the invariance below, not tidiness: the norm after a
        referent adapter can only remove the vision model's scale exactly if
        what reaches it is homogeneous in the input. `W(cx) = cW(x)` gives
        `LN(W(cx)) = LN(W(x))`; `W(cx) + b` does not.

    The norm that follows each adapter is what makes this a claim about the
        whole run rather than about initialisation: a zero-initialised bias
        would open invariant and lose it the moment it learned anything, and a
        `BatchNorm` trunk's output scale really does drift during a run.

    The message interfaces carry a bias, and the asymmetry is the point: no
        cross-backbone scale claim rides on the message side, which arrives
        through the Gumbel channel rather than off a vision model.
    """
    for listener in (_comparer(), _cross_comparer()):
        for name, interface in listener.interfaces.items():
            if name.endswith("referents"):
                assert interface.adapter.bias is None, name
                assert interface.dropout is not None, name
            else:
                assert interface.adapter.bias is not None, name
                # And no mask on a signal the channel already perturbs.
                assert interface.dropout is None, name


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
    interface = listener.interfaces[R.DISCRIMINATOR_REFERENTS]
    referents, messages = _inputs(listener)
    inflated = referents.clone()
    inflated[:, 3, :] *= 50.0

    with torch.no_grad():
        message_repr = listener.language_model(messages, None)
        message = listener.interfaces[R.DISCRIMINATOR_MESSAGE](message_repr)
        projected = listener.discriminator.bilinear(message.mean(1))
        # The interface's adapter without its norm: the width change has to
        #     happen either way, and it is the norm that is the counterfactual.
        raw = torch.einsum(
            "ijh,ih->ij", (interface.adapter(inflated), projected)
        )

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
    interface = listener.interfaces[R.LANGUAGE_MODEL_REFERENTS]
    referents, messages = _inputs(listener)
    inflated = referents.clone()
    inflated[:, 3, :] *= 50.0

    with torch.no_grad():
        # The interface's two halves, taken apart: the adapter has to run
        #     either way to reach `d_model`, and the norm is the counterfactual.
        adapted = interface.adapter(referents)
        adapted_inflated = interface.adapter(inflated)
        encoded = language_model.message_adapter(messages)

        def stage_one(values):
            return language_model.message_decoder.blocks[0].cross_attention(
                encoded, values, values
            )

        raw = stage_one(adapted)
        raw_inflated = stage_one(adapted_inflated)
        normed = stage_one(interface.norm(adapted))
        normed_inflated = stage_one(interface.norm(adapted_inflated))

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

    Both arms end in `BilinearDiscriminator`, so both buy it the same way:
        `/sqrt(referent_dim)` over two layer-normed operands. What matters is
        that neither inherits the backbone's magnitude. The band below is left
        wide rather than tightened to the exact `1/sqrt(3)`, so that it keeps
        testing the width and not the readout;
        `test_the_score_opens_at_one_over_root_three` is the tight one.
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

    Both arms open close to ln 2. They briefly did not: a fixed
        `decision_gain` of 2.0 once opened the attention comparer at 1.07,
        deliberately worse than chance on the argument that sitting at ln 2
        should never be free. See docs/anecdotes.md.
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

def test_both_gates_off_is_the_identity_and_is_the_default():
    """
    The arrangement DEFAULT.toml selects since 2026-09-11, and the one
        `experiments/hinge_vs_bce/`'s hinge arm ran: `readout` is the identity,
        neither parameter exists, and the score reaching the decision is the
        calibrated one straight out of the comparison.

    It is here rather than beside the gate tests because it is now the *default*
        arrangement, and the rest of this file overrides its way back to the
        other one. A hinge makes the volume degenerate with `train.HINGE_MARGIN`
        -- a loudness in front of a fixed margin is a margin -- and inverts the
        direction it fails in: BCE's volume goes quiet, and a hinge rewards one
        that grows until only errors sit inside the margin.

    The offset comes off with it rather than for a reason of its own. Games are
        balanced, so the loss-optimal global offset is near zero and the column
        sits there: 0.005 at epoch 99 on the BCE arm, against a score opening at
        0.577.
    """
    settings = config_section("receiver_discriminator")
    assert settings["scale_score"] is False
    assert settings["bias_score"] is False

    # Built straight from DEFAULT, where this file's other builds override
    #     their way back to `READOUT_ON`.
    bare = build_listener(
        "ReceiverGRULM", "BilinearDiscriminator", REFERENT_DIM
    ).discriminator

    scores = torch.randn(BATCH, N_OBJ)
    assert not bare.learns_score_scale
    assert not bare.learns_score_bias
    assert not hasattr(bare, "log_score_scale")
    assert not hasattr(bare, "score_bias")
    assert torch.equal(bare.readout(scores), scores)


def test_each_discriminator_owns_exactly_one_volume_and_one_offset():
    """
    One of each per arm, under one name, so one config key and one suffix reach
        both. Without the offset the bilinear arm has no bias anywhere --
        `bilinear` is built `bias=False` -- and cannot place its scores against
        `train.py`'s fixed `lis_scores > 0` at all.
    """
    for build in (_comparer, _cross_comparer):
        named = dict(build().named_parameters())
        volumes = [name for name in named if name.endswith("log_score_scale")]
        offsets = [name for name in named if name.endswith("score_bias")]

        assert len(volumes) == 1, sorted(named)
        assert len(offsets) == 1, sorted(named)


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
    Why `readout` is `score_scale * scores + score_bias` in that order.

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
        opening = discriminator.bilinear.weight.norm().item()

        with torch.no_grad():
            discriminator.log_score_scale.fill_(math.log(37.0))
            discriminator.bilinear.weight.mul_(37.0)
        discriminator.reset_parameters()

        assert discriminator.score_scale.item() == pytest.approx(1.0)

        # Not the same draw, so the norm rather than the tensor: what has to
        #     come back is the scale of the opening, which `fan_in` fixes.
        assert discriminator.bilinear.weight.norm().item() == pytest.approx(
            opening, rel=0.1
        )


def test_the_readout_still_carries_gradient_to_the_message():
    """
    The failure mode this whole change is aimed at is a listener that stops
        passing anything back. Normalising the readout must not be a way of
        doing that quietly.
    """
    listener = _cross_comparer()
    referents, messages = _inputs(listener)
    messages = messages.clone().requires_grad_(True)

    F.binary_cross_entropy_with_logits(
        listener(referents, messages), _labels()
    ).backward()

    assert messages.grad is not None
    assert messages.grad.norm().item() > 0.0


# --------------------------------------------------------------------------
# Round seven's property, restored by round eleven: the volume is a plain
#     product, so it multiplies the backward pass into everything upstream of
#     it, and AdamW is what cancels that.
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
    Round eleven, restoring what rounds seven and eight asserted. A scalar at the
        front of the score multiplies every gradient behind it, so a listener
        turning itself down scales down what reaches the speaker. That coupling
        never reaches the optimiser: AdamW updates by `m / sqrt(v)`, so a
        constant factor on a parameter's gradient cancels before it becomes a
        step. Rounds six and nine hid the volume from the backward pass with
        `scale_without_attenuating`; see docs/anecdotes.md for why round eleven
        removed it.

    Pinned so that a reader can see the factor is real and unhidden. The bounds
        are the ones round seven used, and hold with room: the gradient falls
        roughly in proportion to the volume, less a little because the loss's
        own `sigma(s*u + b) - y` moves with `s` too.
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

def _pair_and_optimiser(config_file, **section_overrides):
    """
    A rung built through `models.builder`, with a stub dataloader.

    `section_overrides` patches whole config sections after `get_config`, for a
        test that needs an arm the rung's own file does not select.
    """
    config = parse_config.get_config(rung(config_file))
    config["cuda"] = False

    for section, values in section_overrides.items():
        config[section].update(values)

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

    `score_bias` is the one this arm never had. At the base 1e-4 and birds'
        194 steps an epoch its entire thirty-epoch travel would be bounded at
        0.58, against a score whose opening spread is 0.577.

    `mix_scale_lr` has no successor: one `ScoreVolume` per discriminator means
        one key. A leftover key would be worse than a leftover parameter here --
        `split_out_parameter` raises when nothing matches its suffix,
        deliberately, so a stale key takes every rung down at construction.
    """
    config, pair, optimiser = _pair_and_optimiser(
        "02_birds_baseline.toml", receiver_discriminator=READOUT_ON
    )

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


def test_a_cross_attention_rung_with_a_normalised_channel_elevates_three_scalars_and_nothing_else():
    """
    Which parameters are left in an elevated group, on the top rung. The keys
        cannot be told apart by their rate -- DEFAULT.toml opens the scaling
        scalars and `score_bias` together -- so the assertion is on membership.

    `polarity_embedding` used to be a key here too, and is not in an optimiser
        group at all since 2026-09-28: the tag is frozen, so it is asserted
        below to be absent from every group.

    The speaker contributes its channel scale, `log_logit_scale`, which exists
        only under `normalise_logits` -- pinned on at the call below, because
        dropping it silently would weaken the assertion rather than fail it. The
        listener contributes two: its volume and its offset. A third, the
        attention discriminator's mixing weight, went with that class on
        2026-10-01.

    Exactly one `log_score_scale` and one `score_bias` on the whole listener:
        the cross-attention encoder owns neither.
    """
    config, pair, optimiser = _pair_and_optimiser(
        "18_birds_receiver_cross_attention_lm.toml",
        receiver_discriminator=READOUT_ON,
        sender_language_model={"normalise_logits": True},
    )
    wanted = config["optimiser"]["score_scale_lr"]
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

    named = {id(p): name for name, p in pair.named_parameters()}
    elevated = {
        named[id(p)]
        for group in optimiser.param_groups if group["lr"] == wanted
        for p in group["params"]
    }

    assert elevated == {
        "sender.language_model.log_logit_scale",
        "receiver.discriminator.log_score_scale",
        "receiver.discriminator.score_bias",
    }

    # The frozen tag is in no group: `get_optimiser` skips parameters that do
    #     not require a gradient, so there is no rate for it to be at.
    assert not any(
        named[id(p)] == "sender.language_model.polarity_embedding"
        for group in optimiser.param_groups
        for p in group["params"]
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""
Tests for the listener's score scale in code/models/receiver.py.

Runnable without pytest:  python tests/test_score_scale.py

Both comparers used to let the architecture set how loudly the listener stated
a conclusion, and both have now been stopped from doing it. That is the same
defect `e3fcabd` fixed in three places on the speaker, arriving one module at a
time: whatever multiplies the score also multiplies every gradient in the pair,
so a listener that quietens starves the machinery that would make it worth
listening to.

`BilinearGRUComparer` scored a referent by a raw dot product on unnormalised
backbone output, so on ViT2 the size was set by an `nn.BatchNorm1d` at the end
of broccoli's classification head and on ResNet18 by the trunk's own
normalisation -- per batch, and differently at eval.

`TransformerCrossAttentionComparer` read its score off a bare
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

That held on `BilinearGRUComparer`, which still has it, and failed on the other
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

The tests below split accordingly: shared properties stay parametrised over both
classes, `log_score_scale`'s mechanism is bilinear-only, and the cross-attention
readout has a section of its own -- which now pins the *absence* of guards and
the columns that watch what is unguarded. `29b18ea` still explains why the
surviving bilinear scale is free rather than floored: a healthy pair dips ~0.2
log-units below its opening while the message is still noise and comes back, and
flooring that cost fifteen epochs.
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import models.builder  # noqa: E402
import models.receiver as R  # noqa: E402
import parse_config  # noqa: E402

CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "ablation", "configs"
)

REFERENT_DIM = 512
BATCH, N_OBJ, SEQ = 8, 20, 7


def _comparer(referent_dim=REFERENT_DIM, **overrides):
    config = parse_config.get_config()["receiver_comparer"]
    config = {**config, **overrides}
    torch.manual_seed(0)
    return R.BilinearGRUComparer(referent_dim, **config)


def _cross_comparer(referent_dim=REFERENT_DIM, **overrides):
    """
    Built from rung 11 rather than from DEFAULT, which cannot construct this
        class: DEFAULT's `d_model = 1024` is `BilinearGRUComparer`'s GRU width
        and does not divide `heads = 5`. See the note beside `heads` in
        DEFAULT.toml.
    """
    config = parse_config.get_config(
        os.path.join(CONFIG_DIR, "11_shapeworld_receiver_cross_attention.toml")
    )["receiver_comparer"]
    config = {**config, **overrides}
    torch.manual_seed(0)
    return R.TransformerCrossAttentionComparer(referent_dim, **config)


# Every property below the first section holds of both classes, and is
#     parametrised over them rather than written twice. Where a mechanism
#     differs the test is in the class's own section further down.
BOTH = pytest.mark.parametrize(
    "build", [_comparer, _cross_comparer], ids=["bilinear", "cross_attention"]
)


def _inputs(comparer, referent_scale=1.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    referents = referent_scale * torch.randn(
        BATCH, N_OBJ, comparer.referent_embedding_size, generator=generator
    )
    messages = torch.randn(
        BATCH,
        getattr(comparer, "message_length", SEQ),
        comparer.token_embedding_size,
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
    comparer = _comparer()
    assert comparer.referent_layer_norm.weight is None
    assert comparer.referent_layer_norm.bias is None
    assert comparer.message_layer_norm.weight is None
    assert comparer.message_layer_norm.bias is None


def test_the_cross_attention_norms_that_must_be_affine_free_are():
    """
    Two of them, for two different reasons. `referent_layer_norm` because an
        affine there is a second route to score magnitude; `decision_layer_norm`
        because the RMSNorm immediately above it *does* carry a learnable gain
        -- broccoli's post-norm default, and not ours to turn off since the same
        class is used by `encoding` and by the speaker's stacks. That gain is
        a route to global score magnitude, and `decision_layer_norm` is now the
        only thing standing between it and the score. It was briefly not the
        only thing -- a standardised readout divided any global gain back out --
        so if this test is ever read as belt-and-braces, note that the braces
        were removed on purpose. The RMSNorm's affine is asserted here too, so
        that this test is what fails first if it ever goes away.
    """
    comparer = _cross_comparer()

    assert comparer.referent_layer_norm.weight is None
    assert comparer.decision_layer_norm.weight is None
    assert comparer.decision_layer_norm.bias is None
    assert comparer.referent_self_attention_norm.weight is not None


def test_the_referent_adapter_has_no_bias():
    """
    Load-bearing for the invariance below, not tidiness: the norm after it can
        only remove the vision model's scale exactly if what reaches it is
        homogeneous in the input. `W(cx) = cW(x)` gives `LN(W(cx)) = LN(W(x))`;
        `W(cx) + b` does not.
    """
    assert _cross_comparer().referent_adapter.bias is None


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
    comparer = _comparer().eval()
    referents, messages = _inputs(comparer)
    inflated = referents.clone()
    inflated[:, 3, :] *= 50.0

    with torch.no_grad():
        token_embeddings, _ = comparer.gru(messages)
        projected = comparer.bilinear(token_embeddings[:, -1, ...])
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
    comparer = _cross_comparer(referent_dim=320).eval()
    referents, messages = _inputs(comparer)
    inflated = referents.clone()
    inflated[:, 3, :] *= 50.0

    with torch.no_grad():
        adapted = comparer.referent_adapter(referents)
        adapted_inflated = comparer.referent_adapter(inflated)
        encoded = comparer.message_adapter(messages)

        def stage_one(values):
            return comparer.message_cross_attention(encoded, values, values)

        raw = stage_one(adapted)
        raw_inflated = stage_one(adapted_inflated)
        normed = stage_one(comparer.referent_layer_norm(adapted))
        normed_inflated = stage_one(
            comparer.referent_layer_norm(adapted_inflated)
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

def test_the_bilinear_scale_opens_at_one():
    assert _comparer().score_scale.item() == pytest.approx(1.0)


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
        is `decision_layer_norm`, which puts every candidate at norm `sqrt(d)`
        whatever the backbone emitted, so `decision`'s default init sets the
        opening and the referent width does not -- see
        `test_the_readout_opens_below_a_confident_wrong_answer`.
    """
    comparer = build(referent_dim=referent_dim)
    with torch.no_grad():
        scores = comparer(*_inputs(comparer))

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
    comparer = build(referent_dim=referent_dim)
    with torch.no_grad():
        scores = comparer(*_inputs(comparer))

    loss = F.binary_cross_entropy_with_logits(scores, _labels()).item()

    assert loss < 2.0 * math.log(2.0)


# --------------------------------------------------------------------------
# What the bilinear scale can and cannot do. `BilinearGRUComparer` only, since
#     it is the only class that still has one -- see the module docstring.
# --------------------------------------------------------------------------

def test_the_scale_cannot_change_the_decision():
    """
    `train.py` reads the decision as `scores > 0` and the reference-game branch
        as an argmax, and neither may move with the listener's confidence. The
        scale multiplies an operand shared across the objects of a game, so it
        cannot change which object wins -- only how loudly the listener says so.

    Which is also why the collapse was invisible in `train_acc`: this property
        is what let the loss walk to ln 2 with every prediction unchanged.
    """
    comparer = _comparer().eval()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        quiet = comparer(referents, messages)
        comparer.log_score_scale.fill_(math.log(37.0))
        loud = comparer(referents, messages)

    assert torch.equal(quiet > 0, loud > 0)
    assert torch.equal(quiet.argmax(1), loud.argmax(1))
    assert torch.allclose(loud, 37.0 * quiet, atol=1e-4)


def test_the_scale_does_change_the_loss():
    """
    Which is the whole point of having it: BCE is not scale-invariant, so this
        is the listener's one control over its own confidence. It is also the
        exposure -- the same asymmetry is what makes the descent first-order.
    """
    comparer = _comparer().eval()
    referents, messages = _inputs(comparer)
    labels = _labels()

    with torch.no_grad():
        quiet = F.binary_cross_entropy_with_logits(
            comparer(referents, messages), labels
        ).item()
        comparer.log_score_scale.fill_(math.log(37.0))
        loud = F.binary_cross_entropy_with_logits(
            comparer(referents, messages), labels
        ).item()

    assert loud > quiet


def test_the_scale_receives_gradient():
    comparer = _comparer()
    referents, messages = _inputs(comparer)

    F.binary_cross_entropy_with_logits(
        comparer(referents, messages), _labels()
    ).backward()

    assert comparer.log_score_scale.grad is not None
    assert comparer.log_score_scale.grad.abs().item() > 0.0


def test_bilinear_cannot_reach_the_score_magnitude():
    """
    `message_layer_norm` sits on `bilinear`'s output, so growing `W` changes
        the direction of the comparison and not its volume. This is what leaves
        `score_scale` as the only route, and it is the difference between the
        ordering we shipped and the one where the norm sat on the GRU state.
    """
    comparer = _comparer().eval()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        before = comparer(referents, messages)
        comparer.bilinear.weight.mul_(100.0)
        after = comparer(referents, messages)

    assert torch.allclose(before, after, atol=1e-4)


def test_reset_parameters_returns_the_scale_to_its_opening():
    comparer = _comparer()
    with torch.no_grad():
        comparer.log_score_scale.fill_(math.log(37.0))
    comparer.reset_parameters()

    assert comparer.score_scale.item() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The readout. `TransformerCrossAttentionComparer` only.
#
# It is a plain `nn.Linear(d_model, 1)` on a layer-normed input, which is what
#     it was before two attempts to take the volume out of the listener's hands
#     -- `log_score_scale`, then a fixed-gain BatchNorm. The second closed the
#     collapse it was written for and stopped four rungs learning at all. What
#     these tests pin is therefore mostly the *absence* of guards, and the
#     columns that watch what is no longer guarded.
#
# Tests here run in train mode because everything downstream of the readout
#     (dropout, stochastic depth) is mode-dependent, and because that is the
#     mode the collapse happens in.
# --------------------------------------------------------------------------

def _quiet_cross_comparer(**overrides):
    """
    A cross-attention comparer with every dropout off, so that two calls on the
        same input differ only through what the test changed. Without this the
        train-mode masks are resampled between calls and an exact-invariance
        assertion is measuring dropout.
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

    The bias is back because it was only ever dropped as a dead parameter: a
        mean subtraction downstream absorbed any constant it could add, so its
        gradient was identically zero. Nothing subtracts a mean now.
    """
    comparer = _cross_comparer(referent_dim=320)

    assert not hasattr(comparer, "log_score_scale")
    assert not hasattr(comparer, "score_norm")
    assert not hasattr(comparer, "decision_gain")
    assert comparer.decision.bias is not None


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
        `nn.Linear`'s default init and `decision_layer_norm`'s output scale,
        and pinning it would make this a change-detector for PyTorch.
    """
    comparer = _quiet_cross_comparer()
    with torch.no_grad():
        scores = comparer(*_inputs(comparer))

    loss = F.binary_cross_entropy_with_logits(scores, _labels()).item()

    assert 0.2 < scores.std().item() < 1.5
    assert math.log(2.0) < loss < 1.0


def test_the_decision_head_can_reach_the_score_magnitude():
    """
    The guard that used to be here is gone, deliberately, and this test says so
        rather than leaving its absence to be inferred.

    While the readout was standardised, scaling `decision.weight` by `c` scaled
        the pre-norm logits, their mean and their standard deviation alike, and
        the quotient did not move -- the head could turn but neither shout nor
        fall silent. That is the property `diagnostics/bootstrap_probe.py`
        measured the cost of: it also stopped the listener going quiet while the
        message was still noise, which is what the speaker needs it to do.

    So the collapse route documented in this module's docstring is open again.
        `decision_spread` is what watches it, which is why the assertion below
        is on that column and not only on the scores.
    """
    comparer = _quiet_cross_comparer()
    referents, messages = _inputs(comparer)

    with torch.no_grad():
        before = comparer(referents, messages)
        opening_spread = comparer.decision_spread

        comparer.decision.weight.mul_(10.0)
        comparer.decision.bias.mul_(10.0)
        loud = comparer(referents, messages)

    assert torch.allclose(loud, 10.0 * before, rtol=1e-4, atol=1e-5)
    assert comparer.decision_spread == pytest.approx(10.0 * opening_spread, rel=1e-4)


def test_a_constant_readout_is_reported_rather_than_hidden():
    """
    The end state of the collapse, and what the columns say when it arrives.

    A standardised readout sent a constant to 0 and sigmoid 0.5, which is why
        going quiet was not a way down. A plain one passes the constant
        through, so the listener really can sit at a fixed answer -- and the
        only defence is that it is visible: zero spread, and a kurtosis of NaN
        rather than a 0.0 that would read as "Gaussian, nothing to see".
    """
    comparer = _quiet_cross_comparer()

    class _Constant(torch.nn.Module):
        def forward(self, x):
            return torch.full((*x.shape[:-1], 1), 3.7)

    comparer.decision = _Constant()
    with torch.no_grad():
        scores = comparer(*_inputs(comparer))

    assert torch.allclose(scores, torch.full_like(scores, 3.7))
    assert comparer.decision_spread == pytest.approx(0.0, abs=1e-6)
    assert math.isnan(comparer.decision_kurtosis)


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
    """
    comparer = _quiet_cross_comparer()
    referents, messages = _inputs(comparer)
    n = referents.shape[0] * referents.shape[1]

    class _Shape(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = values

        def forward(self, x):
            return self.values.reshape(*x.shape[:-1], 1)

    bimodal = torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0)

    # 2% of the mass at +-7 and the rest at zero, matched to `bimodal`'s
    #     standard deviation so only the shape differs.
    heavy = torch.zeros(n)
    heavy[: max(2, n // 50)] = 7.0
    heavy[-max(2, n // 50):] = -7.0
    heavy = heavy * (bimodal.std() / heavy.std())

    readings = {}
    for name, values in (("bimodal", bimodal), ("heavy", heavy)):
        comparer.decision = _Shape(values)
        with torch.no_grad():
            comparer(referents, messages)
        readings[name] = (comparer.decision_spread, comparer.decision_kurtosis)

    assert readings["bimodal"][1] == pytest.approx(-2.0, abs=0.05)
    assert readings["heavy"][1] > 5.0
    assert readings["bimodal"][0] == pytest.approx(readings["heavy"][0], rel=1e-3)


def test_the_readout_still_carries_gradient_to_the_message():
    """
    The failure mode this whole change is aimed at is a listener that stops
        passing anything back. Normalising the readout must not be a way of
        doing that quietly.
    """
    comparer = _quiet_cross_comparer()
    referents, messages = _inputs(comparer)
    messages = messages.clone().requires_grad_(True)

    F.binary_cross_entropy_with_logits(
        comparer(referents, messages), _labels()
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


def test_the_scale_lands_in_a_group_at_its_own_rate():
    """
    `BilinearGRUComparer` only. The key is gated on that class now, because the
        cross-attention comparer has no scale for a rate to apply to -- see the
        companion test below.
    """
    config, pair, optimiser = _pair_and_optimiser("02_birds_baseline.toml")
    wanted = config["optimiser"]["score_scale_lr"]
    assert wanted != config["optimiser"]["lr"]

    scale = pair.receiver.comparer.log_score_scale
    holding = [
        group for group in optimiser.param_groups
        if any(p is scale for p in group["params"])
    ]

    assert len(holding) == 1
    assert holding[0]["lr"] == wanted
    assert holding[0]["weight_decay"] == 0.0


def test_a_cross_attention_rung_builds_and_asks_for_no_scale_group():
    """
    The regression this gate exists for. `split_out_parameter` raises when no
        parameter matches its suffix -- deliberately, so that a rename says so
        -- and `score_scale_lr` is set in DEFAULT.toml, so an ungated call would
        take every cross-attention rung down at construction rather than at some
        later point where the cause would be visible.

    Also checks the quieter half: no group is left holding a rate that applies
        to nothing.

    The elevated groups cannot be told apart by their rate -- DEFAULT.toml opens
        `logit_scale_lr`, `polarity_embedding_lr` and `score_scale_lr` all at
        2e-3, and this rung's `SenderTransformerLM` earns the first two -- so
        the assertion is on which parameters are in them. Both survivors are on
        the speaker; the listener contributes nothing.
    """
    config, pair, optimiser = _pair_and_optimiser(
        "12_birds_receiver_cross_attention.toml"
    )
    wanted = config["optimiser"]["score_scale_lr"]
    assert wanted != config["optimiser"]["lr"]

    assert not hasattr(pair.receiver.comparer, "log_score_scale")

    named = {id(p): name for name, p in pair.named_parameters()}
    elevated = {
        named[id(p)]
        for group in optimiser.param_groups if group["lr"] == wanted
        for p in group["params"]
    }

    assert not any(name.endswith("log_score_scale") for name in elevated)
    assert elevated == {
        "sender.language_model.log_logit_scale",
        "sender.language_model.polarity_embedding",
    }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

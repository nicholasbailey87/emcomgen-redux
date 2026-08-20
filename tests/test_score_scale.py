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

The two now share a shape. Everything that could set the score's magnitude is
normalised away -- both operands of the dot product on one, the readout
direction and its input on the other -- and what is left is a single
`log_score_scale` opening at 1.0. `29b18ea` is why it is free rather than
floored: a healthy pair dips ~0.2 log-units below its opening while the message
is still noise and comes back, and flooring that cost fifteen epochs.
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
        exactly the second route the readout normalisation exists to close, so
        the affine on it is asserted here too: if it ever went away, this test
        should be the thing that says the guard below it is now redundant.
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

@BOTH
def test_the_scale_opens_at_one(build):
    assert build().score_scale.item() == pytest.approx(1.0)


@BOTH
@pytest.mark.parametrize("referent_dim", [320, 512])
def test_the_untrained_score_opens_near_unit_variance(build, referent_dim):
    """
    On `BilinearGRUComparer` this is what `1/sqrt(referent_embedding_size)`
        buys; on the cross-attention comparer it is `decision_layer_norm`
        against a unit readout direction. Either way the opening confidence is
        the same whichever backbone the rung mounts, rather than growing with
        its width.
    """
    comparer = build(referent_dim=referent_dim).eval()
    with torch.no_grad():
        scores = comparer(*_inputs(comparer))

    assert 0.3 < scores.std().item() < 3.0


@BOTH
@pytest.mark.parametrize("referent_dim", [320, 512])
def test_untrained_bce_opens_near_ln_2(build, referent_dim):
    """
    The reason the opening confidence matters. A listener that opens by
        shouting wrong answers makes muting the fast descent direction, which
        is the state `e3fcabd` was written about.
    """
    comparer = build(referent_dim=referent_dim).eval()
    with torch.no_grad():
        scores = comparer(*_inputs(comparer))

    loss = F.binary_cross_entropy_with_logits(scores, _labels()).item()

    assert loss < 2.0 * math.log(2.0)


# --------------------------------------------------------------------------
# What the scale can and cannot do.
# --------------------------------------------------------------------------

@BOTH
def test_the_scale_cannot_change_the_decision(build):
    """
    `train.py` reads the decision as `scores > 0` and the reference-game branch
        as an argmax, and neither may move with the listener's confidence. On
        `BilinearGRUComparer` the scale multiplies an operand shared across the
        objects of a game; on the cross-attention comparer it multiplies the
        readout *including* its bias, which is what keeps the boundary at
        `scores = -bias` rather than at `-bias/scale`.
    """
    comparer = build().eval()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        quiet = comparer(referents, messages)
        comparer.log_score_scale.fill_(math.log(37.0))
        loud = comparer(referents, messages)

    assert torch.equal(quiet > 0, loud > 0)
    assert torch.equal(quiet.argmax(1), loud.argmax(1))
    assert torch.allclose(loud, 37.0 * quiet, atol=1e-4)


@BOTH
def test_the_scale_does_change_the_loss(build):
    """
    Which is the whole point of having it: BCE is not scale-invariant, so this
        is the listener's one control over its own confidence.
    """
    comparer = build().eval()
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


@BOTH
def test_the_scale_receives_gradient(build):
    comparer = build()
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


def test_the_decision_head_cannot_reach_the_score_magnitude():
    """
    The counterpart, and the direct regression test for the CUB collapse.
        `forward` normalises `decision.weight` to a unit vector, which leaves
        its gradient orthogonal to itself, so the head can turn but cannot
        shout -- and cannot fall silent, which is what it actually did.
    """
    comparer = _cross_comparer(referent_dim=320).eval()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        before = comparer(referents, messages)
        comparer.decision.weight.mul_(100.0)
        loud = comparer(referents, messages)
        comparer.decision.weight.mul_(1e-6)
        quiet = comparer(referents, messages)

    assert torch.allclose(before, loud, atol=1e-4)
    assert torch.allclose(before, quiet, atol=1e-4)


@BOTH
def test_reset_parameters_returns_the_scale_to_its_opening(build):
    comparer = build()
    with torch.no_grad():
        comparer.log_score_scale.fill_(math.log(37.0))
    comparer.reset_parameters()

    assert comparer.score_scale.item() == pytest.approx(1.0)


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


@pytest.mark.parametrize(
    "config_file",
    ["02_birds_baseline.toml", "12_birds_receiver_cross_attention.toml"],
    ids=["bilinear", "cross_attention"],
)
def test_the_scale_lands_in_a_group_at_its_own_rate(config_file):
    """
    Both comparers, because `score_scale_lr` used to be gated on the class and
        silently did nothing for the cross-attention one. That gate is gone,
        which is also what puts `split_out_parameter`'s `RuntimeError` back on
        duty as a guard against a rename.
    """
    config, pair, optimiser = _pair_and_optimiser(config_file)
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""
Tests for the exploration channel in code/models/sender.py.

Runnable without pytest:  python tests/test_exploration.py

`F.gumbel_softmax(logits, hard=True)` emits `argmax(logits + g)` with
`g ~ Gumbel(0,1)`, whose standard deviation is a fixed 1.283. That noise floor
does not move, so how many symbols survive the channel is set entirely by the
scale of the speaker's logits -- which, before any of this, was an accident of
the architecture: one arm of a ladder passed 99% of its symbols and another 24%,
with nothing in the config saying so.

Three mechanisms are at work between them, and the tests are organised around
which does what.

`layer_norm_logits` pins the emittable logits to unit variance per example and
per position, so every speaker arrives at the channel with logits of the same
magnitude whatever its architecture did. **That** is what makes exploration
comparable across arms, and it is checked directly: two logit tensors differing
by a factor of a hundred must reach the same rate.

`logit_scale` then says what that unit is worth. It is a **cap**, solved once at
construction from `token_max_probability`: the normaliser bounds how
concentrated a shape can get -- one token at `sqrt(V-1)`, the rest at
`-1/sqrt(V-1)` -- so inverting a softmax at that shape gives the constant under
which no speaker can ever hold a token above the configured probability. It does
not change over a run. It replaced an entropy solve whose scale was then learned
from, which climbed until the straight-through estimator was shut.

`estimator` chooses what the speaker learns through. Both branches emit the same
hard one-hot, and at the same seed they emit the *same* messages; they differ in
the backward pass, `"gumbel"` taking the soft sample's `diag(p) - p pT` and
`"identity"` taking `I`. The cap bounds the first; only the second removes it.

The rest is unchanged and still has to hold. The Gumbel-max identity, that a
slot's survival probability is exactly its winning token's softmax probability,
is what lets survival be measured off a softmax with no Monte Carlo. The uniform
mixture's bounds have to survive the ordering -- the scale multiplies *before*
mixing, and the reverse destroys them. Eval has to be greedy, deterministic and
invariant to every train-time knob. And the normaliser has to be
position-invariant, which BatchNorm was not.
"""

import math

import pytest
import torch
from torch.nn import functional as F

import _bootstrap  # noqa: F401
from _bootstrap import config_section

import data.language
import models.model_util as model_util
import models.sender as S
from parse_config import get_config


# ---------------------------------------------------------------- fixtures --

def _masked(logits):
    """Normalise the emittable slice and mask the reserved four, as decode does."""
    vocabulary = logits.size(-1) - 4
    return S.mask_reserved_tokens(S.layer_norm_logits(logits, vocabulary))


def _logit_shapes(vocabulary, seed=0):
    """
    A batch of logit tensors spanning the range a speaker might present.

    The named extremes matter most: `sharp` stands for the Conv4 speaker, whose
    unnormalised logits reached a standard deviation of 159 and a channel that
    passed 99% of symbols, and `flat` for the post-norm Transformer speaker,
    pinned near 1.0 and passing 33%. LayerNorm has to bring both to the same
    rate -- see `test_layer_norm_makes_exploration_scale_invariant`.
    """
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(64, 5, vocabulary + 4, generator=generator)
    peaked = base.clone()
    peaked[..., 4] += 6.0  # one token dominating every slot
    return {
        "sharp": base * 50.0,
        "flat": base * 0.05,
        "typical": base,
        "peaked": peaked,
        "mixed": base * torch.linspace(0.05, 20.0, base.size(0)).view(-1, 1, 1),
    }


def _language_model_config(**overrides):
    return config_section("sender_language_model", **overrides)


def _gru_speaker(**overrides):
    settings = _language_model_config(d_model=32, token_embedding_size=32, **overrides)
    return S.SenderGRULM(16, **settings)


def _transformer_speaker(**overrides):
    """
    The decoder arm, which is what `DEFAULT.toml`'s `bidirectional = false`
        selects and so what the ablation rungs run.
    """
    # d_model 64 over 4 heads gives head_dim 16, which is broccoli's minimum for
    # a 1-D rotary embedding; the DEFAULT.toml width would work but is slow.
    settings = _language_model_config(
        d_model=64,
        token_embedding_size=64,
        heads=4,
        layers=1,
        bidirectional=False,
        **overrides
    )
    return S.SenderTransformerLM(64, **settings)


def _transformer_latent_speaker(**overrides):
    """
    The parallel arm. Built alongside the decoder arm everywhere the exploration
        channel is tested, because the channel is the one thing the two arms
        share exactly -- same normalisation, same scale, same mixture, same
        estimator -- so a change that breaks it on one and not the other is a
        change that has leaked out of the architecture and into the sampling.
    """
    settings = _language_model_config(
        d_model=64,
        token_embedding_size=64,
        heads=4,
        layers=1,
        bidirectional=True,
        **overrides
    )
    return S.SenderTransformerLM(64, **settings)


def _prototypes(speaker, batch_size=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    size = speaker.referent_embedding_size
    return (
        torch.randn(batch_size, size, generator=generator),
        torch.randn(batch_size, size, generator=generator),
    )


def _get_knob(speaker, attribute):
    """
    Read one of the sampling knobs as a plain float.

    All three -- `logit_scale`, `uniform_weight`, `tau` -- are ordinary float
    attributes now that the scale is a constant solved at construction rather
    than a learned parameter read through a property.
    """
    value = getattr(speaker, attribute)
    return value.item() if torch.is_tensor(value) else value


def _set_knob(speaker, attribute, value):
    """
    Set one of the sampling knobs. Plain assignment: nothing here is a
    parameter, so there is no `no_grad` and no log to write through.
    """
    setattr(speaker, attribute, value)


# ------------------------------------ 1. the scale is a cap, not a target --

def test_the_sharpest_shape_is_the_one_the_normaliser_permits():
    """
    `sharpest_logit_margin` is the whole basis of the cap, so pin the shape it
    describes rather than only the number: one token at `sqrt(V-1)` and the rest
    at `-1/sqrt(V-1)` is zero-mean and unit-variance, which is what makes it the
    most concentrated arrangement `layer_norm_logits` can pass.
    """
    for V in (8, 14, 20, 64):
        shape = torch.full((V,), -1.0 / math.sqrt(V - 1))
        shape[0] = math.sqrt(V - 1)

        assert shape.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert shape.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)

        top_two = shape.topk(2).values
        assert (top_two[0] - top_two[1]).item() == pytest.approx(
            S.sharpest_logit_margin(V), abs=1e-5
        )

    # The two shipped vocabularies, quoted in DEFAULT.toml and the docs.
    assert S.sharpest_logit_margin(14) == pytest.approx(3.883, abs=0.001)
    assert S.sharpest_logit_margin(20) == pytest.approx(4.588, abs=0.001)


def test_logit_scale_delivers_the_requested_cap_exactly():
    """
    The solve is a closed-form inverse, so it is exact rather than approximate:
    at the returned constant, the sharpest permitted shape reaches
    `token_max_probability` and not a thousandth more.

    The two shipped constants are pinned numerically because they are quoted in
    DEFAULT.toml, the rung configs and the docs.
    """
    assert S.logit_scale(0.95, 14) == pytest.approx(1.419, abs=0.001)
    assert S.logit_scale(0.95, 20) == pytest.approx(1.283, abs=0.001)

    for probability in (0.5, 0.7, 0.95, 0.99):
        for vocabulary in (8, 14, 20, 64):
            scale = S.logit_scale(probability, vocabulary)
            shape = torch.full(
                (vocabulary,), -1.0 / math.sqrt(vocabulary - 1)
            )
            shape[0] = math.sqrt(vocabulary - 1)

            winner = torch.softmax(scale * shape, dim=-1).max().item()
            assert winner == pytest.approx(probability, abs=1e-6), (
                probability, vocabulary
            )

    # Monotone: a higher cap needs a bigger constant.
    scales = [S.logit_scale(p, 20) for p in (0.5, 0.7, 0.95, 0.99)]
    assert scales == sorted(scales), scales


def test_the_cap_is_a_bound_over_every_shape_not_only_the_sharpest():
    """
    The point of solving against the *sharpest* shape. `layer_norm_logits` pins
    the second moment and says nothing about the shape, so a speaker is free to
    spend its whole budget on one token -- which is the route the 2026-08-29
    ShapeWorld run actually took, at 86% of that budget. A constant chosen
    against a typical shape would not have held.

    Nothing the speaker can present may exceed the cap, so this tries the sharp
    end deliberately as well as randomly.
    """
    V, cap = 14, 0.95
    scale = S.logit_scale(cap, V)

    generator = torch.Generator().manual_seed(0)
    candidates = [
        torch.randn(4096, V + 4, generator=generator) * multiplier
        for multiplier in (0.05, 1.0, 50.0)
    ]

    # Deliberately adversarial: a spike of growing height on one token, which
    # normalises toward the sharpest shape.
    for height in (1.0, 10.0, 1000.0):
        spike = torch.randn(256, V + 4, generator=generator) * 0.01
        spike[..., 4] += height
        candidates.append(spike)

    for logits in candidates:
        masked = _masked(logits)
        unmixed = S.mean_winning_probability(masked, scale, 0.0).item()
        assert unmixed <= cap + 1e-5, unmixed

    # And it is reached, not merely respected: the sharpest shape attains it.
    sharpest = torch.full((1, V + 4), -1.0 / math.sqrt(V - 1))
    sharpest[..., :4] = 0.0
    sharpest[..., 4] = math.sqrt(V - 1)
    attained = S.mean_winning_probability(
        S.mask_reserved_tokens(sharpest), scale, 0.0
    ).item()
    assert attained == pytest.approx(cap, abs=1e-5)


def test_uniform_weight_caps_what_the_listener_sees_and_not_the_gradient():
    """
    The two ceilings are different quantities and this is why there are two
    survival columns.

    `uniform_weight` mixes in probability space, so it caps `realised_survival`
    at `1 - w + w/V` and contributes a constant to the backward pass.
    `token_max_probability` caps `unmixed_survival`, which is the winner's
    probability the straight-through Jacobian is written in. A run pinned
    against the mixture's cap can still be saturating the estimator, which is
    exactly what was missed on 2026-08-29.
    """
    V, w = 14, 0.1

    sharpest = torch.full((1, V + 4), -1.0 / math.sqrt(V - 1))
    sharpest[..., :4] = 0.0
    sharpest[..., 4] = math.sqrt(V - 1)
    masked = S.mask_reserved_tokens(sharpest)

    # At a cap the channel could not have reached before, the mixed column
    # still reads well below 1 while the unmixed one sits at the cap.
    for cap in (0.95, 0.9999):
        scale = S.logit_scale(cap, V)
        realised = S.mean_winning_probability(masked, scale, w).item()
        unmixed = S.mean_winning_probability(masked, scale, 0.0).item()

        assert unmixed == pytest.approx(cap, abs=1e-5)
        assert realised == pytest.approx((1 - w) * cap + w / V, abs=1e-5)

    # The mixture's own ceiling, which no scale can pass.
    assert (1 - w) + w / V == pytest.approx(0.90714, abs=1e-5)


@pytest.mark.parametrize(
    "build", [_gru_speaker, _transformer_speaker, _transformer_latent_speaker]
)
def test_speaker_resolves_its_scale_from_its_own_vocabulary(build):
    """
    The scale is solved at construction from the speaker's *own* vocabulary and
    the configured cap, and it is a plain float: the channel is a constant for
    the whole of a run, not a parameter that learns its way somewhere else.
    """
    speaker = build()
    solved = S.logit_scale(
        get_config()["sender_language_model"]["token_max_probability"],
        speaker.vocabulary,
    )
    assert isinstance(speaker.logit_scale, float)
    assert speaker.logit_scale == pytest.approx(solved)

    # A bigger vocabulary needs a *smaller* constant to reach the same cap:
    # there are more losing tokens to spread the remaining probability over,
    # and `sharpest_logit_margin` grows with V.
    birds = build(vocabulary=20)
    assert birds.logit_scale < speaker.logit_scale


def test_the_documented_operating_point_still_holds():
    """
    The opening is a *consequence* of the cap rather than a second knob, and
    DEFAULT.toml quotes where it lands. Numbers in comments rot; this recomputes
    them.

    The opening matters for bootstrapping: a fresh speaker's argmax barely
    varies with its input, so a confident opening means it emits near enough one
    message for everything from the first batch and the listener co-adapts to
    that before the speaker's embeddings are worth grounding anything on.
    """
    cap, w = 0.95, 0.1
    expected = {14: (1.419, 0.386, 0.862), 20: (1.283, 0.294, 0.860)}

    for V, (constant, opening, realised_cap) in expected.items():
        scale = S.logit_scale(cap, V)
        assert scale == pytest.approx(constant, abs=0.001), V

        # A freshly initialised speaker's normalised logits are i.i.d. standard
        # normal -- random weights through a linear projection whose rows are
        # independent, so nothing correlates the vocabulary dimension yet.
        generator = torch.Generator().manual_seed(0)
        logits = _masked(torch.randn(100000, V + 4, generator=generator))
        assert S.mean_winning_probability(logits, scale, 0.0).item() == (
            pytest.approx(opening, abs=0.005)
        ), V

        assert (1 - w) * cap + w / V == pytest.approx(realised_cap, abs=0.001), V

    # Both openings sit well below the cap, which is the range the speaker has
    # to work through, and well above chance.
    for V, (_, opening, _) in expected.items():
        assert 1.0 / V < opening < cap


# ------------------------------- 2. LayerNorm is what equalises the ladder --

def test_layer_norm_makes_exploration_scale_invariant():
    """
    The claim the per-batch calibration used to make, now made by the
    normaliser: two speakers whose raw logits differ by a constant factor --
    the actual difference between the arms of the ladder -- reach the same
    channel, because the factor is divided out before the scale is applied.

    This is why the solve is redundant. There is nothing left for it to adapt
    to except the *shape* of the logits, which is the speaker's own policy.
    """
    raw = _logit_shapes(14)["typical"]
    scale = S.logit_scale(0.95, 14)

    def survival(logits):
        return S.mean_winning_probability(_masked(logits), scale, 0.02).item()

    baseline = survival(raw)
    for factor in (0.1, 1.0, 10.0, 1e3, 1e5):
        assert abs(survival(raw * factor) - baseline) < 3e-4, factor

    # The named extremes of the real ladder, end to end. `flat` is sd 0.05,
    # which is where the eps floor below starts to be visible at all.
    shapes = _logit_shapes(14)
    realised = {
        name: survival(logits)
        for name, logits in shapes.items()
        if name in ("sharp", "flat", "typical")
    }
    assert max(realised.values()) - min(realised.values()) < 2e-3, realised


def test_scale_invariance_survives_a_collapsing_logit_scale():
    """
    Where the claim above runs out, and why `LAYER_NORM_EPS` is not the default.

    `F.layer_norm` divides by `sqrt(var + eps)`, so a speaker whose logits
    collapse towards `eps` gets normalised by something increasingly unlike its
    own spread and quietly receives a weaker channel -- which `logit_scale`,
    being a constant, cannot absorb the way the per-batch solve silently did.

    At the 1e-5 default this was not academic: a birds run lost realised
    survival 0.47 -> 0.17 to it over 22 epochs, because a channel that noisy
    starves the very gradient that would restore the logits. At 1e-12 the same
    collapse is absorbed four orders further out.
    """
    raw = _logit_shapes(14)["typical"]          # incoming sd ~1.0
    scale = S.logit_scale(0.95, 14)

    def survival(logits):
        return S.mean_winning_probability(_masked(logits), scale, 0.02).item()

    baseline = survival(raw)

    # The range a collapsing speaker actually travels through.
    for factor in (1e-2, 1e-3, 1e-4, 1e-5):
        assert abs(survival(raw * factor) - baseline) < 5e-3, factor

    # It still has to give out somewhere, and that somewhere is ~1e-6.
    assert abs(survival(raw * 1e-7) - baseline) > 0.1

    # And it never bites upwards.
    assert abs(survival(raw * 1e6) - baseline) < 3e-4

    # The constant is the thing under test, not the behaviour of some default.
    assert S.LAYER_NORM_EPS == 1e-12


def test_shape_still_moves_the_channel():
    """
    The other half of the same point: LayerNorm removes *scale* and only scale.
    A speaker that concentrates its mass gets a cleaner channel at the same
    setting, and that is the finding `realised_survival` exists to report --
    the thing the calibration used to erase.
    """
    scale = S.logit_scale(0.95, 14)
    shapes = _logit_shapes(14)

    typical = S.mean_winning_probability(_masked(shapes["typical"]), scale, 0.02)
    peaked = S.mean_winning_probability(_masked(shapes["peaked"]), scale, 0.02)

    assert peaked.item() > typical.item() + 0.1


def test_the_gain_multiplies_the_forward_logits():
    """
    The scaled logits are exactly `logit_scale * normalised`, so
    `realised_survival`, `unmixed_survival` and the mixture with
    `uniform_weight` all see what they always did.

    A plain product, and briefly not one. `7b10d47` put this scalar through
    `model_util.scale_without_attenuating` -- same forward, but `d/dnormalised`
    forced to 1 -- so that a falling scale would not multiply down the gradient
    reaching `outputs2vocab`, the stack and the vision model behind it. The
    helper is gone and so are the three tests that pinned its properties; what
    it was for is in docs/anecdotes.md, with the measured gradients.

    The short of it: AdamW updates by `m / sqrt(v)`, so a uniform factor on a
    parameter's gradient cancels, and `train.py`'s per-submodule
    `clip_gradients` renormalises whatever survives that. The coupling the
    helper removed was not reaching the optimiser. What it did reach was
    consistency -- `dL/dscale = x` is the true partial while
    `dL/dnormalised = J` is not -- so the stack was being shaped for a channel
    of volume 1 whatever the forward used.
    """
    speaker = _transformer_speaker()
    vocabulary = speaker.vocabulary
    raw = _logit_shapes(vocabulary)["typical"]

    normalised = S.layer_norm_logits(raw, vocabulary)

    for scale in (0.05, 1.0, 20.0):
        _set_knob(speaker, "logit_scale", scale)
        assert torch.allclose(
            normalised * speaker.logit_scale, scale * normalised, atol=1e-5
        )


def test_the_gain_carries_into_the_gradient_behind_it():
    """
    What the plain product does, stated so that a future round has the number
    rather than the intuition.

    The gradient into the raw logits is a product of three factors:

        dL/draw  =  J_gumbel(scaled)  x  d(scaled)/d(normalised)  x  d(normalised)/d(raw)

    The middle one is `logit_scale`. Below the sampler's saturation the first is
    roughly flat, so the whole thing tracks the scale -- which is what the
    helper was removing and is asserted here as the live behaviour.

    Above roughly 1.0 the product turns over, because `J_gumbel` is a function
    of `scaled = logit_scale * normalised` and the soft surrogate
    `softmax((scaled + g) / tau)` saturates. That belongs to the sampler, is
    deliberately kept, and is why the assertion below stays in the low range.

    A `"gumbel"` property, pinned explicitly rather than taken from the default.
    The whole point of `"identity"` is that its gradient does *not* do this --
    see `test_the_identity_gradient_is_invariant_to_the_scale`.
    """
    speaker = _transformer_speaker(estimator="gumbel")
    vocabulary = speaker.vocabulary
    raw = _logit_shapes(vocabulary)["typical"]

    def gradient_into_logits(scale):
        logits = raw.clone().requires_grad_(True)
        _set_knob(speaker, "logit_scale", scale)

        torch.manual_seed(0)
        onehot, _ = speaker.sample_symbols(logits)
        onehot.sum().backward()

        return logits.grad.norm().item()

    at_one_quarter = gradient_into_logits(0.25)

    assert gradient_into_logits(0.05) < at_one_quarter
    assert gradient_into_logits(1.0) > at_one_quarter


# ---------------------------------------------------------- 3. Gumbel-max id

def test_gumbel_max_identity():
    """
    The assumption `mean_winning_probability` rests on: the probability that the
    noise leaves a slot's argmax alone is exactly the winning token's softmax
    probability. If this were only approximate, measuring survival would need a
    Monte Carlo over noise draws and a seed to be reproducible.
    """
    torch.manual_seed(0)
    logits = _masked(_logit_shapes(14, seed=3)["typical"])
    mixed = S.flatten_logit_distribution(logits * 8.0, 0.02)

    predicted = mixed.softmax(-1)
    intended = predicted.argmax(-1)

    draws = 400
    survived = torch.zeros_like(intended, dtype=torch.float64)
    for _ in range(draws):
        sampled = F.gumbel_softmax(mixed, tau=1.0, hard=True, dim=-1).argmax(-1)
        survived += (sampled == intended).double()
    empirical = survived / draws

    expected = predicted.max(-1).values.double()

    # Per-slot agreement, at ~3 standard errors of a binomial with n = draws.
    tolerance = 3.0 * (expected * (1 - expected) / draws).sqrt() + 1e-3
    assert (empirical - expected).abs().le(tolerance).float().mean() > 0.99

    # And in aggregate, which is the quantity actually reported.
    assert abs(empirical.mean().item() - expected.mean().item()) < 0.005


# ---------------------------------------------------------------- 4. bounds

def test_mixing_bounds_hold_when_scale_comes_first():
    """
    Mixing caps a slot's winner at `1 - w + w/V` and floors its losers at `w/V`.
    Those bounds are the permanent exploration floor -- 1.86% of symbols flipped
    at w = 0.02, V = 14, which training cannot reduce -- so they have to hold
    however large the scale gets.
    """
    vocabulary, uniform_weight = 14, 0.02
    cap = 1.0 - uniform_weight + uniform_weight / vocabulary
    floor = uniform_weight / vocabulary

    logits = _masked(_logit_shapes(vocabulary, seed=1)["typical"])
    mixed = S.flatten_logit_distribution(logits * 500.0, uniform_weight)
    probabilities = mixed.softmax(-1)

    emittable = probabilities[..., 4:]
    assert torch.allclose(
        probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)), atol=1e-5
    )
    assert emittable.max().item() <= cap + 1e-5
    assert emittable.min().item() >= floor - 1e-6
    # The cap is not merely respected, it binds: this is where the irreducible
    # corruption rate comes from.
    assert emittable.max().item() > cap - 1e-3

    # Reserved tokens take no share of the uniform component.
    assert probabilities[..., :4].abs().max().item() == 0.0


def test_mixing_bounds_break_when_scale_comes_second():
    """
    The previous ordering -- mix, then scale -- measured p_min 0.00000 and
    p_max 1.00000, i.e. it destroyed exactly the bounds the mixture exists to
    impose. Pinned so that a future reordering fails here rather than quietly
    removing the exploration floor.
    """
    vocabulary, uniform_weight = 14, 0.02
    cap = 1.0 - uniform_weight + uniform_weight / vocabulary
    floor = uniform_weight / vocabulary

    logits = _masked(_logit_shapes(vocabulary, seed=1)["typical"])
    wrong_order = S.flatten_logit_distribution(logits, uniform_weight) * 500.0
    probabilities = wrong_order.softmax(-1)[..., 4:]

    assert probabilities.max().item() > cap
    assert probabilities.min().item() < floor


def test_uniform_weight_is_the_ceiling_on_fidelity():
    """
    `uniform_weight` is the one knob a speaker cannot out-learn. However peaked
    its logits and however large the scale, survival saturates at `1 - w + w/V`,
    so `w * (1 - 1/V)` of symbols are always flipped -- the property that keeps
    late training from committing the channel entirely.
    """
    vocabulary, uniform_weight = 20, 0.02
    cap = 1.0 - uniform_weight + uniform_weight / vocabulary

    logits = _masked(_logit_shapes(vocabulary, seed=2)["peaked"])
    for scale in (10.0, 1e3, 1e5):
        survival = S.mean_winning_probability(logits, scale, uniform_weight).item()
        assert survival <= cap + 1e-6, f"scale {scale}: {survival}"

    # It binds rather than merely bounding, and the residual is the documented
    # 0.019 for birds.
    assert survival == pytest.approx(cap, abs=1e-4)
    assert 1.0 - cap == pytest.approx(0.019, abs=1e-4)

    # With no mixture there is no ceiling, which is jayelm's CUB setting.
    assert S.mean_winning_probability(logits, 1e5, 0.0).item() > cap


def test_unmixed_survival_is_what_the_gumbel_gradient_sees():
    """
    The mixture is a ceiling on the *reported* number and not on the channel, so
    `realised_survival` cannot say how saturated the softmax actually is.

    On the `"gumbel"` branch the gradient runs through the soft sample, whose
    Jacobian is `diag(p) - p pT`, and the `p` in it is pre-mixture. On the
    2026-08-29 ShapeWorld run a reported 0.90670 against a cap of 0.90714 was an
    unmixed 0.99951 -- so `1 - p` was 4.9e-4 where the mixed column suggested
    0.093, a factor of 190 in the gradient that was invisible.

    Named for the branch deliberately. Under `estimator = "identity"` the
    Jacobian is `I` and this column reaches the gradient not at all; it is still
    the channel's fidelity there, but it is no longer a gradient diagnostic.
    `token_max_probability` now bounds it on both branches.

    The two are the same function with the mixture switched off, which is what
    stops them drifting, and the mixture is affine in the model's probability,
    so the unmixed value is recoverable exactly.
    """
    vocabulary, uniform_weight = 14, 0.1
    cap = 1.0 - uniform_weight + uniform_weight / vocabulary
    logits = _masked(_logit_shapes(vocabulary, seed=3)["peaked"])

    # Never below the mixed reading: mixing in a uniform can only flatten.
    for scale in (0.5, 2.0, 10.0, 1e3):
        mixed = S.mean_winning_probability(logits, scale, uniform_weight).item()
        unmixed = S.mean_winning_probability(logits, scale, 0.0).item()
        assert unmixed >= mixed - 1e-6, f"scale {scale}: {unmixed} < {mixed}"

        # `p_mixed = (1 - w) * p_model + w / V`, inverted.
        recovered = (mixed - uniform_weight / vocabulary) / (1.0 - uniform_weight)
        assert recovered == pytest.approx(unmixed, abs=1e-5)

    # And the point of the column: at the ceiling the mixed reading is pinned
    # while the unmixed one keeps going, which is the regime that kills a run.
    assert mixed == pytest.approx(cap, abs=1e-4)
    assert unmixed > 0.999


def test_logit_prior_share_separates_confident_from_mute():
    """
    A peaked distribution whose peak moves with the input is a perfect channel;
    one that peaks on the same token every time is confidence with no
    information. `realised_survival`, `logit_scale` and `logit_margin` are
    identical in the two cases -- this is the column that is not.

    The decomposition is orthogonal, so the share is exactly the fraction of the
    logits' energy that is common to every input.
    """
    vocabulary = 14

    def normalised(raw):
        return S.mask_reserved_tokens(S.layer_norm_logits(raw, vocabulary))

    # A speaker with nothing in common across inputs sits at about 1/batch:
    # the mean of B independent rows carries 1/B of their energy.
    for batch in (8, 32, 128):
        generator = torch.Generator().manual_seed(0)
        raw = torch.randn(batch, 5, vocabulary + 4, generator=generator)
        share = S.logit_prior_share(normalised(raw)).item()
        assert share == pytest.approx(1.0 / batch, rel=0.35), (batch, share)

    # A speaker emitting the same logits whatever it saw sits at exactly 1.
    generator = torch.Generator().manual_seed(1)
    one = torch.randn(1, 5, vocabulary + 4, generator=generator)
    collapsed = one.expand(32, 5, vocabulary + 4).contiguous()
    assert S.logit_prior_share(normalised(collapsed)).item() == pytest.approx(1.0)

    # And it moves monotonically between the two as the shared part grows.
    varied = torch.randn(32, 5, vocabulary + 4, generator=generator)
    shares = [
        S.logit_prior_share(
            normalised(one * common + varied * (1.0 - common))
        ).item()
        for common in (0.0, 0.25, 0.5, 0.75)
    ]
    assert shares == sorted(shares), shares

    # A ratio of two sums of squares, so the gain cannot move it -- which is why
    # it reads the shape where `logit_scale` reads the volume.
    reference = S.logit_prior_share(normalised(varied)).item()
    for scale in (0.1, 3.0, 40.0):
        assert S.logit_prior_share(
            normalised(varied) * scale
        ).item() == pytest.approx(reference, rel=1e-5)

    # A batch of one is its own mean, so the share is 1 by construction and
    # means nothing. NaN rather than a confident-looking 1.0.
    assert math.isnan(S.logit_prior_share(normalised(one)).item())


def test_the_shape_budget_is_bounded_by_the_normaliser():
    """
    Why `logit_margin` and `logit_prior_share` are worth having at all: fidelity
    does not have to come from `logit_scale`, so a clamp on the scale does not
    bound saturation.

    `layer_norm_logits` fixes the logits' second moment, and the most
    concentrated shape that allows is one token at `sqrt(V-1)` and the rest at
    `-1/sqrt(V-1)`. That reaches survival 0.717 against a 0.907 ceiling at a
    scale of *one*, which is most of the way to a committed channel with the
    scale untouched.
    """
    vocabulary, uniform_weight = 14, 0.1
    low = -1.0 / math.sqrt(vocabulary - 1)
    high = -(vocabulary - 1) * low

    extreme = torch.full((1, 1, vocabulary + 4), low)
    extreme[..., :4] = float("-inf")
    extreme[..., 4] = high

    # It is a legal output of the normaliser: zero mean, unit variance.
    emittable = extreme[..., 4:]
    assert emittable.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert emittable.var(unbiased=False).item() == pytest.approx(1.0)

    assert S.mean_logit_margin(extreme).item() == pytest.approx(3.883, abs=1e-3)

    at_unit_scale = S.mean_winning_probability(extreme, 1.0, uniform_weight)
    assert at_unit_scale.item() == pytest.approx(0.717, abs=1e-3)

    # And the cap is reachable from here with a scale under 3.
    cap = 1.0 - uniform_weight + uniform_weight / vocabulary
    assert S.mean_winning_probability(
        extreme, 2.62, uniform_weight
    ).item() == pytest.approx(cap, abs=1e-3)


@pytest.mark.parametrize(
    "build", [_gru_speaker, _transformer_speaker, _transformer_latent_speaker]
)
def test_both_survival_columns_reach_metrics(build):
    """
    The pair has to be written by every speaker arm, be NaN before a train pass
    the way `realised_survival` is, and collapse onto each other exactly when
    there is no mixture to remove -- which is CUB's setting.
    """
    torch.manual_seed(0)
    speaker = build().train()

    assert math.isnan(speaker.unmixed_survival)
    assert math.isnan(speaker.logit_margin)
    assert math.isnan(speaker.logit_prior_share)

    speaker.decode(_prototypes(speaker))
    assert speaker.unmixed_survival >= speaker.realised_survival - 1e-6
    assert speaker.logit_margin > 0.0
    assert 0.0 <= speaker.logit_prior_share <= 1.0

    unmixed = build(uniform_weight=0.0).train()
    unmixed.decode(_prototypes(unmixed))
    assert unmixed.unmixed_survival == pytest.approx(
        unmixed.realised_survival, abs=1e-6
    )


def test_logit_margin_reads_the_top_two_gap_in_sd_units():
    """
    The shape parameter that saturates the channel alongside the scale, and the
    one `layer_norm_logits` leaves free: it pins the second moment and says
    nothing about the gap between the top two.

    Constructed so the answer is known: standard normal logits with a fixed
    amount added to one token, after normalisation.
    """
    vocabulary = 14
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(64, 5, vocabulary + 4, generator=generator)

    # A known gap, read off the normalised tensor the speaker actually samples.
    normalised = S.mask_reserved_tokens(S.layer_norm_logits(raw, vocabulary))
    emittable = normalised[..., 4:]
    top_two = emittable.topk(2, dim=-1).values
    expected = (top_two[..., 0] - top_two[..., 1]).mean().item()

    assert S.mean_logit_margin(normalised).item() == pytest.approx(expected)

    # The reserved slots are -inf and must not be picked up as the runner-up.
    assert math.isfinite(S.mean_logit_margin(normalised).item())

    # Homogeneous in the logits, which is exactly why it must be fed the
    # *pre-gain* tensor: fed the scaled one it would report `scale * margin` and
    # stop being the independent shape reading `logit_scale` cannot give.
    for scale in (0.5, 3.0, 50.0):
        assert S.mean_logit_margin(normalised * scale).item() == pytest.approx(
            expected * scale, rel=1e-5
        )

    # It moves with the shape, which `logit_scale` cannot see. Sharpening one
    # token raises the margin while the normaliser holds the variance at 1.
    peaked = raw.clone()
    peaked[..., 4] += 6.0
    sharper = S.mean_logit_margin(
        S.mask_reserved_tokens(S.layer_norm_logits(peaked, vocabulary))
    ).item()
    assert sharper > expected + 0.5, (sharper, expected)

    # A Gaussian sanity check against the documented reference: the expected
    # top-two spacing of V standard normals is about `1 / sqrt(2 ln V)`.
    assert expected == pytest.approx(1.0 / math.sqrt(2.0 * math.log(vocabulary)), abs=0.2)


# --------------------------------------------------- 5. eval is the policy --

def _decode_message(speaker, prototypes):
    return speaker.decode(prototypes)[0].argmax(-1)


@pytest.mark.parametrize(
    "build", [_gru_speaker, _transformer_speaker, _transformer_latent_speaker]
)
def test_eval_is_deterministic_and_greedy(build):
    """
    Eval decodes greedily, so repeated calls agree and the message is the argmax
    of the normalised, masked logits. Both speakers previously sampled here, so
    every reported accuracy and topsim was read through the noisy channel.
    """
    torch.manual_seed(0)
    speaker = build().eval()
    prototypes = _prototypes(speaker)

    with torch.no_grad():
        first = _decode_message(speaker, prototypes)
        second = _decode_message(speaker, prototypes)

    assert torch.equal(first, second)

    # Reserved tokens are -inf, so they can never be emitted mid-message; SOS
    # and EOS are attached by the speaker at the two ends.
    content = first[:, 1:-1]
    assert content.min().item() >= 4

    with torch.no_grad():
        greedy = _greedy_reference(speaker, prototypes)
    assert torch.equal(content, greedy)


def _greedy_reference(speaker, prototypes):
    """Recompute the greedy message straight from the logits, independently."""
    if isinstance(speaker, S.SenderTransformerLM) and speaker.bidirectional:
        logits = speaker.outputs2vocab(speaker.embeddings(prototypes))
        logits = S.layer_norm_logits(logits, speaker.vocabulary)
        return S.mask_reserved_tokens(logits).argmax(-1)

    if isinstance(speaker, S.SenderTransformerLM):
        # The causal arm, which is autoregressive and so needs the loop. Built as
        #     a growing prefix over a *blanked* tail rather than as `decode`'s
        #     in-place overwrite, deliberately: the two agree only if the causal
        #     mask really does stop the slots after the one being read from
        #     reaching it, so this reference checks that claim as a side effect
        #     of checking the greedy policy. `decode` leaves those slots holding
        #     their latent vectors; this one zeroes them.
        latents = speaker.latent_layer_norm(speaker.encode(prototypes))
        rows = list(latents.unbind(1))

        tokens = []
        for i in range(speaker.content_length):
            slot = speaker.first_message_slot + i

            sequence = torch.stack(
                rows[: slot + 1]
                + [torch.zeros_like(rows[0])] * (speaker.latent_length - slot - 1),
                dim=1,
            )

            logits = speaker.outputs2vocab(speaker.transformer(sequence)[:, slot, :])
            logits = S.layer_norm_logits(logits, speaker.vocabulary)
            chosen = S.mask_reserved_tokens(logits).argmax(-1)
            tokens.append(chosen)
            rows[slot] = (
                F.one_hot(chosen, speaker.vocabulary + 4).float()
                @ speaker.token_embedding.weight
            )

        return torch.stack(tokens, 1)

    # The GRU is autoregressive, so the reference has to run the loop too.
    batch_size = prototypes[0].size(0)
    states = (
        speaker.init_h(torch.cat(prototypes, 1))
        .view(batch_size, speaker.layers, speaker.directions, speaker.d_model)
        .permute(1, 2, 0, 3).contiguous()
        .view(speaker.layers * speaker.directions, batch_size, speaker.d_model)
    )
    onehot = torch.zeros(batch_size, 1, speaker.vocabulary + 4)
    onehot[:, 0, 1] = 1.0  # SOS
    gru_in = onehot @ speaker.token_embedding.weight

    tokens = []
    for _ in range(speaker.message_length - 2):
        gru_out, states = speaker.gru(gru_in, states)
        logits = speaker.outputs2vocab(gru_out[:, -1, :])
        logits = S.layer_norm_logits(logits, speaker.vocabulary)
        logits = S.mask_reserved_tokens(logits)
        chosen = logits.argmax(-1)
        tokens.append(chosen)
        onehot = F.one_hot(chosen, speaker.vocabulary + 4).float().unsqueeze(1)
        gru_in = onehot @ speaker.token_embedding.weight

    return torch.stack(tokens, 1)


@pytest.mark.parametrize(
    "build", [_gru_speaker, _transformer_speaker, _transformer_latent_speaker]
)
def test_eval_is_invariant_to_the_training_knobs(build):
    """
    argmax is invariant to a positive rescale, to the uniform mixture and to
    tau, so none of the three may touch an eval message. If one does, eval is
    not measuring the policy.
    """
    torch.manual_seed(0)
    speaker = build().eval()
    prototypes = _prototypes(speaker)

    with torch.no_grad():
        reference = _decode_message(speaker, prototypes)

        for attribute, value in (
            ("uniform_weight", 0.5),
            ("tau", 7.0),
            ("logit_scale", 137.0),
            ("logit_scale", 0.001),
        ):
            original = _get_knob(speaker, attribute)
            _set_knob(speaker, attribute, value)
            assert torch.equal(_decode_message(speaker, prototypes), reference), (
                f"eval message changed with {attribute} = {value}"
            )
            _set_knob(speaker, attribute, original)


# ------------------------------------ 6. survival is measured, not targeted --

@pytest.mark.parametrize(
    "build", [_gru_speaker, _transformer_speaker, _transformer_latent_speaker]
)
def test_realised_survival_reports_the_channel_in_use(build):
    """
    `realised_survival` is a measurement at the fixed scale, so raising the
    scale must raise it. Under the calibration this was flat by construction --
    it restated `1 - token_exploration_rate` whatever the speaker did -- which
    is precisely the diagnostic that was lost.
    """
    torch.manual_seed(0)
    speaker = build().train()
    prototypes = _prototypes(speaker)

    assert math.isnan(speaker.realised_survival)

    speaker.decode(prototypes)
    baseline = speaker.realised_survival
    assert 0.0 < baseline < 1.0

    _set_knob(speaker, "logit_scale", speaker.logit_scale * 20.0)
    speaker.decode(prototypes)
    assert speaker.realised_survival > baseline + 0.1

    # A freshly initialised speaker starts genuinely uncertain, with room to
    # earn confidence rather than beginning at its own ceiling. At the default
    # `token_max_probability` of 0.95 that is an argmax it holds a bit over a
    # third of the time -- see `test_the_documented_operating_point_still_holds`
    # for where that number comes from.
    fresh = build().train()
    fresh.decode(_prototypes(fresh))
    assert 0.24 < fresh.realised_survival < 0.46, fresh.realised_survival


@pytest.mark.parametrize(
    "build", [_gru_speaker, _transformer_speaker, _transformer_latent_speaker]
)
def test_survival_does_not_move_during_eval(build):
    """Eval samples nothing, so it measures nothing."""
    torch.manual_seed(0)
    speaker = build().train()
    prototypes = _prototypes(speaker)

    speaker.decode(prototypes)
    trained = speaker.realised_survival
    spread = speaker.logit_spread

    speaker.eval()
    with torch.no_grad():
        for _ in range(3):
            speaker.decode(prototypes)

    assert speaker.realised_survival == trained
    assert speaker.logit_spread == spread


@pytest.mark.parametrize(
    "build", [_gru_speaker, _transformer_speaker, _transformer_latent_speaker]
)
def test_logit_spread_reads_the_pre_norm_scale(build):
    """
    `logit_spread` is what separates a speaker learning a flatter policy from
    its logit scale collapsing -- the two look identical in `realised_survival`,
    and a birds run was lost to the second while the first was assumed.

    It has to be read *before* normalisation, so it must track a rescaling of
    the speaker's output layer that `realised_survival` is (correctly) blind to.
    """
    torch.manual_seed(0)
    speaker = build().train()
    prototypes = _prototypes(speaker)

    assert math.isnan(speaker.logit_spread)

    torch.manual_seed(1)
    speaker.decode(prototypes)
    before, survival = speaker.logit_spread, speaker.realised_survival
    assert before > 0.0

    torch.manual_seed(0)
    collapsed = build().train()
    with torch.no_grad():
        collapsed.outputs2vocab.weight.mul_(1e-3)
        collapsed.outputs2vocab.bias.mul_(1e-3)
    torch.manual_seed(1)
    collapsed.decode(prototypes)

    # The spread follows the collapse; survival does not, because the normaliser
    # absorbs it. That division of labour is the point of logging both -- and at
    # the 1e-5 default this same collapse drops survival to 0.09.
    assert collapsed.logit_spread == pytest.approx(before * 1e-3, rel=0.1)
    assert collapsed.realised_survival == pytest.approx(survival, abs=5e-3)


@pytest.mark.parametrize(
    "build", [_gru_speaker, _transformer_speaker, _transformer_latent_speaker]
)
def test_no_calibration_state_reaches_the_checkpoint(build):
    """
    The scale is a constant, so nothing about exploration is checkpointed. Also
    guards the two normalisers that were removed on the way here: no BatchNorm
    running statistics and no affine LayerNorm parameters on the logits.
    """
    speaker = build()
    speaker.train()
    speaker.decode(_prototypes(speaker))

    keys = list(speaker.state_dict())
    assert not any("exploration" in key for key in keys), keys
    assert not any("batch_norm" in key for key in keys), keys

    # And a round trip does not need it: two speakers sharing a state_dict
    # sample through the same channel.
    restored = build()
    restored.load_state_dict(speaker.state_dict())
    assert restored.logit_scale == speaker.logit_scale


# ------------------------------------------------- 7. position invariance --

@pytest.mark.parametrize(
    "build", [_gru_speaker, _transformer_speaker, _transformer_latent_speaker]
)
def test_layer_norm_is_position_invariant(build):
    """
    A per-position offset added to every token of a slot survives normalisation
    as a uniform rescale of that slot -- it cannot be annihilated at one
    position and preserved at another. BatchNorm did exactly that: it removed
    per-position offsets for the GRU, which normalises one position at a time,
    and kept them for the Transformer, which normalises all of them together.
    Nobody should reintroduce a batch-statistic normaliser without this failing.
    """
    vocabulary = 14
    torch.manual_seed(0)
    logits = torch.randn(8, 5, vocabulary + 4)

    offsets = torch.linspace(-3.0, 3.0, logits.size(1)).view(1, -1, 1)
    scales = torch.linspace(0.5, 4.0, logits.size(1)).view(1, -1, 1)

    plain = S.layer_norm_logits(logits, vocabulary)[..., 4:]
    shifted = S.layer_norm_logits(logits + offsets, vocabulary)[..., 4:]
    rescaled = S.layer_norm_logits(logits * scales, vocabulary)[..., 4:]

    assert torch.allclose(plain, shifted, atol=1e-4)
    assert torch.allclose(plain, rescaled, atol=1e-4)

    # Every slot arrives at the same magnitude, which is what `logit_scale` is
    # expressed against.
    per_slot_sd = plain.std(-1, unbiased=False)
    assert (per_slot_sd - 1.0).abs().max().item() < 1e-3

    # And the reserved columns are left alone rather than folded into the
    # statistics, so they cannot drag the emittable ones around.
    assert torch.equal(
        S.layer_norm_logits(logits, vocabulary)[..., :4], logits[..., :4]
    )

    # The same two properties on the logits a real speaker actually produces,
    # captured off its vocabulary projection during a real training decode. The
    # GRU projects one position per call and the Transformer all of them at
    # once, which is precisely where BatchNorm treated the two differently.
    speaker = build().train()
    captured = []
    handle = speaker.outputs2vocab.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach())
    )
    speaker.decode(_prototypes(speaker))
    handle.remove()

    raw = torch.stack(captured, 1) if captured[0].dim() == 2 else captured[0]
    assert raw.size(1) == speaker.message_length - 2

    real_offsets = torch.linspace(-5.0, 5.0, raw.size(1)).view(1, -1, 1)
    normed = S.layer_norm_logits(raw, speaker.vocabulary)[..., 4:]
    normed_shifted = S.layer_norm_logits(
        raw + real_offsets, speaker.vocabulary
    )[..., 4:]

    assert torch.allclose(normed, normed_shifted, atol=1e-4)
    assert (normed.std(-1, unbiased=False) - 1.0).abs().max().item() < 1e-3


# ------------------------------------------- 8. the two gradient estimators --
#
# The forward pass is the same on both branches -- a hard one-hot drawn as
# `argmax(logits + Gumbel)` -- so everything above applies to both and only the
# backward pass is at issue here.
#
# What `"identity"` is for is *rank*, not magnitude. The per-token gradients are
# summed into one vector before they reach the language model and the vision
# trunk, and `diag(p) - p pT` at `p` near one-hot has rank ~1, so all but one
# direction is gone before any optimiser or clipper sees it. Magnitude largely
# cancels in AdamW; a rank does not come back.

_ESTIMATOR_BUILDS = [
    _gru_speaker, _transformer_speaker, _transformer_latent_speaker
]


def _upstream_gradient(shape, seed=7):
    """
    A random `dL/dy` to backward with.

    Not `onehot.sum()`, which several older tests use: a one-hot's entries sum
    to a constant, so that objective's `dL/dy` is all-ones and
    `(diag(p) - p pT) @ 1` is exactly zero. It measures the residual rather than
    the estimator, and on the identity branch it measures `layer_norm_logits`
    alone.
    """
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _normalised_and_onehot(speaker, raw, seed=0):
    """
    Run `sample_symbols` and hand back both the one-hot and the tensor it was
    taken from, with the graph intact.
    """
    logits = raw.clone().requires_grad_(True)
    torch.manual_seed(seed)
    onehot, _pre_gain = speaker.sample_symbols(logits)
    return logits, onehot


@pytest.mark.parametrize("build", _ESTIMATOR_BUILDS)
def test_the_identity_surrogate_forwards_exactly_the_one_hot(build):
    """
    The surrogate changes the backward pass and nothing else. If it moved the
    forward value the listener would be reading something that is not a message,
    and every downstream measurement would be measuring the estimator.

    Bit-exact, and that is a stronger claim than it looks. `z - z.detach()` is
    algebraically zero but `onehot + z - z.detach()` associates left, so it
    computes `(1 + z) - z` and lands on 1.0000001 in float32 -- a perturbation
    of the winning token on every step. Forming the zero first is what makes the
    addition exact, so this is pinned against the gumbel branch rather than
    against a tolerance.
    """
    speaker = build(estimator="identity").train()
    raw = _logit_shapes(speaker.vocabulary)["typical"]

    _logits, onehot = _normalised_and_onehot(speaker, raw)
    values = onehot.detach()

    assert torch.equal(values, values.round())
    assert torch.equal(values.sum(-1), torch.ones_like(values.sum(-1)))
    assert values[..., :4].abs().max().item() == 0.0

    # The two branches agree to the last bit, which says the surrogate
    # contributes exactly nothing to the forward value.
    gumbel = build(estimator="gumbel").train()
    _logits, from_gumbel = _normalised_and_onehot(gumbel, raw)
    assert torch.equal(values, from_gumbel.detach())


@pytest.mark.parametrize("build", _ESTIMATOR_BUILDS)
def test_the_identity_gradient_is_the_upstream_gradient(build):
    """
    `dL/dnormalised == dL/dy` exactly on the emittable slice, which is the whole
    claim: the speaker's gradient becomes the receiver's per-token embedding
    sensitivity, with nothing in between.

    Taken at the channel rather than through `sample_symbols`, so what is
    asserted is the surrogate itself and not `layer_norm_logits` composed with
    it.
    """
    speaker = build(estimator="identity").train()
    vocabulary = speaker.vocabulary
    raw = _logit_shapes(vocabulary)["typical"]

    normalised = (
        S.layer_norm_logits(raw, vocabulary).detach().requires_grad_(True)
    )

    torch.manual_seed(0)
    with torch.no_grad():
        sampled = speaker._gumbel_sample(normalised)

    emittable = normalised[..., 4:]
    onehot = torch.cat(
        [sampled[..., :4], sampled[..., 4:] + (emittable - emittable.detach())],
        dim=-1,
    )

    upstream = _upstream_gradient(onehot.shape)
    gradient, = torch.autograd.grad((onehot * upstream).sum(), normalised)

    assert torch.equal(gradient[..., 4:], upstream[..., 4:])
    assert gradient[..., :4].abs().max().item() == 0.0


@pytest.mark.parametrize("build", _ESTIMATOR_BUILDS)
def test_the_identity_gradient_is_invariant_to_the_scale(build):
    """
    The exact inverse of `test_the_gain_carries_into_the_gradient_behind_it`.
    The surrogate taps the *unscaled* logits, so the constant leaves the
    backward pass altogether and the same seed gives a bit-identical gradient
    across three orders of magnitude of it.

    The scale is a constant now, so this is not a claim about a run's
    trajectory. It is what makes the estimator indifferent to where
    `token_max_probability` was set.
    """
    raw = _logit_shapes(_gru_speaker().vocabulary)["typical"]
    upstream = None

    gradients = []
    for scale in (0.05, 1.0, 20.0):
        speaker = build(estimator="identity").train()
        _set_knob(speaker, "logit_scale", scale)

        logits, onehot = _normalised_and_onehot(speaker, raw)
        if upstream is None:
            upstream = _upstream_gradient(onehot.shape)
        (onehot * upstream).sum().backward()
        gradients.append(logits.grad.clone())

    for other in gradients[1:]:
        assert torch.equal(gradients[0], other)


@pytest.mark.parametrize("build", _ESTIMATOR_BUILDS)
def test_the_identity_gradient_survives_saturation(build):
    """
    The assertion the whole change exists for.

    Saturation is a property of the logits' *shape*, not their magnitude --
    `layer_norm_logits` divides magnitude out -- so this interpolates toward the
    sharpest shape the normaliser permits, which is the route the 2026-08-29
    ShapeWorld run actually took. At the far end the speaker sits at
    `token_max_probability`, where `diag(p) - p pT` has effectively collapsed to
    rank one.

    The inputs are normalised *before* being handed to the speaker so that
    `layer_norm_logits`'s own Jacobian is comparable at the two ends and what is
    being compared is the estimator alone. Without that the raw magnitude grows
    with the interpolation and shrinks both branches together, which is a fact
    about the normaliser and not about the estimator.
    """
    vocabulary = _gru_speaker().vocabulary

    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(64, 5, vocabulary + 4, generator=generator)
    spike = torch.zeros_like(noise)
    spike[..., 4] = 10.0

    upstream = _upstream_gradient(noise.shape)

    def gradient_norm(estimator, mixing):
        speaker = build(estimator=estimator).train()
        raw = S.layer_norm_logits(
            (1.0 - mixing) * noise + mixing * spike, vocabulary
        ).detach()

        logits, onehot = _normalised_and_onehot(speaker, raw)
        (onehot * upstream).sum().backward()
        return logits.grad[..., 4:].norm().item()

    flat, saturated = 0.0, 1.0

    # Confirm the far end really is saturated, so the comparison is about what
    # it claims to be.
    sharpest = _masked(
        S.layer_norm_logits(spike, vocabulary).detach()
    )
    speaker = build(estimator="gumbel")
    assert S.mean_winning_probability(
        sharpest, speaker.logit_scale, 0.0
    ).item() == pytest.approx(
        get_config()["sender_language_model"]["token_max_probability"],
        abs=1e-4,
    )

    gumbel_ratio = gradient_norm("gumbel", saturated) / gradient_norm(
        "gumbel", flat
    )
    identity_ratio = gradient_norm("identity", saturated) / gradient_norm(
        "identity", flat
    )

    # The gumbel branch loses most of its gradient to shape alone, and only
    # this little because `token_max_probability` bounds how sharp it can get.
    assert gumbel_ratio < 0.3, gumbel_ratio

    # The identity branch does not notice.
    assert identity_ratio > 0.95, identity_ratio


@pytest.mark.parametrize("build", _ESTIMATOR_BUILDS)
def test_the_surrogate_leaves_the_reserved_slots_alone(build):
    """
    The `-inf` trap. `masked` holds `-inf` in the four reserved columns and
    `-inf - (-inf)` is NaN, so the surrogate is built on the emittable slice
    instead. That also means the reserved columns take no gradient at all:
    `outputs2vocab` rows 0-3 and the stack behind them are never trained toward
    tokens that cannot be emitted.
    """
    for estimator in ("gumbel", "identity"):
        speaker = build(estimator=estimator).train()
        raw = _logit_shapes(speaker.vocabulary)["peaked"]

        logits, onehot = _normalised_and_onehot(speaker, raw)
        onehot.sum().backward()

        assert not torch.isnan(logits.grad).any(), estimator
        assert not torch.isinf(logits.grad).any(), estimator
        assert logits.grad[..., :4].abs().max().item() == 0.0, estimator


@pytest.mark.parametrize("build", _ESTIMATOR_BUILDS)
def test_both_estimators_emit_the_same_messages(build):
    """
    The control that makes an A/B between the branches worth running. Both go
    through the same `_gumbel_sample`, and `hard=True` emits
    `argmax(logits + g)`, which is invariant to `tau` -- so at the same seed the
    two branches emit not similar messages but identical ones, and any
    difference in a run is the backward pass and nothing else.
    """
    raw = _logit_shapes(_gru_speaker().vocabulary)["typical"]

    messages = {}
    for estimator in ("gumbel", "identity"):
        speaker = build(estimator=estimator).train()
        _logits, onehot = _normalised_and_onehot(speaker, raw)
        messages[estimator] = onehot.detach().argmax(-1)

    assert torch.equal(messages["gumbel"], messages["identity"])


@pytest.mark.parametrize("build", _ESTIMATOR_BUILDS)
def test_eval_is_the_same_policy_under_both_estimators(build):
    """
    `sample_symbols` returns before the branch outside training, so eval is
    greedy, deterministic and identical on the two. It has to be: the estimator
    is a training-time choice, and a difference here would mean the two branches
    were being scored on different policies.
    """
    raw = _logit_shapes(_gru_speaker().vocabulary)["typical"]

    emitted = {}
    for estimator in ("gumbel", "identity"):
        speaker = build(estimator=estimator).eval()
        torch.manual_seed(0)
        onehot, pre_gain = speaker.sample_symbols(raw)

        assert pre_gain is None, estimator
        emitted[estimator] = onehot.argmax(-1)

        # And still greedy: the argmax of the normalised, masked logits.
        expected = _masked(raw).argmax(-1)
        assert torch.equal(emitted[estimator], expected), estimator

    assert torch.equal(emitted["gumbel"], emitted["identity"])


# ------------------------------------------------------------ 9. the config --

def test_config_rejects_a_missing_or_invalid_token_max_probability():
    """
    `SafeDict` only warns on a missing key and hands back None, which would fail
    confusingly deep inside the decode, so `parse_config` checks it up front.

    Both bounds are real rather than defensive. `token_max_probability` is a
    ceiling on the winning token's probability, and `logit_scale` inverts it: at
    or below `1/V` there is no positive constant that flattens a distribution
    that far, and at 1 the constant is infinite. Anyone who reads the key as a
    percentage and writes `95` must be told, not handed an unusable channel.
    """
    import parse_config

    vocabulary = get_config()["sender_language_model"]["vocabulary"]

    for bad in (None, 0.0, -1.0, 1.0, 1.5, 95, 1.0 / vocabulary):
        config = get_config()
        config["sender_language_model"]["token_max_probability"] = bad
        with pytest.raises(
            parse_config.InvalidConfig, match="token_max_probability"
        ):
            parse_config.validate_config(config)

    # And the shipped value passes, so the bounds are not merely tight.
    parse_config.validate_config(get_config())


def test_config_rejects_an_unknown_estimator():
    """
    The two branches differ only in the backward pass, so a typo here would run
    a whole experiment under the wrong estimator and look like a result rather
    than a mistake. Same reason as the check above, and the same `SafeDict`
    behaviour behind it.
    """
    import parse_config

    for bad in (None, "", "gumble", "Identity", "straight-through", True):
        config = get_config()
        config["sender_language_model"]["estimator"] = bad
        with pytest.raises(parse_config.InvalidConfig, match="estimator"):
            parse_config.validate_config(config)

    for good in ("gumbel", "identity"):
        config = get_config()
        config["sender_language_model"]["estimator"] = good
        parse_config.validate_config(config)


if __name__ == "__main__":
    import itertools

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        marks = getattr(fn, "pytestmark", [])
        builds = list(
            itertools.chain.from_iterable(
                m.args[1] for m in marks if m.name == "parametrize"
            )
        )
        for arguments in ([(b,) for b in builds] or [()]):
            fn(*arguments)
            passed += 1
        print(f"ok  {fn.__name__}")
    print(f"\n{passed} tests passed")

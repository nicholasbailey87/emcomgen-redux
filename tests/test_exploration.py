"""
Tests for the Gumbel exploration channel in code/models/sender.py.

Runnable without pytest:  python tests/test_exploration.py

`F.gumbel_softmax(logits, hard=True)` emits `argmax(logits + g)` with
`g ~ Gumbel(0,1)`, whose standard deviation is a fixed 1.283. That noise floor
does not move, so how many symbols survive the channel is set entirely by the
scale of the speaker's logits -- which, before any of this, was an accident of
the architecture: one arm of a ladder passed 99% of its symbols and another 24%,
with nothing in the config saying so.

Two mechanisms fix that between them, and the tests are organised around which
does what.

`layer_norm_logits` pins the emittable logits to unit variance per example and
per position, so every speaker arrives at the channel with logits of the same
magnitude whatever its architecture did. **That** is what makes exploration
comparable across arms, and it is checked directly: two logit tensors differing
by a factor of a hundred must reach the same rate.

`logit_scale` then says what that unit is worth, as the constant `c * ln(V)`.
The vocabulary term is not cosmetic -- a winner must beat the largest of `V`
Gumbel draws and `E[max g] = ln V + gamma`, so without it the same coefficient
would mean a different channel for ShapeWorld's V=14 than for CUB's V=20. It is
a constant rather than a per-batch solve against a requested rate, because
solving on top of LayerNorm also pins the speaker's *shape*: it overwrites the
speaker's own confidence in both directions, hardest at initialisation, where
forcing high fidelity onto a speaker whose argmax is nearly input-independent
means emitting one message for every input from the first batch.

So the contract is a starting point and a range, not a target. What a speaker
actually achieves is `realised_survival`, and it is expected to move over a run.

The rest is unchanged and still has to hold. The Gumbel-max identity, that a
slot's survival probability is exactly its winning token's softmax probability,
is what lets survival be measured off a softmax with no Monte Carlo. The uniform
mixture's bounds have to survive the ordering -- the scale multiplies *before*
mixing, and the reverse destroys them. Eval has to be greedy, deterministic and
invariant to every train-time knob. And the normaliser has to be
position-invariant, which BatchNorm was not.
"""

import math
import os
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import models.sender as S  # noqa: E402
from parse_config import get_config  # noqa: E402


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
    settings = dict(get_config()["sender_language_model"])
    settings.update(overrides)
    return settings


def _gru_speaker(**overrides):
    settings = _language_model_config(d_model=32, token_embedding_size=32, **overrides)
    return S.SenderGRULM(16, **settings)


def _transformer_speaker(**overrides):
    # d_model 64 over 4 heads gives head_dim 16, which is broccoli's minimum for
    # a 1-D rotary embedding; the DEFAULT.toml width would work but is slow.
    settings = _language_model_config(
        d_model=64, token_embedding_size=64, heads=4, layers=1, **overrides
    )
    return S.SenderTransformerLM(64, **settings)


def _prototypes(speaker, batch_size=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    size = speaker.referent_embedding_size
    return (
        torch.randn(batch_size, size, generator=generator),
        torch.randn(batch_size, size, generator=generator),
    )


# ------------------------------------------------- 1. the scale is V-aware --

def test_logit_scale_resolves_against_the_vocabulary():
    """
    `c * ln(V)`, and the two values the shipped datasets actually use. Pinned
    numerically because these are quoted in DEFAULT.toml and the README.
    """
    assert S.logit_scale(0.66, 14) == pytest.approx(1.742, abs=0.001)
    assert S.logit_scale(0.66, 20) == pytest.approx(1.977, abs=0.001)

    # Monotone in both arguments, and a bare coefficient is recovered at V = e.
    assert S.logit_scale(0.66, 20) > S.logit_scale(0.66, 14)
    assert S.logit_scale(1.0, math.e) == pytest.approx(1.0)


@pytest.mark.parametrize("build", [_gru_speaker, _transformer_speaker])
def test_speaker_resolves_its_scale_from_its_own_vocabulary(build):
    """
    The scale is a plain float resolved at construction, not a buffer and not
    something a training step can move.
    """
    speaker = build()
    assert speaker.logit_scale == pytest.approx(
        S.logit_scale(
            get_config()["sender_language_model"]["logit_scale_coefficient"],
            speaker.vocabulary,
        )
    )
    assert isinstance(speaker.logit_scale, float)

    birds = build(vocabulary=20)
    assert birds.logit_scale > speaker.logit_scale


def test_initial_operating_point_is_vocabulary_invariant():
    """
    The point of the `ln(V)` term. A freshly initialised speaker -- whose
    normalised logits are near enough i.i.d. normal -- must flip roughly the
    same fraction of its symbols whatever its vocabulary size, because the
    scale grows with the noise floor it has to clear.

    Without the term this fails loudly: at a flat scale of 2.0 the same speaker
    flips 50% of its symbols at V=14 and 76% at V=128, which would be a
    difference in the channel masquerading as a difference between datasets.
    """
    coefficient = 0.66
    rates = {}
    for vocabulary in (8, 14, 20, 32, 64):
        generator = torch.Generator().manual_seed(0)
        raw = torch.randn(4096, vocabulary + 4, generator=generator)
        logits = _masked(raw)
        survival = S.mean_winning_probability(
            logits, S.logit_scale(coefficient, vocabulary), 0.0
        ).item()
        rates[vocabulary] = 1.0 - survival

    # Tight where it matters: the two vocabularies the shipped datasets use.
    assert abs(rates[14] - rates[20]) < 0.02, rates

    # Looser across a 8x range of vocabulary, where the log fit is approximate.
    assert max(rates.values()) - min(rates.values()) < 0.06, (
        f"initial rate varies with V: {rates}"
    )

    # And the level is the one DEFAULT.toml documents: a speaker that starts
    # genuinely uncertain, with room to earn confidence.
    for vocabulary, rate in rates.items():
        assert 0.40 < rate < 0.62, f"V={vocabulary} starts at {rate:.3f}"

    # The control: drop the vocabulary term and the invariance goes away.
    flat = {}
    for vocabulary in (14, 64):
        generator = torch.Generator().manual_seed(0)
        logits = _masked(torch.randn(4096, vocabulary + 4, generator=generator))
        flat[vocabulary] = 1.0 - S.mean_winning_probability(logits, 2.0, 0.0).item()
    assert flat[64] - flat[14] > 0.1, flat


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
    scale = S.logit_scale(0.66, 14)

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


def test_scale_invariance_stops_at_the_layer_norm_epsilon():
    """
    Where the claim above runs out, pinned so it is a known limit rather than a
    surprise. `F.layer_norm` divides by `sqrt(var + eps)` with `eps = 1e-5`, so
    a speaker whose logits collapse towards that variance is normalised by
    something increasingly unlike its own spread, and `logit_scale` -- a
    constant -- cannot absorb it the way the per-batch solve could.

    Three orders of magnitude clear of the real ladder, whose logits ran from a
    standard deviation of 1 to 159. It is pinned because a future change to the
    normaliser (or to `eps`) should move this deliberately, not silently.
    """
    raw = _logit_shapes(14)["typical"]
    scale = S.logit_scale(0.66, 14)

    def survival(logits):
        return S.mean_winning_probability(_masked(logits), scale, 0.02).item()

    baseline = survival(raw)  # incoming sd ~1.0

    assert abs(survival(raw * 0.1) - baseline) < 1e-3      # sd 0.1: fine
    assert abs(survival(raw * 0.01) - baseline) > 5e-3     # sd 0.01: visible
    assert abs(survival(raw * 0.001) - baseline) > 0.1     # sd 0.001: gone

    # And it only bites downwards: there is no upper limit.
    assert abs(survival(raw * 1e6) - baseline) < 3e-4


def test_shape_still_moves_the_channel():
    """
    The other half of the same point: LayerNorm removes *scale* and only scale.
    A speaker that concentrates its mass gets a cleaner channel at the same
    setting, and that is the finding `realised_survival` exists to report --
    the thing the calibration used to erase.
    """
    scale = S.logit_scale(0.66, 14)
    shapes = _logit_shapes(14)

    typical = S.mean_winning_probability(_masked(shapes["typical"]), scale, 0.02)
    peaked = S.mean_winning_probability(_masked(shapes["peaked"]), scale, 0.02)

    assert peaked.item() > typical.item() + 0.1


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


# --------------------------------------------------- 5. eval is the policy --

def _decode_message(speaker, prototypes):
    return speaker.decode(prototypes)[0].argmax(-1)


@pytest.mark.parametrize("build", [_gru_speaker, _transformer_speaker])
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
    if isinstance(speaker, S.SenderTransformerLM):
        logits = speaker.outputs2vocab(speaker.embeddings(prototypes))
        logits = S.layer_norm_logits(logits, speaker.vocabulary)
        return S.mask_reserved_tokens(logits).argmax(-1)

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


@pytest.mark.parametrize("build", [_gru_speaker, _transformer_speaker])
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
            original = getattr(speaker, attribute)
            setattr(speaker, attribute, value)
            assert torch.equal(_decode_message(speaker, prototypes), reference), (
                f"eval message changed with {attribute} = {value}"
            )
            setattr(speaker, attribute, original)


# ------------------------------------ 6. survival is measured, not targeted --

@pytest.mark.parametrize("build", [_gru_speaker, _transformer_speaker])
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

    speaker.logit_scale *= 20.0
    speaker.decode(prototypes)
    assert speaker.realised_survival > baseline + 0.1

    # A freshly initialised speaker starts genuinely uncertain, with room to
    # earn confidence rather than beginning at its own ceiling.
    fresh = build().train()
    fresh.decode(_prototypes(fresh))
    assert 0.35 < fresh.realised_survival < 0.70, fresh.realised_survival


@pytest.mark.parametrize("build", [_gru_speaker, _transformer_speaker])
def test_survival_does_not_move_during_eval(build):
    """Eval samples nothing, so it measures nothing."""
    torch.manual_seed(0)
    speaker = build().train()
    prototypes = _prototypes(speaker)

    speaker.decode(prototypes)
    trained = speaker.realised_survival

    speaker.eval()
    with torch.no_grad():
        for _ in range(3):
            speaker.decode(prototypes)

    assert speaker.realised_survival == trained


@pytest.mark.parametrize("build", [_gru_speaker, _transformer_speaker])
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

@pytest.mark.parametrize("build", [_gru_speaker, _transformer_speaker])
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


# ------------------------------------------------------------ 8. the config --

def test_config_rejects_a_missing_or_invalid_coefficient():
    """
    `SafeDict` only warns on a missing key and hands back None, which would fail
    confusingly deep inside the decode, so `parse_config` checks it up front.
    """
    import parse_config

    for bad in (None, 0.0, -1.0):
        config = get_config()
        config["sender_language_model"]["logit_scale_coefficient"] = bad
        with pytest.raises(parse_config.InvalidConfig, match="logit_scale_coefficient"):
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

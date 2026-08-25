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

import pytest
import torch
from torch.nn import functional as F

import _bootstrap  # noqa: F401
from _bootstrap import config_section

import data.language
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

    `logit_scale` is no longer one: it is a read-only property over the learned
    `log_logit_scale` parameter, so it comes back as a grad-requiring tensor
    that `pytest.approx` cannot take a `float()` of. `uniform_weight` and `tau`
    are still ordinary attributes.
    """
    value = getattr(speaker, attribute)
    return value.item() if torch.is_tensor(value) else value


def _set_knob(speaker, attribute, value):
    """
    Set one of the sampling knobs, whatever it is stored as underneath.

    `logit_scale` has no setter -- it is stored as its log so gradient descent
    cannot walk it through zero -- so it is written through `log_logit_scale`
    under `no_grad`, which is what `reset_logit_scale` does.
    """
    if attribute == "logit_scale":
        with torch.no_grad():
            speaker.log_logit_scale.fill_(math.log(value))
    else:
        setattr(speaker, attribute, value)


# ------------------------------------ 1. the scale is solved, not stated --

def test_logit_scale_resolves_the_requested_entropy():
    """
    The solve inverts `initial_energy`, and the two scales the shipped datasets
    actually use. Pinned numerically because these are quoted in DEFAULT.toml
    and the README.
    """
    assert S.logit_scale(0.9, 14, 0.02) == pytest.approx(0.802, abs=0.002)
    assert S.logit_scale(0.9, 20, 0.02) == pytest.approx(0.839, abs=0.002)

    # It is an inverse, so it must round-trip at whatever it returns.
    for energy in (0.5, 0.7, 0.9, 0.99):
        for vocabulary in (8, 14, 20, 64):
            scale = S.logit_scale(energy, vocabulary, 0.02)
            assert S.initial_energy(scale, vocabulary, 0.02) == pytest.approx(
                energy, abs=0.002
            ), (energy, vocabulary)

    # Monotone: asking for more entropy means a smaller scale.
    scales = [S.logit_scale(e, 20, 0.02) for e in (0.5, 0.7, 0.9, 0.99)]
    assert scales == sorted(scales, reverse=True), scales


def test_the_solve_is_deterministic():
    """
    The resolved scale must not depend on global RNG state, or two runs of the
    same config would get different channels and nobody would think to look
    here. `initial_logit_sample` owns its own generator for this reason.
    """
    torch.manual_seed(1234)
    first = S.logit_scale(0.9, 20, 0.02)
    torch.manual_seed(999)
    _ = torch.randn(1000)
    second = S.logit_scale(0.9, 20, 0.02)

    assert first == second


def test_uniform_weight_floors_the_achievable_entropy():
    """
    `uniform_weight` owns the sharp end: mixing caps a slot's winner at
    `1 - w + w/V`, so there is entropy no scale can remove. Asking for less than
    that is a config error the speaker cannot honour, and it warns rather than
    silently starting somewhere else.
    """
    floor = S.initial_energy(S.ENERGY_SCALE_MAX, 20, 0.02)
    assert 0.04 < floor < 0.06, floor

    with pytest.warns(UserWarning, match="below the floor"):
        S.logit_scale(floor / 2, 20, 0.02)

    # And the two knobs barely interact at the flat end, which is why the
    # default can be reasoned about without reference to `uniform_weight`.
    assert S.initial_energy(0.84, 20, 0.02) == pytest.approx(
        S.initial_energy(0.84, 20, 0.0), abs=0.01
    )


@pytest.mark.parametrize(
    "build", [_gru_speaker, _transformer_speaker, _transformer_latent_speaker]
)
def test_speaker_resolves_its_scale_from_its_own_vocabulary(build):
    """
    The scale is solved at construction from the speaker's *own* vocabulary and
    uniform weight, not stated in the config and not shared between speakers.

    It is no longer a plain float: it is learned, stored as `log_logit_scale`
    and read back through the `logit_scale` property, so what is pinned here is
    where it *starts*. `initial_logit_scale` keeps that starting value for
    `reset_logit_scale` and for the tau coupling to measure against, so the two
    must agree before any training step has run.
    """
    speaker = build()
    solved = S.logit_scale(
        get_config()["sender_language_model"]["init_energy"],
        speaker.vocabulary,
        speaker.uniform_weight,
    )
    assert speaker.logit_scale.item() == pytest.approx(solved)
    assert speaker.initial_logit_scale == pytest.approx(solved)

    # It is learned, so it has to be a parameter of the speaker and it has to
    # be the log that carries the gradient -- exponentiating at the read is
    # what keeps the scale positive under any step the optimiser takes.
    assert isinstance(speaker.log_logit_scale, torch.nn.Parameter)
    assert speaker.log_logit_scale.requires_grad
    assert speaker.logit_scale.requires_grad
    assert isinstance(speaker.initial_logit_scale, float)

    # A bigger vocabulary needs a bigger scale to hold the same entropy.
    birds = build(vocabulary=20)
    assert birds.logit_scale.item() > speaker.logit_scale.item()


def test_the_documented_reference_points_still_hold():
    """
    `logit_scale`'s "Rederiving the default" section is a table, and numbers in
    comments rot. This recomputes it. The default of 0.9 is set from where a
    measured birds run went, so the mapping from entropy to what an observer
    actually sees in metrics.csv -- `realised_survival` -- is the thing that
    must not drift.
    """
    V = 20                                          # birds
    expected = {                                    # entropy -> argmax prob
        0.94: 0.143,
        0.90: 0.185,
        0.85: 0.234,
        0.77: 0.310,
        0.62: 0.445,
        0.57: 0.489,
    }

    for energy, argmax_probability in expected.items():
        scale = S.logit_scale(energy, V, 0.02)
        generator = torch.Generator().manual_seed(0)
        logits = _masked(torch.randn(100000, V + 4, generator=generator))
        survival = S.mean_winning_probability(logits, scale, 0.02).item()
        assert survival == pytest.approx(argmax_probability, abs=0.015), (
            f"{energy} retained should show as survival ~{argmax_probability}, "
            f"got {survival:.3f}"
        )

    # The default sits between where the measured run started and the extreme
    # it flattened itself to. Both bounds are load-bearing: below the first is
    # the premature-sharpening failure the scheme exists to avoid, above the
    # second the messages may carry too little for the listener to learn from.
    assert 0.62 < 0.9 < 0.94


def test_initial_entropy_is_vocabulary_invariant():
    """
    The point of solving rather than stating a scale. Two speakers with
    different vocabularies must start equally uncommitted, or a difference in
    the channel would masquerade as a difference between datasets.

    The control is the `c * ln(V)` form this replaced: it was derived to hold a
    *survival rate* constant, and over-corrects badly for entropy.
    """
    energy = get_config()["sender_language_model"]["init_energy"]

    realised = {}
    for vocabulary in (8, 14, 20, 32, 64):
        scale = S.logit_scale(energy, vocabulary, 0.02)
        realised[vocabulary] = S.initial_energy(scale, vocabulary, 0.02)

    for vocabulary, value in realised.items():
        assert value == pytest.approx(energy, abs=0.002), (
            f"V={vocabulary} starts at {value:.4f}, asked for {energy}"
        )

    # The control: a fixed `c * ln(V)` drifts across the same range, and by
    # much more than the solve's residual.
    fitted = {
        vocabulary: S.initial_energy(
            0.28 * math.log(vocabulary), vocabulary, 0.02
        )
        for vocabulary in (8, 64)
    }
    assert abs(fitted[64] - fitted[8]) > 0.05, fitted


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
    scale = S.logit_scale(0.9, 14, 0.02)

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
    scale = S.logit_scale(0.9, 14, 0.02)

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
    scale = S.logit_scale(0.9, 14, 0.02)
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

    _set_knob(speaker, "logit_scale", speaker.logit_scale.item() * 20.0)
    speaker.decode(prototypes)
    assert speaker.realised_survival > baseline + 0.1

    # A freshly initialised speaker starts genuinely uncertain, with room to
    # earn confidence rather than beginning at its own ceiling. At the default
    # `init_energy` of 0.9 that is an argmax it holds about a fifth of the time
    # -- see the reference table in `logit_scale`.
    fresh = build().train()
    fresh.decode(_prototypes(fresh))
    assert 0.12 < fresh.realised_survival < 0.32, fresh.realised_survival


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


# ------------------------------------------------------------ 8. the config --

def test_config_rejects_a_missing_or_invalid_init_energy():
    """
    `SafeDict` only warns on a missing key and hands back None, which would fail
    confusingly deep inside the decode, so `parse_config` checks it up front.

    The upper bound matters as much as the lower one: `init_energy` is a
    fraction of maximum entropy, so anyone who reads it as a percentage and
    writes `90` must be told, not quietly given a scale of 0.001.
    """
    import parse_config

    for bad in (None, 0.0, -1.0, 1.5, 90):
        config = get_config()
        config["sender_language_model"]["init_energy"] = bad
        with pytest.raises(parse_config.InvalidConfig, match="init_energy"):
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

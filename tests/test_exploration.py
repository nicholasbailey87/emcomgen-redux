"""
Tests for the calibrated Gumbel exploration channel in code/models/sender.py.

Runnable without pytest:  python tests/test_exploration.py

`F.gumbel_softmax(logits, hard=True)` emits `argmax(logits + g)` with
`g ~ Gumbel(0,1)`, whose standard deviation is a fixed 1.283. That noise floor
does not move, so how many symbols survive the channel is set entirely by the
scale of the speaker's logits -- which, before this, was an accident of the
architecture: one arm of a ladder passed 99% of its symbols and another 24%,
with nothing in the config saying so. The point of the machinery under test is
to make that a stated number, identical across speakers.

Six things have to hold for it to be that.

The calibration has to *land*, across logit distributions of wildly different
scale and sharpness -- scale-invariance is the entire claim, so a sharp and a
flat distribution are checked at every setting.

It rests on the Gumbel-max identity, that a slot's survival probability is
exactly its winning token's softmax probability. That is what makes the solve
exact and free of Monte Carlo, so it is pinned against actual sampling here.

The uniform mixture's bounds have to survive the ordering. Gain multiplies
*before* mixing; the reverse order destroys them, which is checked so that a
future reordering regresses loudly rather than silently.

Eval has to be greedy and deterministic, and invariant to every train-time knob.
Both speakers used to call `gumbel_softmax` unconditionally, so every reported
accuracy and topsim was measured through the noisy channel rather than on the
policy.

The gain is a buffer, not a parameter, so it has to survive a checkpoint
round-trip and stay put during eval.

And the normaliser has to be position-invariant. BatchNorm was not: it
annihilated per-position offsets in the GRU, which sees one position per call,
while preserving them in the Transformer, which sees them all at once. That
asymmetry is exactly what LayerNorm removes.
"""

import os
import sys
import warnings

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
    pinned near 1.0 and passing 33%. If the calibration is scale-invariant it
    has to bring both to the same rate.
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


# -------------------------------------------------------------- 1. round-trip

def test_calibration_round_trip():
    """
    The solved gain delivers the requested rate, whatever the logits looked
    like going in. This is the whole contract: `token_exploration_rate` is a
    number the config states rather than a number the architecture happens to
    produce.
    """
    for rate in (0.05, 0.1, 0.2):
        for uniform_weight in (0.0, 0.02):
            for vocabulary in (14, 20):
                for name, raw in _logit_shapes(vocabulary).items():
                    logits = _masked(raw)
                    gain = S.calibrate_exploration_gain(
                        logits, rate, uniform_weight
                    )
                    realised = S.mean_winning_probability(
                        logits, gain.item(), uniform_weight
                    ).item()
                    assert abs(realised - (1.0 - rate)) < 0.005, (
                        f"{name} logits, V={vocabulary}, w={uniform_weight}, "
                        f"rate={rate}: realised {realised:.4f} at gain "
                        f"{gain.item():.3f}"
                    )


def test_calibration_is_scale_invariant():
    """
    Two speakers whose logits differ only by a constant factor -- the actual
    difference between the arms of the ladder -- end up at the same rate, by
    landing on gains that differ by the reciprocal of that factor.
    """
    logits = _masked(_logit_shapes(14)["typical"])

    gain_one = S.calibrate_exploration_gain(logits, 0.1, 0.02).item()
    gain_hundred = S.calibrate_exploration_gain(logits * 100.0, 0.1, 0.02).item()

    assert abs(gain_one / gain_hundred - 100.0) / 100.0 < 0.02


# ---------------------------------------------------------- 2. Gumbel-max id

def test_gumbel_max_identity():
    """
    The assumption the calibration rests on: the probability that the noise
    leaves a slot's argmax alone is exactly the winning token's softmax
    probability. If this were only approximate, the solve would need a Monte
    Carlo over noise draws and a seed to be reproducible.
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

    # And in aggregate, which is the quantity the calibration actually targets.
    assert abs(empirical.mean().item() - expected.mean().item()) < 0.005


# ---------------------------------------------------------------- 3. bounds

def test_mixing_bounds_hold_when_gain_comes_first():
    """
    Mixing caps a slot's winner at `1 - w + w/V` and floors its losers at `w/V`.
    Those bounds are the permanent exploration floor -- 1.86% of symbols flipped
    at w = 0.02, V = 14, which training cannot reduce -- so they have to hold
    however large the gain gets.
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


def test_mixing_bounds_break_when_gain_comes_second():
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


# --------------------------------------------------------- 4. floor warning

def test_rate_below_floor_warns():
    """
    At w = 0.02 and V = 14 the mixture alone flips 1.86% of symbols, so a
    request for 1% is unreachable and would otherwise be silently missed.
    """
    with pytest.warns(UserWarning, match="below"):
        S.check_exploration_rate_floor(0.01, 0.02, 14)

    assert abs(S.exploration_rate_floor(0.02, 14) - 0.0186) < 1e-4
    assert abs(S.exploration_rate_floor(0.02, 20) - 0.019) < 1e-4


def test_rate_above_floor_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        S.check_exploration_rate_floor(0.1, 0.02, 14)


def test_speaker_construction_warns_below_floor():
    with pytest.warns(UserWarning, match="below"):
        _gru_speaker(token_exploration_rate=0.01, uniform_weight=0.02)


# --------------------------------------------------- 5/6. eval is the policy

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
        if speaker.layer_norm_logits:
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
        if speaker.layer_norm_logits:
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
        ):
            original = getattr(speaker, attribute)
            setattr(speaker, attribute, value)
            assert torch.equal(_decode_message(speaker, prototypes), reference), (
                f"eval message changed with {attribute} = {value}"
            )
            setattr(speaker, attribute, original)

        speaker.exploration_gain.fill_(137.0)
        assert torch.equal(_decode_message(speaker, prototypes), reference)


# ------------------------------------------------------------ 7. buffer state

@pytest.mark.parametrize("build", [_gru_speaker, _transformer_speaker])
def test_gain_buffer_survives_a_state_dict_round_trip(build):
    torch.manual_seed(0)
    speaker = build()
    prototypes = _prototypes(speaker)

    speaker.train()
    for _ in range(3):
        speaker.decode(prototypes)

    assert speaker.exploration_gain_updates.item() == 3
    # The first update sets the buffer outright, so batch one is not sampled at
    # a gain of 1.0.
    assert speaker.exploration_gain.item() != 1.0
    assert 1e-2 <= speaker.exploration_gain.item() <= 1e4

    trained_gain = speaker.exploration_gain.item()

    restored = build()
    assert restored.exploration_gain.item() == 1.0
    restored.load_state_dict(speaker.state_dict())
    assert restored.exploration_gain.item() == trained_gain
    assert restored.exploration_gain_updates.item() == 3

    # No affine LayerNorm parameters and no BatchNorm running statistics reached
    # the checkpoint.
    assert not any("batch_norm" in key for key in speaker.state_dict())


@pytest.mark.parametrize("build", [_gru_speaker, _transformer_speaker])
def test_gain_does_not_move_during_eval(build):
    torch.manual_seed(0)
    speaker = build()
    prototypes = _prototypes(speaker)

    speaker.train()
    speaker.decode(prototypes)
    gain = speaker.exploration_gain.item()
    updates = speaker.exploration_gain_updates.item()

    speaker.eval()
    with torch.no_grad():
        for _ in range(3):
            speaker.decode(prototypes)

    assert speaker.exploration_gain.item() == gain
    assert speaker.exploration_gain_updates.item() == updates


@pytest.mark.parametrize("build", [_gru_speaker, _transformer_speaker])
def test_training_converges_on_the_requested_rate(build):
    """
    End to end, through the speaker rather than the helper: after enough batches
    for the EMA to settle, the channel a speaker is actually sampling through
    passes `1 - token_exploration_rate` of its symbols.
    """
    torch.manual_seed(0)
    speaker = build(token_exploration_rate=0.1).train()

    realised = []
    for step in range(60):
        speaker.decode(_prototypes(speaker, seed=step))
        realised.append(speaker.realised_survival)

    # `realised_survival` is a single-batch statistic and jitters by a couple of
    # points; the log-space EMA is what smooths the gain itself, so the settled
    # rate is read over the tail rather than off one batch.
    settled = sum(realised[-10:]) / 10
    assert abs(settled - 0.9) < 0.02, f"settled at {settled:.4f}"
    assert 1e-2 < speaker.exploration_gain.item() < 1e4


# ------------------------------------------------- 8. position invariance --

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

    # Every slot arrives at the same magnitude, which is what the gain is
    # calibrated against.
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

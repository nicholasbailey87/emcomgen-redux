"""
Tests for the speaker's optional logit BatchNorm,
`[sender_language_model] batch_norm_logits`.

Runnable without pytest:  python tests/test_logit_batch_norm.py

An affine-free `BatchNorm1d` over the emittable logits, ahead of the layer norm
in `GumbelChannel.sample_symbols`: batch statistics in training, running
statistics at eval, one mean and variance per token shared across positions,
and the four reserved columns untouched.
"""

import pytest
import torch
import torch.nn as nn

import _bootstrap  # noqa: F401
from _bootstrap import config_section

import models.sender as S


VOCABULARY = 14


def _settings(**overrides):
    return config_section(
        "sender_language_model",
        d_model=32,
        token_embedding_size=32,
        heads=2,
        vocabulary=VOCABULARY,
        **overrides,
    )


def _gru(**overrides):
    return S.SenderGRULM(16, **_settings(**overrides))


def _transformer(**overrides):
    return S.SenderTransformerLM(32, **_settings(**overrides))


def _logits(*shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    # A per-token offset every example shares, which is what the BatchNorm is
    #     there to remove.
    offset = 5.0 * torch.arange(VOCABULARY + 4, dtype=torch.float32)
    return torch.randn(*shape, VOCABULARY + 4, generator=generator) + offset


def test_on_by_default():
    """
    `DEFAULT.toml` turned it on on 2026-10-26; it was off when added.
    """
    assert config_section("sender_language_model")["batch_norm_logits"] is True
    assert _gru().batch_norms_logits


@pytest.mark.parametrize("build", [_gru, _transformer])
def test_off_it_adds_nothing(build):
    speaker = build(batch_norm_logits=False)

    assert not speaker.batch_norms_logits
    assert not hasattr(speaker, "logit_batch_norm")
    assert not any(isinstance(m, nn.BatchNorm1d) for m in speaker.modules())

    logits = _logits(8)
    assert torch.equal(speaker.batch_norm_logits(logits), logits)


@pytest.mark.parametrize("build", [_gru, _transformer])
def test_on_it_has_no_parameters_and_reaches_both_speakers(build):
    speaker = build(batch_norm_logits=True)

    assert isinstance(speaker.logit_batch_norm, nn.BatchNorm1d)
    assert speaker.logit_batch_norm.num_features == VOCABULARY
    assert list(speaker.logit_batch_norm.parameters()) == []


@pytest.mark.parametrize("shape", [(64,), (16, 5)])
def test_training_centres_each_token_over_the_batch(shape):
    """
    One step `(batch, V + 4)` and a whole message `(batch, length, V + 4)`
    both normalise each emittable token over every example and position.
    """
    speaker = _gru(batch_norm_logits=True).train()
    logits = _logits(*shape)

    out = speaker.batch_norm_logits(logits)

    assert torch.equal(out[..., :4], logits[..., :4])

    emittable = out[..., 4:].reshape(-1, VOCABULARY)
    assert emittable.mean(0).abs().max() < 1e-5
    assert (emittable.var(0, unbiased=False) - 1).abs().max() < 1e-3

    # The shared offset is what the running mean picked up.
    running_mean = speaker.logit_batch_norm.running_mean
    assert running_mean[-1] > running_mean[0]


def test_eval_uses_the_running_statistics():
    speaker = _gru(batch_norm_logits=True).train()
    for seed in range(5):
        speaker.batch_norm_logits(_logits(64, seed=seed))

    speaker.eval()
    batch_norm = speaker.logit_batch_norm
    logits = _logits(3, seed=99)

    expected = (logits[:, 4:] - batch_norm.running_mean) / torch.sqrt(
        batch_norm.running_var + batch_norm.eps
    )
    out = speaker.batch_norm_logits(logits)

    assert torch.allclose(out[:, 4:], expected, atol=1e-6)
    assert torch.equal(out[:, :4], logits[:, :4])


def test_sample_symbols_applies_it_before_the_layer_norm():
    """
    At eval the emitted symbol is the argmax of the masked logits. A shared
    offset favouring the last token wins every example without the BatchNorm;
    with it, the running mean takes the offset out.
    """
    plain = _gru(batch_norm_logits=False).eval()
    normed = _gru(batch_norm_logits=True).train()
    for seed in range(20):
        normed.batch_norm_logits(_logits(64, seed=seed))
    normed.eval()

    logits = _logits(64, seed=99)
    winners_plain = plain.sample_symbols(logits)[0].argmax(-1)
    winners_normed = normed.sample_symbols(logits)[0].argmax(-1)

    assert winners_plain.unique().tolist() == [VOCABULARY + 3]
    assert len(winners_normed.unique()) > 1


@pytest.mark.parametrize("build", [_gru, _transformer])
def test_reset_clears_the_running_statistics(build):
    speaker = build(batch_norm_logits=True).train()
    speaker.batch_norm_logits(_logits(64))
    assert speaker.logit_batch_norm.running_mean.abs().max() > 0

    speaker.reset_parameters()

    batch_norm = speaker.logit_batch_norm
    assert (batch_norm.running_mean == 0).all()
    assert (batch_norm.running_var == 1).all()
    assert int(batch_norm.num_batches_tracked) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

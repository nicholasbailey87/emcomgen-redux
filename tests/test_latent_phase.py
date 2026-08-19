"""
`SenderTransformerLM`'s latent phase, its polarity tag, and its two arms.

A learned query array cross-attends into the two prototypes to build a latent
array, and what happens next depends on `bidirectional`. The parallel arm
(`true`) runs a self-attention stack over that array and a second learned query
reads it back down to the message's length. The decoder arm (`false`, the
default) treats it as a memory and generates the message one symbol at a time.
Everything up to and including the latent array is shared, which is why the
polarity tests below run against `encode` rather than against either arm's
output.

Three things follow that are worth pinning down, because all three are invisible
in a training curve.

**The latent length must not leak downstream.** It is the whole point of the
decoder cross-attention that `latent_message_multiplier` can move without
`message_length` moving, and a regression there would silently change what the
listener is scored on.

**The polarity tag must break the prototype symmetry.** Without it the encoder
cross-attention is a weighted *sum* over two keys carrying no positional or type
encoding, so its output is bit-identical under swapping the positive and negative
prototype -- the speaker cannot tell "the concept is X" from "the concept is
not-X". `SenderGRULM` never had the problem, because `init_h` reads
`torch.cat(prototypes, 1)` and each polarity lands in its own weight columns.

**The decoder arm must actually be autoregressive.** Its whole justification is
that symbol `i` is conditioned on symbols `< i`, and nothing in a loss curve
would reveal a causal mask that had stopped biting -- the model would simply be
the parallel arm wearing the wrong config. `decode` builds the sequence in a
fixed-length buffer with the tail held at zero, which is exact only if the mask
holds, so the test that checks the mask is also the test that checks the buffer.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import models.sender as S  # noqa: E402
from parse_config import get_config  # noqa: E402

D_MODEL = 320
HEADS = 5  # head_dim 64, as everywhere


def _settings(**overrides):
    settings = dict(get_config()["sender_language_model"])
    settings.update(
        d_model=D_MODEL, token_embedding_size=D_MODEL, heads=HEADS, layers=2
    )
    settings.update(overrides)
    return settings


def _speaker(**overrides):
    """The decoder arm, as `DEFAULT.toml` selects it."""
    return S.SenderTransformerLM(D_MODEL, **_settings(**overrides))


def _latent_speaker(**overrides):
    overrides.setdefault("bidirectional", True)
    return S.SenderTransformerLM(D_MODEL, **_settings(**overrides))


def _prototypes(batch=3):
    return (torch.randn(batch, D_MODEL), torch.randn(batch, D_MODEL))


# ------------------------------------------------------- the latent length --

@pytest.mark.parametrize(
    "message_length,multiplier,expected",
    [
        # ShapeWorld's 7 (5 content symbols) and CUB's 10 (8).
        (7, 1.0, 5),
        (7, 2.0, 10),
        (10, 1.0, 8),
        (10, 2.0, 16),
        # Rounded, not floored: 5 * 1.5 is 7.5.
        (7, 1.5, 8),
        (10, 1.5, 12),
    ],
)
def test_latent_length_tracks_the_multiplier(message_length, multiplier, expected):
    speaker = _speaker(
        message_length=message_length, latent_message_multiplier=multiplier
    )

    assert speaker.content_length == message_length - 2
    assert speaker.latent_length == expected
    assert speaker.query.shape == (expected, D_MODEL)


def test_a_multiplier_that_rounds_to_nothing_is_rejected():
    """A silent zero-length latent array would be a very confusing forward."""
    with pytest.raises(ValueError, match="latent_message_multiplier"):
        _speaker(message_length=7, latent_message_multiplier=0.01)


@pytest.mark.parametrize("multiplier", [1.0, 1.5, 2.0, 3.0])
def test_the_message_length_is_free_of_the_latent_length(multiplier):
    """
    The decoder query fixes the output, so the multiplier is invisible from
    outside the speaker. This is what lets it be swept without changing the game.
    """
    speaker = _speaker(latent_message_multiplier=multiplier).eval()

    with torch.no_grad():
        onehot, embeddings = speaker.decode(_prototypes())

    assert onehot.shape == (3, speaker.message_length, speaker.vocabulary + 4)
    assert embeddings.shape == (3, speaker.content_length, D_MODEL)


def test_the_decoder_is_built_even_at_a_multiplier_of_one():
    """
    1.0 must not quietly restore the pre-latent architecture. The knob's job is
    to vary the latent width and nothing else -- if it also removed a module then
    a sweep over it would confound two changes, and `state_dict` shapes would
    move with it, so checkpoints could not be compared across sweep points.
    """
    one = _latent_speaker(latent_message_multiplier=1.0)
    two = _latent_speaker(latent_message_multiplier=2.0)

    assert one.decode_attention is not None
    assert set(dict(one.named_parameters())) == set(dict(two.named_parameters()))


def test_the_gru_speaker_ignores_the_multiplier():
    """It has no latent array; the key reaches it through the same config splat."""
    settings = dict(get_config()["sender_language_model"])
    settings.update(latent_message_multiplier=4.0)

    speaker = S.SenderGRULM(512, **settings)

    assert not hasattr(speaker, "latent_length")


# --------------------------------------------------------- the polarity tag --

def test_the_tag_opens_at_zero():
    """
    So the rung starts at the untagged speaker's behaviour exactly and departs
    only where the loss pays for it, as `AttentionPrototyper`'s scoring weights
    do.
    """
    speaker = _speaker()

    assert torch.equal(
        speaker.polarity_embedding, torch.zeros(2, D_MODEL)
    )
    assert speaker.polarity_separation != speaker.polarity_separation  # NaN


def test_without_the_tag_the_prototypes_are_interchangeable():
    """
    The bug the tag exists for, stated as the property that used to hold. At
    zero the encoder cross-attention is still a plain weighted sum over two
    unmarked keys, so swapping them changes nothing.

    Against `encode` rather than against a whole forward pass, because that is
    where the tag acts and where the symmetry lives. It also makes the test arm
    -independent: both arms read the prototypes through this one module and
    neither can recover a distinction the latent array does not carry.
    """
    speaker = _speaker().eval()
    positive, negative = _prototypes()

    with torch.no_grad():
        forwards = speaker.encode((positive, negative))
        backwards = speaker.encode((negative, positive))

    assert torch.allclose(forwards, backwards, atol=1e-5)


def test_a_learned_tag_tells_the_prototypes_apart():
    speaker = _speaker().eval()
    with torch.no_grad():
        speaker.polarity_embedding.normal_(std=0.5)

    positive, negative = _prototypes()

    with torch.no_grad():
        forwards = speaker.encode((positive, negative))
        backwards = speaker.encode((negative, positive))

    assert not torch.allclose(forwards, backwards, atol=1e-5)


def test_the_two_tag_rows_receive_different_gradients():
    """
    Zero-init is safe here only if the rows separate. They do because the
    gradient at each row is the gradient of the sequence position it was added
    to, and the two prototypes differ in content -- there is no symmetry between
    the rows to break, unlike a zero-initialised hidden layer.
    """
    speaker = _speaker()

    speaker.encode(_prototypes()).pow(2).sum().backward()
    gradient = speaker.polarity_embedding.grad

    assert gradient is not None
    assert (gradient != 0).any()
    assert not torch.allclose(gradient[0], gradient[1])


def test_the_separation_diagnostic_reports_the_gap():
    """
    `norm(e_pos - e_neg)` is the only part of the tag the cross-attention can
    act on: a constant added to both rows shifts every key and value alike and
    cannot separate them. Without the column, "the speaker learned to tell its
    prototypes apart" and "the speaker never used the tag" are the same row in
    metrics.csv.
    """
    speaker = _speaker()
    speaker.train()

    with torch.no_grad():
        speaker.polarity_embedding[0].fill_(3.0)
        speaker.polarity_embedding[1].fill_(0.0)

    speaker.decode(_prototypes())

    expected = (3.0 ** 2 * D_MODEL) ** 0.5
    assert speaker.polarity_separation == pytest.approx(expected, rel=1e-4)


def test_reset_parameters_restores_the_tag_and_the_diagnostic():
    speaker = _speaker()

    with torch.no_grad():
        speaker.polarity_embedding.normal_(std=1.0)
    speaker.polarity_separation = 12.0

    speaker.reset_parameters()

    assert torch.equal(speaker.polarity_embedding, torch.zeros(2, D_MODEL))
    assert speaker.polarity_separation != speaker.polarity_separation  # NaN


# ---------------------------------------------------------------- the arms --

def test_each_arm_builds_only_its_own_modules():
    """
    The two arms are one class, so the thing that stops them blurring is that
    neither carries the other's parameters. A `state_dict` from one must fail
    loudly against the other rather than load with the shapes that happen to
    match.
    """
    decoder = _speaker()
    latent = _latent_speaker()

    assert hasattr(decoder, "decoder")
    assert hasattr(decoder, "token_embedding")
    assert not hasattr(decoder, "transformer")
    assert not hasattr(decoder, "decode_attention")
    assert not hasattr(decoder, "output_query")

    assert hasattr(latent, "transformer")
    assert hasattr(latent, "decode_attention")
    assert hasattr(latent, "output_query")
    assert not hasattr(latent, "token_embedding")

    assert set(dict(decoder.named_parameters())) != set(
        dict(latent.named_parameters())
    )


@pytest.mark.parametrize("build", [_speaker, _latent_speaker])
@pytest.mark.parametrize("multiplier", [1.0, 2.0, 3.0])
def test_both_arms_speak_at_the_configured_length(build, multiplier):
    """
    `latent_message_multiplier` must not leak downstream on either arm. On the
    parallel arm the readout query fixes the length; on the decoder arm the
    sampling loop does. Different mechanisms, same guarantee.
    """
    speaker = build(latent_message_multiplier=multiplier).eval()

    onehot, embeddings = speaker.decode(_prototypes())

    assert onehot.shape == (3, speaker.message_length, speaker.vocabulary + 4)
    assert embeddings.shape == (3, speaker.content_length, D_MODEL)


def test_the_decoder_arm_conditions_on_the_symbols_it_emitted():
    """
    The property the arm exists for, and the one no loss curve would show.

    Recomputed by teacher forcing: feed the message the loop actually emitted
    back through the stack in one pass, and the embeddings must match what the
    loop produced. That holds only if each step really did see the sampled
    prefix and only the prefix -- if the causal mask had stopped biting, the
    loop's steps would each have seen a different zero-padded tail and the two
    would disagree.
    """
    speaker = _speaker().eval()
    prototypes = _prototypes()

    with torch.no_grad():
        onehot, embeddings = speaker.decode(prototypes)

        memory = speaker.latent_layer_norm(speaker.encode(prototypes))

        # SOS, then every emitted symbol but the last, which is never fed back.
        sos = torch.zeros(3, 1, speaker.vocabulary + 4)
        sos[:, 0, 1] = 1.0  # SOS_IDX
        sequence = (
            torch.cat([sos, onehot[:, 1:-2, :]], dim=1)
            @ speaker.token_embedding.weight
        )

        teacher_forced = speaker.decoder(sequence, memory)

    assert torch.allclose(teacher_forced, embeddings, atol=1e-5)


def test_the_parallel_arm_conditions_on_nothing():
    """
    The counterpart claim, and the reason rung 13 answers a different question
    from rung 7: this arm's embeddings are a function of the prototypes alone,
    so they are unchanged by anything about the symbols drawn from them.
    """
    speaker = _latent_speaker().eval()
    prototypes = _prototypes()

    with torch.no_grad():
        first = speaker.decode(prototypes)[1]
        second = speaker.embeddings(prototypes)

    assert torch.equal(first, second)


def test_the_decoder_arm_reports_its_diagnostics_over_every_position():
    """
    `realised_survival` and `logit_spread` are pooled after the loop rather than
    taken from one step, as `SenderGRULM` pools them. A speaker that recorded
    only its last position would look steadily more confident than it is, since
    later positions are the ones with the most context.
    """
    speaker = _speaker()
    speaker.train()

    speaker.decode(_prototypes())

    assert speaker.realised_survival == speaker.realised_survival  # not NaN
    assert speaker.logit_spread == speaker.logit_spread
    assert 0.0 < speaker.realised_survival <= 1.0
    assert speaker.logit_spread > 0.0

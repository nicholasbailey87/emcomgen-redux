"""
`SenderTransformerLM`'s latent phase, its polarity tag, and its two arms.

A learned query array cross-attends into the two prototypes to build a latent
array, and both arms then run the same blocks over it and read the message from
its *tail*. `bidirectional` selects a mask and nothing else: `true` runs the
blocks unmasked and takes the whole tail in one shot, `false` masks them and
takes the tail one slot at a time, overwriting each with the symbol it produced
so the next slot is conditioned on it. Everything up to and including the latent
array is shared, which is why the polarity tests below run against `encode`
rather than against either arm's output.

The arms used to be two architectures -- Perceiver IO with an `output_query`
readout on one side, a cross-attending `TransformerDecoder` over a message-length
sequence on the other. Several tests here were written against that split and
now check the opposite thing: that the arms *share* their stack.

Four things follow that are worth pinning down, because all four are invisible
in a training curve.

**The latent length must not leak downstream.** `latent_message_multiplier` can
move without `message_length` moving -- the message is the last `content_length`
slots however long the array is -- and a regression there would silently change
what the listener is scored on. Note the multiplier now has 1.0 as a hard floor:
an array shorter than the message has nowhere to put it.

**Every message slot must open concept-derived.** This is what the redesign is
for. The old causal arm began its sequence at SOS, one learned vector shared by
every example, so symbol 0's residual stream carried nothing about the concept
and the referents reached it only through cross-attention branches scaled by
DeepNorm's `beta / alpha`. At init that left one seed in five emitting the same
first symbol for every concept. See docs/anecdotes.md.

**The polarity tag must break the prototype symmetry.** Without it the encoder
cross-attention is a weighted *sum* over two keys carrying no positional or type
encoding, so its output is bit-identical under swapping the positive and negative
prototype -- the speaker cannot tell "the concept is X" from "the concept is
not-X". `SenderGRULM` never had the problem, because `init_h` reads
`torch.cat(prototypes, 1)` and each polarity lands in its own weight columns.

**The causal arm must actually be autoregressive.** Its whole justification is
that symbol `i` is conditioned on symbols `< i`, and nothing in a loss curve
would reveal a causal mask that had stopped biting -- the model would simply be
the parallel arm wearing the wrong config. Since the mask is now the *only*
difference between the arms, that is the entire content of `bidirectional`.

**Slots after the one being read must not reach it.** `decode_autoregressively`
leaves the slots it has not sampled yet holding their latent vectors rather than
blanking them, which is free only if the mask holds. The test for that is
therefore also the test for the mask.
"""


import pytest
import torch

import _bootstrap  # noqa: F401
from _bootstrap import config_section

import models.sender as S
from parse_config import get_config

D_MODEL = 320
HEADS = 5  # head_dim 64, as everywhere


def _settings(**overrides):
    return config_section(
        "sender_language_model",
        **{
            "d_model": D_MODEL,
            "token_embedding_size": D_MODEL,
            "heads": HEADS,
            "layers": 2,
            **overrides,
        },
    )


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
    # The message is the tail, so the free slots are whatever is left in front.
    assert speaker.first_message_slot == expected - (message_length - 2)
    assert speaker.first_message_slot >= 0


@pytest.mark.parametrize("multiplier", [0.01, 0.5, 0.9])
def test_a_multiplier_below_one_is_rejected(multiplier):
    """
    1.0 is a hard floor now, where it used to be "at least one position". The
    message is the tail of the latent array, so an array shorter than the message
    has nowhere to put it -- and the failure would otherwise be a negative
    `first_message_slot` silently indexing from the wrong end.
    """
    with pytest.raises(ValueError, match="latent_message_multiplier"):
        _speaker(message_length=7, latent_message_multiplier=multiplier)


@pytest.mark.parametrize("multiplier", [1.0, 1.5, 2.0, 3.0])
def test_the_message_length_is_free_of_the_latent_length(multiplier):
    """
    The message is the last `content_length` slots however long the array is, so
    the multiplier is invisible from outside the speaker. This is what lets it be
    swept without changing the game.
    """
    speaker = _speaker(latent_message_multiplier=multiplier).eval()

    with torch.no_grad():
        onehot, embeddings = speaker.decode(_prototypes())

    assert onehot.shape == (3, speaker.message_length, speaker.vocabulary + 4)
    assert embeddings.shape == (3, speaker.content_length, D_MODEL)


@pytest.mark.parametrize("build", [_speaker, _latent_speaker])
def test_the_multiplier_moves_only_the_free_slots(build):
    """
    The knob's job is to vary how many slots sit ahead of the message, and
    nothing else. At 1.0 there are none and every slot is a message slot; above
    it the extra slots are pure prefix.

    Those free slots are load-bearing rather than spare. Nothing overwrites them,
    so under the causal mask every message slot reads them at every step, which
    is what cross-attending into a memory used to do. `docs/anecdotes.md` has
    what removing them costs.

    `query` is the only parameter whose shape moves with the knob, and it has to:
    it is one learned row per latent slot. That does mean `state_dict` shapes
    move across sweep points, which they did not when the readout was a separate
    fixed-size module -- a sweep over this knob cannot share checkpoints.
    """
    one = build(latent_message_multiplier=1.0)
    two = build(latent_message_multiplier=2.0)

    assert one.first_message_slot == 0
    assert two.first_message_slot == two.content_length
    assert one.content_length == two.content_length

    moved = {
        name
        for name, tensor in one.named_parameters()
        if dict(two.named_parameters())[name].shape != tensor.shape
    }
    assert moved == {"query"}


def test_the_gru_speaker_ignores_the_multiplier():
    """It has no latent array; the key reaches it through the same config splat."""
    settings = dict(get_config()["sender_language_model"])
    settings.update(latent_message_multiplier=4.0)

    speaker = S.SenderGRULM(512, **settings)

    assert not hasattr(speaker, "latent_length")


# --------------------------------------------------------- the polarity tag --

def test_the_tag_opens_antipodal_at_the_prototype_scale():
    """
    One draw, negated for the negative row. Only `e_pos - e_neg` reaches the
    cross-attention, so an antipodal pair buys twice the readable separation per
    unit of tag magnitude that two independent draws would, and there is no
    traverse out of zero before the tag can be read at all.

    The scale is the draw's own: the tag is summed with `referent_layer_norm`'s
    output, which is at per-element unit variance when that norm is reset, so
    `randn_like` already lands at the scale of the thing it is tagging. That
    fixes the opening separation at `2 * sqrt(d_model)` up to sampling noise,
    which is why the tolerances below are loose rather than exact.
    """
    speaker = _speaker()

    assert torch.equal(
        speaker.polarity_embedding[0], -speaker.polarity_embedding[1]
    )
    assert (speaker.polarity_embedding != 0).any()
    assert speaker.polarity_embedding.std().item() == pytest.approx(1.0, rel=0.2)

    separation = (
        speaker.polarity_embedding[0] - speaker.polarity_embedding[1]
    ).norm().item()
    assert separation == pytest.approx(2 * D_MODEL ** 0.5, rel=0.15)

    assert speaker.polarity_separation != speaker.polarity_separation  # NaN


def test_without_the_tag_the_prototypes_are_interchangeable():
    """
    The bug the tag exists for, stated as the property that used to hold. With
    the tag zeroed the encoder cross-attention is a plain weighted sum over two
    unmarked keys, so swapping them changes nothing. Zeroing it by hand rather
    than relying on the init, which now opens at a draw.

    Against `encode` rather than against a whole forward pass, because that is
    where the tag acts and where the symmetry lives. It also makes the test arm
    -independent: both arms read the prototypes through this one module and
    neither can recover a distinction the latent array does not carry.
    """
    speaker = _speaker().eval()
    positive, negative = _prototypes()

    with torch.no_grad():
        speaker.polarity_embedding.zero_()
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
    The rows open antipodal and have to stay free to move apart from there. They
    do because the gradient at each row is the gradient of the sequence position
    it was added to, and the two prototypes differ in content -- nothing ties
    `e_neg` to `-e_pos` after the init, so the pair is a starting point rather
    than a constraint.
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
        speaker.polarity_embedding[1].copy_(speaker.polarity_embedding[0])
    speaker.polarity_separation = 12.0

    speaker.reset_parameters()

    assert torch.equal(
        speaker.polarity_embedding[0], -speaker.polarity_embedding[1]
    )
    assert (speaker.polarity_embedding != 0).any()
    assert speaker.polarity_separation != speaker.polarity_separation  # NaN


# ---------------------------------------------------------------- the arms --

def test_the_mask_is_the_whole_of_what_bidirectional_selects():
    """
    The claim the redesign rests on, and the one that would rot first.

    `bidirectional` used to pick between two architectures; it now picks a mask.
    If that drifted -- a causal flag that stopped reaching the blocks, say --
    nothing in a loss curve would distinguish the causal arm from the parallel
    one wearing the wrong config, because they would be the same model.
    """
    causal = _speaker()
    parallel = _latent_speaker()

    assert {block.attn.causal for block in causal.transformer.blocks} == {True}
    assert {block.attn.causal for block in parallel.transformer.blocks} == {False}


def test_the_arms_differ_by_one_module_and_it_is_the_one_that_reads_symbols_back():
    """
    The two arms are one class and now share their stack, so what stops them
    blurring is no longer that they carry disjoint parameters -- they very nearly
    carry the same ones. The single difference is `token_embedding`, which only
    the causal arm needs because only it feeds a sampled symbol back in.

    This test used to assert the opposite: that neither arm carried the other's
    modules, when one had a `TransformerDecoder` and the other a `transformer`
    plus an `output_query` readout. Recorded here rather than deleted, because
    the inversion is the change.
    """
    causal = _speaker()
    parallel = _latent_speaker()

    assert hasattr(causal, "transformer")
    assert hasattr(parallel, "transformer")

    assert hasattr(causal, "token_embedding")
    assert not hasattr(parallel, "token_embedding")

    for gone in ("decoder", "decode_attention", "output_query"):
        assert not hasattr(causal, gone)
        assert not hasattr(parallel, gone)

    only_causal = set(dict(causal.named_parameters())) - set(
        dict(parallel.named_parameters())
    )
    only_parallel = set(dict(parallel.named_parameters())) - set(
        dict(causal.named_parameters())
    )

    assert only_causal == {"token_embedding.weight"}
    assert only_parallel == set()


def test_every_message_slot_opens_concept_derived():
    """
    The property the redesign exists for, and it is invisible in a loss curve.

    Each message slot holds a vector `encode` built from the prototypes when it
    is read, so the residual stream that produces its symbol *starts* as the
    referent and DeepNorm's `alpha` amplifies it. That is structurally what
    `SenderGRULM.init_h` does. The old causal arm opened at SOS -- one learned
    vector shared by every example -- so symbol 0 began from a constant.

    Two claims, and the first slot is the one that matters: the message slots
    must vary with the prototypes, and they must vary *across the batch*, which a
    constant could not do.
    """
    speaker = _speaker().eval()

    with torch.no_grad():
        one = speaker.latent_layer_norm(speaker.encode(_prototypes()))
        two = speaker.latent_layer_norm(speaker.encode(_prototypes()))

    first = speaker.first_message_slot

    for slot in range(first, speaker.latent_length):
        assert not torch.allclose(one[:, slot, :], two[:, slot, :])
        assert one[:, slot, :].std(0).min() > 0.0


@pytest.mark.parametrize("build", [_speaker, _latent_speaker])
@pytest.mark.parametrize("multiplier", [1.0, 2.0, 3.0])
def test_both_arms_speak_at_the_configured_length(build, multiplier):
    """
    `latent_message_multiplier` must not leak downstream on either arm. Same
    mechanism on both now -- the message is the last `content_length` slots --
    where it used to be a readout query on one arm and the sampling loop on the
    other.
    """
    speaker = build(latent_message_multiplier=multiplier).eval()

    onehot, embeddings = speaker.decode(_prototypes())

    assert onehot.shape == (3, speaker.message_length, speaker.vocabulary + 4)
    assert embeddings.shape == (3, speaker.content_length, D_MODEL)


def test_the_causal_arm_cannot_see_the_slots_it_has_not_sampled_yet():
    """
    The property `decode_autoregressively` relies on to leave future slots
    holding their latent vectors rather than blanking them.

    Replay the loop, and at each step overwrite everything *after* the slot being
    read with noise. The embedding must not move. If the mask had stopped biting
    it would move a lot, and the loop would be reading its own unwritten future.

    This replaces a teacher-forcing reconstruction, which no longer works as an
    independent check: the sequence a step sees is now exactly the array with its
    prefix overwritten, so rebuilding it is rebuilding the loop.
    """
    speaker = _speaker().eval()

    with torch.no_grad():
        rows = list(
            speaker.latent_layer_norm(speaker.encode(_prototypes())).unbind(1)
        )

        for i in range(speaker.content_length):
            slot = speaker.first_message_slot + i

            clean = speaker.transformer(torch.stack(rows, dim=1))[:, slot, :]

            scribbled = list(rows)
            for later in range(slot + 1, speaker.latent_length):
                scribbled[later] = torch.randn_like(scribbled[later])

            assert torch.allclose(
                speaker.transformer(torch.stack(scribbled, dim=1))[:, slot, :],
                clean,
                atol=1e-5,
            ), f"slot {slot} moved when a later slot changed"

            # Stand in for the symbol the loop would have committed here.
            rows[slot] = torch.randn_like(rows[slot])


def test_the_causal_arm_conditions_on_the_symbols_it_committed():
    """
    The other half, and the one the arm exists for: a slot *does* move when an
    earlier slot changes. Without this the test above would pass on a stack that
    ignored its input entirely.
    """
    speaker = _speaker().eval()

    with torch.no_grad():
        rows = list(
            speaker.latent_layer_norm(speaker.encode(_prototypes())).unbind(1)
        )
        first = speaker.first_message_slot

        before = speaker.transformer(torch.stack(rows, dim=1))[:, first + 1, :]

        rows[first] = torch.randn_like(rows[first])
        after = speaker.transformer(torch.stack(rows, dim=1))[:, first + 1, :]

    assert not torch.allclose(before, after, atol=1e-5)


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


def test_the_causal_arm_reports_its_diagnostics_over_every_position():
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

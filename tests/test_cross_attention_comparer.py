"""
Tests for the cross-attention listener's architecture, in
code/models/receiver.py.

Runnable without pytest:  python tests/test_cross_attention_comparer.py

`ReceiverCrossAttentionLM` is the message half of what used to be
`TransformerCrossAttentionComparer`; the scoring half became an attention
discriminator, which was removed on 2026-10-01, and the encoder now feeds
`BilinearDiscriminator` like every other language model. This file follows the
whole path because the claims below are claims about the path. What is here is
the *shape*: which stage can see what, and whether the residual stream stays
where DeepNorm's constants assume it is. Everything about scale is in
test_score_scale.py, and the two-slot contract itself is in
test_receiver_slots.py.

The message reads the candidate set. That is the design -- the meaning it
refines is discriminative rather than absolute -- and it is also the shortcut:
a set summary lets the listener score "which cluster" whatever the tokens say,
which `lr_sweep_6_receiver_cross_attention_lm` measured and which is why this
encoder is the top rung. Both halves are pinned below.

The residuals are post-normed rather than added bare. `MHAttention` already
RMS-normalises its output, so `x + attn(x)` sums two tensors of norm `sqrt(d)`
and the stream grows by `sqrt(2)` a stage.

Nothing here reads the referent ordering, and `test_no_stage_can_read_the_
referent_ordering` is the test that says so: the candidate set is
cross-attention memory with no positional embedding, and anything able to index
its sequence axis could score off a position rather than off the message.
"""

import sys

import pytest
import torch

import _bootstrap  # noqa: F401

import models.receiver as R

from _bootstrap import build_listener, rung

REFERENT_DIM = 320
BATCH, N_OBJ = 32, 20

CROSS_RUNG = "17_shapeworld_receiver_cross_attention_lm.toml"


def _listener(language_model_overrides=None, discriminator_overrides=None):
    """
    The cross-attention arm end to end: `ReceiverCrossAttentionLM` feeding
        `BilinearDiscriminator`, composed the way `Receiver` composes them and
        built from rung 17.
    """
    return build_listener(
        "ReceiverCrossAttentionLM",
        "BilinearDiscriminator",
        REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
        language_model_overrides=language_model_overrides,
        discriminator_overrides=discriminator_overrides,
    ).eval()


def _inputs(listener, seed=1, correlated=True):
    """
    Referents that share a component by default, because real ones do: they
        come from one backbone over one game's worth of images, and a test on
        independent gaussians would overstate how separable the candidates are.
    """
    generator = torch.Generator().manual_seed(seed)
    spread = torch.randn(
        BATCH, N_OBJ, REFERENT_DIM, generator=generator
    )
    if correlated:
        shared = torch.randn(BATCH, 1, REFERENT_DIM, generator=generator)
        referents = 0.7 * shared + 0.3 * spread
    else:
        referents = spread
    messages = torch.randn(
        BATCH,
        listener.message_length,
        listener.token_embedding_size,
        generator=generator,
    )
    return referents, messages


def _stages(listener, referents, messages):
    """
    Both `forward`s, opened up and joined. Kept in step with them by
        `test_the_staged_walkthrough_matches_the_forward_pass` below, so that a
        change to one that is not made to the other fails loudly instead of
        leaving these tests measuring a module nobody runs.

    Note the two slots receive the referents *separately*. Each declares the
        width it wants and `Receiver` builds it an interface of its own -- a
        linear map, a non-affine norm and a mask -- so there is one projected
        copy per slot and the masks are independent. See
        `model_util.LinearInterface`.
    """
    language_model = listener.language_model
    discriminator = listener.discriminator
    seen = {}

    def record(name, tensor):
        seen[name] = tensor.clone()
        return tensor

    encoder_referents = record(
        "encoder referents",
        listener.deliver(R.LANGUAGE_MODEL_REFERENTS, referents),
    )
    encoded = record(
        "encoded message",
        language_model.message_decoder(
            language_model.message_adapter(messages), encoder_referents
        ),
    )

    scored_referents = record(
        "scored referents",
        listener.deliver(R.DISCRIMINATOR_REFERENTS, referents),
    )
    memory = record(
        "memory", listener.deliver(R.DISCRIMINATOR_MESSAGE, encoded)
    )
    # The bilinear form by hand, before the readout's two scalars: the last
    #     slot, projected, dotted with each candidate and calibrated.
    projected = discriminator.bilinear(memory[:, -1, :])
    record(
        "bilinear score",
        torch.einsum("ijh,ih->ij", (scored_referents, projected))
        / discriminator.referent_embedding_size ** 0.5,
    )
    return seen


def test_the_staged_walkthrough_matches_the_forward_pass():
    """
    The path, rebuilt from its stages: encode the message against the
        candidates, deliver it through the message interface, take the last
        slot, score bilinearly, then the readout -- multiply by `score_scale`,
        add `score_bias`. Nothing else.

    The offset is after the volume, so that it is an offset on the score rather
        than one the volume rescales. See test_score_scale.py.
    """
    # Both readout scalars pinned on: they stopped being the default on
    #     2026-09-11 with `loss = "hinge"`, and the arithmetic this test walks
    #     through is the readout *with* them. `readout` is the identity when
    #     they are off, which test_score_scale.py pins.
    listener = _listener(
        discriminator_overrides={"scale_score": True, "bias_score": True}
    )
    discriminator = listener.discriminator
    referents, messages = _inputs(listener)

    with torch.no_grad():
        stages = _stages(listener, referents, messages)
        rebuilt = (
            discriminator.score_scale * stages["bilinear score"]
            + discriminator.score_bias
        )
        actual = listener(referents, messages)

    assert torch.allclose(rebuilt, actual, atol=1e-6)


# --------------------------------------------------------------------------
# What each stage can see.
# --------------------------------------------------------------------------

def test_the_encoded_message_depends_on_the_candidate_set():
    """
    The point of the message stack's cross-attention: the message's meaning is
        allowed to be discriminative rather than absolute. Nothing else in the
        suite would notice if that branch were removed, because the module would
        still run and still score. It is also the route of the clustering
        shortcut, which is why this encoder is the top rung.
    """
    listener = _listener()
    referents, messages = _inputs(listener)
    perturbed = referents.clone()
    perturbed[:, 7, :] += torch.randn(
        BATCH, REFERENT_DIM, generator=torch.Generator().manual_seed(9)
    )

    with torch.no_grad():
        before = _stages(listener, referents, messages)["encoded message"]
        after = _stages(listener, perturbed, messages)["encoded message"]

    moved = ((after - before).norm(dim=-1) / before.norm(dim=-1)).mean().item()
    assert moved > 0.01


def test_the_score_still_depends_on_the_message():
    """
    The guard on the test above. A listener that had learned to ignore the
        message entirely would pass every structural test here, and would be
        the muting failure rather than a fixed one.
    """
    listener = _listener()
    referents, messages = _inputs(listener)
    with torch.no_grad():
        before = listener(referents, messages)
        after = listener(referents, torch.randn_like(messages))

    assert (after - before).abs().mean().item() > 0.01


def test_the_scores_are_not_a_function_of_one_referent_alone():
    """
    The message stack's cross-attention is the only place a score may depend on
        the rest of the set: every candidate is read by the encoded message, so
        perturbing one moves what every other is scored against. That is what a
        criterion like "the odd one out" would need, and also what "pick the
        cluster" needs. `BilinearDiscriminator` alone scores each candidate
        independently; this is the encoder's doing.
    """
    listener = _listener()
    referents, messages = _inputs(listener)
    perturbed = referents.clone()
    perturbed[:, 0, :] += 3.0 * torch.randn(
        BATCH, REFERENT_DIM, generator=torch.Generator().manual_seed(4)
    )

    with torch.no_grad():
        before = listener(referents, messages)
        after = listener(perturbed, messages)

    others = (after[:, 1:] - before[:, 1:]).abs().mean().item()
    assert others > 1e-4


def test_no_stage_can_read_the_referent_ordering():
    """
    Referent order *is* the label vector in this codebase:
        `data.util.split_spk_lis` writes positives into the first half of each
        agent's view and negatives into the second, and the augmentation
        permutes only within each half. Any stage able to index its own
        sequence axis could score perfectly while ignoring the message.

    Permuting the candidates must therefore permute the scores and change
        nothing else.
    """
    listener = _listener()
    referents, messages = _inputs(listener)
    order = torch.randperm(N_OBJ, generator=torch.Generator().manual_seed(3))

    with torch.no_grad():
        before = listener(referents, messages)
        after = listener(referents[:, order, :], messages)

    assert torch.allclose(before[:, order], after, atol=1e-5)


# --------------------------------------------------------------------------
# The residual stream.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "stage",
    [
        "encoded message",
    ],
)
def test_the_residual_stream_does_not_grow(stage):
    """
    Every join inside a `DecoderBlock` is `RMSNorm(alpha * x + beta * branch)`,
        so each stack leaves its stream at unit RMS per token. Replace any of
        them with a bare `x + attn(x)` and this fails at `sqrt(2)`, because
        `MHAttention` has already normalised what it returns.
    """
    listener = _listener()
    referents, messages = _inputs(listener)
    with torch.no_grad():
        state = _stages(listener, referents, messages)[stage]

    rms = state.pow(2).mean(dim=-1).sqrt()
    assert rms.mean().item() == pytest.approx(1.0, abs=1e-3)


def test_a_bare_add_would_have_grown_it():
    """
    The counterfactual, so the test above is not just asserting that RMSNorm
        normalises. Two tensors at norm `sqrt(d)` sum to one at `sqrt(2d)`.
    """
    listener = _listener()
    referents, messages = _inputs(listener)
    with torch.no_grad():
        stages = _stages(listener, referents, messages)
        stream = stages["encoded message"]
        candidates = stages["encoder referents"]
        attended = listener.language_model.message_decoder.blocks[
            0
        ].cross_attention(stream, candidates, candidates)
        bare = stream + attended

    assert bare.pow(2).mean(dim=-1).sqrt().mean().item() > 1.3


def test_the_memory_reaches_the_scored_stack_normalised():
    """
    Whatever the language model hands over arrives at whatever magnitude it
        happens to have. `message_decoder`'s last post-norm makes that safe by
        accident; a GRU state would not, and the slot is swappable. Hence the
        norm on the message interface, which makes it safe on purpose and
        unconditionally.
    """
    listener = _listener()
    referents, messages = _inputs(listener)
    with torch.no_grad():
        memory = _stages(listener, referents, messages)["memory"]

    rms = memory.pow(2).mean(dim=-1).sqrt()
    assert rms.mean().item() == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# Construction.
# --------------------------------------------------------------------------

def test_the_depth_key_sizes_the_stack():
    """
    The stack's depth is `[receiver_language_model] layers`, the same key
        `ReceiverGRULM` reads for its own depth.
    """
    listener = _listener(language_model_overrides=dict(layers=2))

    assert len(listener.language_model.message_decoder.blocks) == 2


@pytest.mark.parametrize("layers", [2, 5])
def test_the_stack_gets_deepnorm_for_its_own_depth(layers):
    """
    `decoder=True`, because these blocks have three residual branches rather
        than two -- self-attention, cross-attention into the candidates, and a
        feedforward.
    """
    listener = _listener(language_model_overrides=dict(layers=layers))

    assert listener.language_model.alpha == pytest.approx((3 * layers) ** 0.25)
    assert listener.language_model.beta == pytest.approx((12 * layers) ** -0.25)


def test_stochastic_depth_is_suppressed_only_at_a_single_layer():
    """
    `depthwise_linear_stochastic_depth` spreads the rate linearly across
        layers, so a one-layer stack would get a single rate of 0.0 anyway. It
        used to be gated on `layers // 2 > 1`, which silenced it at three layers
        -- a live depth for this module.

    The rate is passed in rather than inherited from the config, which it used
        to be. `DEFAULT.toml` set 0.1 everywhere when this was written and now
        sets 0.0 everywhere (see the comment at `[sender_language_model]
        stochastic_depth`), which quietly turned the "> 0.0" assertions into
        assertions about the default rather than about the gating. Stating the
        rate here tests the thing the test is named for whatever the default
        becomes.
    """
    rate = dict(stochastic_depth=0.1)

    single = _listener(language_model_overrides=dict(layers=1, **rate))
    assert single.language_model.stochastic_depth == 0.0

    for layers in (2, 3, 4):
        deep = _listener(language_model_overrides=dict(layers=layers, **rate))
        assert deep.language_model.stochastic_depth > 0.0


def test_the_candidate_set_is_read_without_positions():
    """
    The construction half of the ordering guard. The message stream runs rotary
        -- its axis is an order -- but the cross-attention into the candidates
        carries no positional embedding, so no block can index the candidate
        axis. `test_no_stage_can_read_the_referent_ordering` measures the
        consequence; this names the setting.
    """
    listener = _listener()

    for block in listener.language_model.message_decoder.blocks:
        assert block.cross_first is False
        assert block.cross_attention.rotary_embedding is None
        assert block.self_attention.rotary_embedding is not None


def test_reset_parameters_leaves_nothing_trained():
    """
    The adapters were missing from this list once, so a reset listener kept the
        projections that map referents and messages into `d_model` while
        everything downstream of them was re-drawn. They are `Receiver`'s
        interfaces now and the reset is a walk over the container rather than a
        list of names, which is the shape that bug argues for -- so this covers
        the interfaces too, through the shim's mirror of
        `Receiver.reset_parameters`.
    """
    listener = _listener()
    with torch.no_grad():
        for parameter in listener.parameters():
            parameter.add_(1.0)
    before = [p.detach().clone() for p in listener.parameters()]

    listener.reset_parameters()

    # broccoli owns these and does not re-draw them, which is correct for both:
    #     `rotary_embedding.freqs` is a deterministic function of position, so
    #     there is nothing to draw, and `swish_beta` is the activation's own
    #     parameter rather than the block's. Excluded by name rather than by
    #     loosening the assertion, so that a *new* untouched parameter still
    #     fails.
    BROCCOLI_INERT = ("rotary_embedding.freqs", "swish_beta")
    unchanged = [
        name
        for (name, after), stale in zip(listener.named_parameters(), before)
        if torch.equal(after, stale)
        and not name.endswith(BROCCOLI_INERT)
    ]
    assert not unchanged, unchanged


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

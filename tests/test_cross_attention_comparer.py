"""
Tests for the attention listener's architecture, in code/models/receiver.py.

Runnable without pytest:  python tests/test_cross_attention_comparer.py

`ReceiverCrossAttentionLM` and `AttentionDiscriminator` are the two halves of
what used to be `TransformerCrossAttentionComparer`, and this file follows the
whole path because the claims below are claims about the path. What is here is
the *shape*: which stage can see what, and whether the residual stream stays
where DeepNorm's constants assume it is. Everything about scale is in
test_score_scale.py, and the two-slot contract itself is in
test_receiver_slots.py.

Four claims, each of which failed in some version this replaces and each of
which cost something measurable on the ablation's rungs 11 and 12.

The message reads the candidate set. Without that the message stack sees the
message alone and can only build an absolute meaning, when the task is
discriminative.

The message reaches the scored stream in every referent block, not once. The
structure before this one crossed it in exactly once, so the only thing that
could separate two candidates was the difference between their attention weights
over the message -- a small perturbation about a near-flat softmax, and second
order at initialisation where the bilinear comparison's message term is first
order. Rungs 11 to 14 sat at 0.5000 accuracy for thirty epochs.

Every stage joins its input back on. An earlier `cross_attention` had no
residual, so a candidate reached the score only through near-uniform attention
weights: between-object variance was 41.5% of the total going into that stage
and 22.1% coming out. That is rung 11 taking four times as long as the baseline
to become informative.

And the residuals are post-normed rather than added bare. `MHAttention` already
RMS-normalises its output, so `x + attn(x)` sums two tensors of norm `sqrt(d)`
and the stream grows by `sqrt(2)` a stage.

Nothing here reads the referent ordering, and `test_no_stage_can_read_the_
referent_ordering` is the test that says so: the referent stack is built
`causal=False`, and a causal mask over candidates would let a score be read off
a position rather than off the message.
"""

import sys

import pytest
import torch

import _bootstrap  # noqa: F401

import models.receiver as R

from _bootstrap import build_listener, rung

REFERENT_DIM = 320
BATCH, N_OBJ = 32, 20

CROSS_RUNG = "15_shapeworld_receiver_cross_attention_lm.toml"


def _listener(language_model_overrides=None, discriminator_overrides=None):
    """
    The attention arm end to end: `ReceiverCrossAttentionLM` feeding
        `AttentionDiscriminator`, composed the way `Receiver` composes them and
        built from rung 11, which is the config that states widths for both.
    """
    return build_listener(
        "ReceiverCrossAttentionLM",
        "AttentionDiscriminator",
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

    Note the two slots adapt the referents *separately*. Each owns its
        projection and its norm, because the width a projection targets is a
        property of the consumer; see test_receiver_slots.py for why that is not
        shared.
    """
    language_model = listener.language_model
    discriminator = listener.discriminator
    seen = {}

    def record(name, tensor):
        seen[name] = tensor.clone()
        return tensor

    encoder_referents = record(
        "encoder referents",
        language_model.referent_layer_norm(
            language_model.referent_adapter(referents)
        ),
    )
    encoded = record(
        "encoded message",
        language_model.message_decoder(
            language_model.message_adapter(messages), encoder_referents
        ),
    )

    scored_referents = record(
        "scored referents",
        discriminator.referent_layer_norm(
            discriminator.referent_adapter(referents)
        ),
    )
    memory = record(
        "memory",
        discriminator.memory_layer_norm(discriminator.memory_adapter(encoded)),
    )
    refined = record(
        "refined referents",
        discriminator.referent_decoder(scored_referents, memory),
    )
    record("attention readout", discriminator.decision(refined).squeeze(-1))
    record("bilinear readout", discriminator.bilinear(referents, encoded))
    return seen


def _object_share(scores_or_states):
    """
    How much of the variation is between the objects of a game rather than
        between games. The quantity the missing residual was destroying.
    """
    within = (
        scores_or_states - scores_or_states.mean(dim=1, keepdim=True)
    ).std()
    between = scores_or_states.mean(dim=1).std()
    return (within / between).item()


def test_the_staged_walkthrough_matches_the_forward_pass():
    """
    The readout is `decision`, then the mix. `decision` alone used to be the
        whole of it -- and before that `F.normalize(decision.weight)` against a
        learnable `score_scale`, and then `decision_layer_norm -> decision`; see
        test_score_scale.py for the first change and docs/architecture.md for
        the second.

    What follows it now is the interpolation with the bilinear path, and this is
        the test that pins its arithmetic: mix the two readouts at
        `mix_weight`, add a bias. Nothing else. There was briefly a
        `BatchNorm1d(1)` and a fixed gain in this position, which had to be
        rebuilt on the same flattening `forward` used and left this test
        sensitive to call order through the running estimates. Both are gone,
        and so is the `standardise` on each path that stood here after them --
        it survives only in the telemetry block, so the branches now reach the
        mix at their own magnitudes and there is no volume scalar to apply.
    """
    listener = _listener()
    discriminator = listener.discriminator
    referents, messages = _inputs(listener)

    with torch.no_grad():
        stages = _stages(listener, referents, messages)
        weight = discriminator.mix_weight
        rebuilt = (
            (1.0 - weight) * stages["bilinear readout"]
            + weight * stages["attention readout"]
        ) + discriminator.mix_bias
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
        still run and still score.
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


def test_referent_identity_survives_to_the_readout():
    """
    The residuals through the referent stack, measured as the thing they
        protect. Without one on the cross-attention, that stage roughly halved
        the between-object share of the variance -- 0.415 in, 0.221 out -- and
        the score inherited the loss.

    Measured on the attention path's own readout rather than on the returned
        score, and that is not a softening. `standardise` sets the
        between-object spread of every game to one by construction, so a
        measurement taken after it would report the same number whatever the
        stack did, and would pass a comparer that had destroyed the signal
        entirely.
    """
    listener = _listener()
    referents, messages = _inputs(listener)
    with torch.no_grad():
        stages = _stages(listener, referents, messages)

    entering = _object_share(stages["scored referents"])
    leaving = _object_share(stages["attention readout"].unsqueeze(-1))

    assert leaving > 0.6 * entering


def test_the_scores_are_not_a_function_of_one_referent_alone():
    """
    The referent stack's self-attention is the only place a score may depend on
        the rest of the set, and it is what a criterion like "the odd one out"
        would need. Perturbing one candidate must move the others' scores.
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
        nothing else. Note this now covers the mix as well: `standardise` works
        over the candidate axis, so a reduction there that was not permutation
        equivariant would show up here.
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
        "refined referents",
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
        adapted = stages["scored referents"]
        memory = stages["memory"]
        attended = listener.discriminator.referent_decoder.blocks[
            0
        ].cross_attention(adapted, memory, memory)
        bare = adapted + attended

    assert bare.pow(2).mean(dim=-1).sqrt().mean().item() > 1.3


def test_the_memory_reaches_the_scored_stack_normalised():
    """
    A post-norm stack normalises its own stream and never its memory, so
        whatever the language model hands over arrives at whatever magnitude it
        happens to have. `message_decoder`'s last post-norm used to make that
        safe by accident; a GRU state would not, and the slot is swappable now.
        Hence `memory_layer_norm`, which makes it safe on purpose.
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

def test_each_depth_key_sizes_its_own_stack_and_nothing_else():
    """
    One key was once a total, split inside a single module between two stacks,
        so asking for one more block moved two. Now each stack's depth is the
        `layers` key of its own config table, which is what stops that
        recurring -- and is the same key `ReceiverGRULM` reads for its own
        depth, on the same argument that these are different modules.
    """
    listener = _listener(
        language_model_overrides=dict(layers=2),
        discriminator_overrides=dict(layers=5),
    )

    assert len(listener.language_model.message_decoder.blocks) == 2
    assert len(listener.discriminator.referent_decoder.blocks) == 5


def test_each_stack_gets_deepnorm_for_its_own_depth():
    """
    `decoder=True`, because these blocks have three residual branches rather
        than two, and at its own depth, because the stacks are sized
        independently. Resolving both from one number would scale the shallower
        stack's branches as if it were the deeper one.
    """
    listener = _listener(
        language_model_overrides=dict(layers=2),
        discriminator_overrides=dict(layers=5),
    )

    assert listener.language_model.alpha == pytest.approx((3 * 2) ** 0.25)
    assert listener.language_model.beta == pytest.approx((12 * 2) ** -0.25)
    assert listener.discriminator.alpha == pytest.approx((3 * 5) ** 0.25)
    assert listener.discriminator.beta == pytest.approx((12 * 5) ** -0.25)


def test_stochastic_depth_is_suppressed_only_at_a_single_layer():
    """
    `depthwise_linear_stochastic_depth` spreads the rate linearly across
        layers, so a one-layer stack would get a single rate of 0.0 anyway. It
        used to be gated on `layers // 2 > 1`, which silenced it at three layers
        -- a live depth for this module. Asked of each stack separately, so a
        one-block stack beside a deep one still gets nothing.

    The rate is passed in rather than inherited from the config, which it used
        to be. `DEFAULT.toml` set 0.1 everywhere when this was written and now
        sets 0.0 everywhere (see the comment at `[sender_language_model]
        stochastic_depth`), which quietly turned the "> 0.0" assertions into
        assertions about the default rather than about the gating. Stating the
        rate here tests the thing the test is named for whatever the default
        becomes.
    """
    rate = dict(stochastic_depth=0.1)

    single = _listener(
        language_model_overrides=dict(layers=1, **rate),
        discriminator_overrides=dict(layers=4, **rate),
    )
    assert single.language_model.stochastic_depth == 0.0
    assert single.discriminator.stochastic_depth > 0.0

    for layers in (2, 3, 4):
        deep = _listener(
            language_model_overrides=dict(layers=layers, **rate),
            discriminator_overrides=dict(layers=layers, **rate),
        )
        assert deep.language_model.stochastic_depth > 0.0
        assert deep.discriminator.stochastic_depth > 0.0


def test_the_referent_stack_is_not_causal_and_reads_the_message_first():
    """
    Two settings that `DecoderBlock` defaults the other way, because its default
        caller is a speaker generating a sequence.

    `causal=False` is what `test_no_stage_can_read_the_referent_ordering` above
        measures the consequence of; this is the same claim read off the
        construction, so a regression names itself rather than showing up as a
        permutation failure.

    `cross_first` puts the message ahead of the candidates reading each other,
        so what the self-attention compares is message-informed. That
        self-attention is the route to the concept game's clustering shortcut,
        which is reachable with the message scrambled entirely.
    """
    listener = _listener()

    for block in listener.discriminator.referent_decoder.blocks:
        assert block.causal is False
        assert block.self_attention.causal is False
        assert block.cross_first is True

    for block in listener.language_model.message_decoder.blocks:
        assert block.cross_first is False


def test_the_referent_stack_carries_no_positional_information():
    """
    The other half of the ordering guard. A rotary embedding on the candidate
        axis would let a block index its own sequence even without a mask.
    """
    listener = _listener()
    referent_decoder = listener.discriminator.referent_decoder

    assert referent_decoder.absolute_position_embedding is None
    for block in referent_decoder.blocks:
        assert block.rotary_embedding is None


def test_reset_parameters_leaves_nothing_trained():
    """
    The adapters were missing from this list once, so a reset listener kept the
        projections that map referents and messages into `d_model` while
        everything downstream of them was re-drawn.
    """
    listener = _listener()
    with torch.no_grad():
        for parameter in listener.parameters():
            parameter.add_(1.0)
    before = [p.detach().clone() for p in listener.parameters()]

    listener.language_model.reset_parameters()
    listener.discriminator.reset_parameters()

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

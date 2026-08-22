"""
Tests for `TransformerCrossAttentionComparer`'s architecture, in
code/models/receiver.py.

Runnable without pytest:  python tests/test_cross_attention_comparer.py

Everything about this module's *scale* is in test_score_scale.py. What is here
is the shape: which stage can see what, and whether the residual stream stays
where DeepNorm's constants assume it is.

The module is two `TransformerDecoder` stacks. Four claims, each of which failed
in some version this replaces and each of which cost something measurable on the
ablation's rungs 11 and 12.

The message reads the candidate set. Without that the message stack sees the
message alone and can only build an absolute meaning, when the task is
discriminative.

The message reaches the scored stream in every referent block, not once. The
structure before this one crossed it in exactly once, so the only thing that
could separate two candidates was the difference between their attention weights
over the message -- a small perturbation about a near-flat softmax, and second
order at initialisation where the bilinear comparer's message term is first
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

from _bootstrap import config_section, rung

REFERENT_DIM = 320
BATCH, N_OBJ = 32, 20


def _comparer(**overrides):
    config = config_section(
        "receiver_comparer",
        rung("11_shapeworld_receiver_cross_attention.toml"),
        **overrides,
    )
    torch.manual_seed(0)
    return R.TransformerCrossAttentionComparer(REFERENT_DIM, **config).eval()


def _inputs(comparer, seed=1, correlated=True):
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
        comparer.message_length,
        comparer.token_embedding_size,
        generator=generator,
    )
    return referents, messages


def _stages(comparer, referents, messages):
    """
    `forward`, opened up. Kept in step with it by
        `test_the_staged_walkthrough_matches_the_forward_pass` below, so that a
        change to one that is not made to the other fails loudly instead of
        leaving these tests measuring a module nobody runs.
    """
    seen = {}

    def record(name, tensor):
        seen[name] = tensor.clone()
        return tensor

    referents = comparer.referent_adapter(referents)
    referents = comparer.referent_layer_norm(referents)
    referents = record("normed referents", comparer.input_dropout(referents))

    messages = comparer.message_adapter(messages)
    encoded = record(
        "encoded message", comparer.message_decoder(messages, referents)
    )
    record(
        "refined referents", comparer.referent_decoder(referents, encoded)
    )
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
    The readout is `decision` alone. It used to be
        `F.normalize(decision.weight)` against a learnable `score_scale`, and
        later `decision_layer_norm -> decision`; see test_score_scale.py for
        the first change and docs/architecture.md for the second. The referent
        stack's last post-norm is an `RMSNorm`, so the candidates already reach
        the readout at equal length and the extra norm bought nothing.

    Nothing follows `decision`. There was briefly a `BatchNorm1d(1)` and a
        fixed gain between it and the return, which had to be rebuilt on the
        same flattening `forward` used and left this test sensitive to call
        order through the running estimates. Both are gone, so the rebuild is
        one line and order-independent.
    """
    comparer = _comparer()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        refined = _stages(comparer, referents, messages)["refined referents"]
        rebuilt = comparer.decision(refined).squeeze(-1)
        actual = comparer(referents, messages)

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
    comparer = _comparer()
    referents, messages = _inputs(comparer)
    perturbed = referents.clone()
    perturbed[:, 7, :] += torch.randn(
        BATCH, REFERENT_DIM, generator=torch.Generator().manual_seed(9)
    )

    with torch.no_grad():
        before = _stages(comparer, referents, messages)["encoded message"]
        after = _stages(comparer, perturbed, messages)["encoded message"]

    moved = ((after - before).norm(dim=-1) / before.norm(dim=-1)).mean().item()
    assert moved > 0.01


def test_the_score_still_depends_on_the_message():
    """
    The guard on the test above. A comparer that had learned to ignore the
        message entirely would pass every structural test here, and would be
        the muting failure rather than a fixed one.
    """
    comparer = _comparer()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        before = comparer(referents, messages)
        after = comparer(referents, torch.randn_like(messages))

    assert (after - before).abs().mean().item() > 0.01


def test_referent_identity_survives_to_the_readout():
    """
    The residuals through the referent stack, measured as the thing they
        protect. Without one on the cross-attention, that stage roughly halved
        the between-object share of the variance -- 0.415 in, 0.221 out -- and
        the score inherited the loss.
    """
    comparer = _comparer()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        stages = _stages(comparer, referents, messages)
        scores = comparer(referents, messages)

    entering = _object_share(stages["normed referents"])
    leaving = _object_share(scores.unsqueeze(-1))

    assert leaving > 0.6 * entering


def test_the_scores_are_not_a_function_of_one_referent_alone():
    """
    The referent stack's self-attention is the only place a score may depend on
        the rest of the set, and it is what a criterion like "the odd one out"
        would need. Perturbing one candidate must move the others' scores.
    """
    comparer = _comparer()
    referents, messages = _inputs(comparer)
    perturbed = referents.clone()
    perturbed[:, 0, :] += 3.0 * torch.randn(
        BATCH, REFERENT_DIM, generator=torch.Generator().manual_seed(4)
    )

    with torch.no_grad():
        before = comparer(referents, messages)
        after = comparer(perturbed, messages)

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
    comparer = _comparer()
    referents, messages = _inputs(comparer)
    order = torch.randperm(N_OBJ, generator=torch.Generator().manual_seed(3))

    with torch.no_grad():
        before = comparer(referents, messages)
        after = comparer(referents[:, order, :], messages)

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
    comparer = _comparer()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        state = _stages(comparer, referents, messages)[stage]

    rms = state.pow(2).mean(dim=-1).sqrt()
    assert rms.mean().item() == pytest.approx(1.0, abs=1e-3)


def test_a_bare_add_would_have_grown_it():
    """
    The counterfactual, so the test above is not just asserting that RMSNorm
        normalises. Two tensors at norm `sqrt(d)` sum to one at `sqrt(2d)`.
    """
    comparer = _comparer()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        stages = _stages(comparer, referents, messages)
        adapted = stages["normed referents"]
        encoded = stages["encoded message"]
        attended = comparer.referent_decoder.blocks[0].cross_attention(
            adapted, encoded, encoded
        )
        bare = adapted + attended

    assert bare.pow(2).mean(dim=-1).sqrt().mean().item() > 1.3


# --------------------------------------------------------------------------
# Construction.
# --------------------------------------------------------------------------

def test_each_depth_key_sizes_its_own_stack_and_nothing_else():
    """
    One key was once a total, split inside the module between two stacks, so
        asking for one more block moved two. Two keys, each a block count of the
        stack it is named for, is what stops that recurring.
    """
    comparer = _comparer(message_layers=2, referent_layers=5)

    assert len(comparer.message_decoder.blocks) == 2
    assert len(comparer.referent_decoder.blocks) == 5


def test_each_stack_gets_deepnorm_for_its_own_depth():
    """
    `decoder=True`, because these blocks have three residual branches rather
        than two, and at its own depth, because the stacks are sized
        independently. Resolving both from one number would scale the shallower
        stack's branches as if it were the deeper one.
    """
    comparer = _comparer(message_layers=2, referent_layers=5)

    assert comparer.message_alpha == pytest.approx((3 * 2) ** 0.25)
    assert comparer.message_beta == pytest.approx((12 * 2) ** -0.25)
    assert comparer.referent_alpha == pytest.approx((3 * 5) ** 0.25)
    assert comparer.referent_beta == pytest.approx((12 * 5) ** -0.25)


def test_stochastic_depth_is_suppressed_only_at_a_single_layer():
    """
    `depthwise_linear_stochastic_depth` spreads the rate linearly across
        layers, so a one-layer stack would get a single rate of 0.0 anyway. It
        used to be gated on `layers // 2 > 1`, which silenced it at three layers
        -- a live depth for this module. Asked of each stack separately, so a
        one-block stack beside a deep one still gets nothing.
    """
    single = _comparer(message_layers=1, referent_layers=4)
    assert single.message_stochastic_depth == 0.0
    assert single.referent_stochastic_depth > 0.0

    for layers in (2, 3, 4):
        deep = _comparer(message_layers=layers, referent_layers=layers)
        assert deep.message_stochastic_depth > 0.0
        assert deep.referent_stochastic_depth > 0.0


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
    comparer = _comparer()

    for block in comparer.referent_decoder.blocks:
        assert block.causal is False
        assert block.self_attention.causal is False
        assert block.cross_first is True

    for block in comparer.message_decoder.blocks:
        assert block.cross_first is False


def test_the_referent_stack_carries_no_positional_information():
    """
    The other half of the ordering guard. A rotary embedding on the candidate
        axis would let a block index its own sequence even without a mask.
    """
    comparer = _comparer()

    assert comparer.referent_decoder.absolute_position_embedding is None
    for block in comparer.referent_decoder.blocks:
        assert block.rotary_embedding is None


def test_reset_parameters_leaves_nothing_trained():
    """
    The adapters were missing from this list once, so a reset listener kept the
        projections that map referents and messages into `d_model` while
        everything downstream of them was re-drawn.
    """
    comparer = _comparer()
    with torch.no_grad():
        for parameter in comparer.parameters():
            parameter.add_(1.0)
    before = [p.detach().clone() for p in comparer.parameters()]

    comparer.reset_parameters()

    # broccoli owns these and does not re-draw them, which is correct for both:
    #     `rotary_embedding.freqs` is a deterministic function of position, so
    #     there is nothing to draw, and `swish_beta` is the activation's own
    #     parameter rather than the block's. Excluded by name rather than by
    #     loosening the assertion, so that a *new* untouched parameter still
    #     fails.
    BROCCOLI_INERT = ("rotary_embedding.freqs", "swish_beta")
    unchanged = [
        name
        for (name, after), stale in zip(comparer.named_parameters(), before)
        if torch.equal(after, stale)
        and not name.endswith(BROCCOLI_INERT)
    ]
    assert not unchanged, unchanged


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

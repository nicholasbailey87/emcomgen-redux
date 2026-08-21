"""
Tests for `TransformerCrossAttentionComparer`'s architecture, in
code/models/receiver.py.

Runnable without pytest:  python tests/test_cross_attention_comparer.py

Everything about this module's *scale* is in test_score_scale.py. What is here
is the shape: which stage can see what, and whether the residual stream stays
where DeepNorm's constants assume it is.

Three claims, each of which failed in the version this replaces and each of
which cost something measurable on the ablation's rungs 11 and 12.

The message reads the candidate set before it is encoded. Without that the
encoder sees the message alone and can only build an absolute meaning, when the
task is discriminative.

Every stage joins its input back on. The old `cross_attention` had no residual,
so a candidate reached the score only through near-uniform attention weights:
between-object variance was 41.5% of the total going into that stage and 22.1%
coming out, and it was the only stage that lost any. That is rung 11 taking
four times as long as the baseline to become informative.

And the residuals are post-normed rather than added bare. `MHAttention` already
RMS-normalises its output, so `x + attn(x)` sums two tensors of norm `sqrt(d)`
and the stream grows by `sqrt(2)` a stage.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import models.receiver as R  # noqa: E402
import parse_config  # noqa: E402

CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "ablation", "configs"
)

REFERENT_DIM = 320
BATCH, N_OBJ = 32, 20


def _comparer(**overrides):
    config = parse_config.get_config(
        os.path.join(CONFIG_DIR, "11_shapeworld_receiver_cross_attention.toml")
    )["receiver_comparer"]
    config = {**config, **overrides}
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
    # Each stage is snapshotted as it is produced, not at the end, because
    #     broccoli's `TransformerEncoder.preprocess` adds its position
    #     embedding with `x += position_embedding` (transformer.py:856) -- in
    #     place, on the tensor handed to it. So `comparer.encoding(read)`
    #     mutates `read`, and reading it afterwards gives the message *plus* a
    #     position embedding. That is how this helper first reported the
    #     residual stream at 1.42: `sqrt(2)`, and indistinguishable from what a
    #     missing post-norm would look like. See the note in
    #     `TransformerCrossAttentionComparer.forward`.
    seen = {}

    def record(name, tensor):
        seen[name] = tensor.clone()
        return tensor

    referents = comparer.referent_adapter(referents)
    referents = comparer.referent_layer_norm(referents)
    referents = record("normed referents", comparer.input_dropout(referents))

    messages = comparer.message_adapter(messages)
    read = record(
        "message read against the set",
        comparer._residual(
            messages,
            comparer.message_cross_attention(messages, referents, referents),
            comparer.message_residual_norm,
        ),
    )
    encoded = record("encoded message", comparer.encoding(read))
    enriched = record(
        "enriched referents",
        comparer._residual(
            referents,
            comparer.referent_cross_attention(referents, encoded, encoded),
            comparer.referent_residual_norm,
        ),
    )
    record(
        "refined referents",
        comparer._residual(
            enriched,
            comparer.referent_self_attention(enriched, enriched, enriched),
            comparer.referent_self_attention_norm,
        ),
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
    The readout is rebuilt here as `decision -> score_norm -> decision_gain`,
        which is what `forward` does now. It used to be
        `F.normalize(decision.weight)` against a learnable `score_scale`; see
        test_score_scale.py for why that changed.

    `score_norm` has to be run on the same flattening `forward` uses -- one
        statistic across every slot, not one per slot -- so this reproduces the
        `reshape` rather than calling the module on the `(batch, n_obj)` tensor,
        which would not even be a valid shape for `BatchNorm1d(1)`.

    Both calls run in train mode, so both normalise by their own batch
        statistics. Running the rebuild first would otherwise leave the running
        estimates one update further along than `forward` expects, and the two
        would differ for a reason that has nothing to do with the stages.
    """
    comparer = _comparer()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        refined = _stages(comparer, referents, messages)["refined referents"]
        normed = comparer.decision_layer_norm(refined)
        scores = comparer.decision(normed).squeeze(-1)
        rebuilt = comparer.decision_gain * comparer.score_norm(
            scores.reshape(-1, 1)
        ).reshape(scores.shape)
        actual = comparer(referents, messages)

    assert torch.allclose(rebuilt, actual, atol=1e-6)


# --------------------------------------------------------------------------
# What each stage can see.
# --------------------------------------------------------------------------

def test_the_encoded_message_depends_on_the_candidate_set():
    """
    The point of reading the referents before encoding: the message's meaning
        is allowed to be discriminative rather than absolute. Nothing else in
        the suite would notice if that stage were removed, because the module
        would still run and still score.
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
    The residual on the cross-attention, measured as the thing it protects.
        Without it this stage roughly halved the between-object share of the
        variance -- 0.415 in, 0.221 out -- and the score inherited the loss.
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
    `referent_self_attention` is the only stage at which a score may depend on
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
        "message read against the set",
        "encoded message",
        "enriched referents",
        "refined referents",
    ],
)
def test_the_residual_stream_does_not_grow(stage):
    """
    Every join is `RMSNorm(alpha * x + beta * attended)`, so each one leaves the
        stream at unit RMS per token. Replace any of them with a bare `x +
        attn(x)` and this fails at `sqrt(2)`, because `MHAttention` has already
        normalised what it returns.
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
        adapted = comparer.referent_layer_norm(
            comparer.referent_adapter(referents)
        )
        encoded = comparer.encoding(
            comparer.message_adapter(messages)
        )
        attended = comparer.referent_cross_attention(adapted, encoded, encoded)
        bare = adapted + attended

    assert bare.pow(2).mean(dim=-1).sqrt().mean().item() > 1.3


# --------------------------------------------------------------------------
# Construction.
# --------------------------------------------------------------------------

def test_layers_is_the_encoder_depth_and_nothing_elses():
    """
    It used to be a total split between a reading stack and a fusion stack, so
        `layers = 5` bought a 3-layer encoder and asking for one more block
        moved two.
    """
    for layers in (1, 3, 4, 7):
        assert len(_comparer(layers=layers).encoding.blocks) == layers


def test_stochastic_depth_is_suppressed_only_at_a_single_layer():
    """
    `depthwise_linear_stochastic_depth` spreads the rate linearly across
        layers, so a one-layer stack would get a single rate of 0.0 anyway. It
        used to be gated on `layers // 2 > 1`, which silenced it at `layers = 3`
        -- a live depth for this module.
    """
    assert _comparer(layers=1).stochastic_depth == 0.0
    for layers in (2, 3, 4):
        assert _comparer(layers=layers).stochastic_depth > 0.0


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

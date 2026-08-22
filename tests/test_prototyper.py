"""
Tests for the two prototypers in code/models/sender.py.

Runnable without pytest:  python tests/test_prototyper.py

The prototyper pools a concept's examples into the one vector the language model
speaks from, and `AttentionPrototyper` does it with a softmax over the examples.
That softmax had the same defect the softmax over tokens had before
`layer_norm_logits`: its input is `w . x`, so both where it starts and how fast
it moves are set by the magnitude of whatever the backbone emits, not by
anything configured or learned.

Where it starts. With broccoli's default init the scoring direction is random,
and the sharpness of the resulting softmax goes as the between-example standard
deviation of the embeddings. Measured across the ablation's backbones that
spans a factor of fifty, which puts a fresh Conv4 pooler within a whisker of
selecting a single example while a fresh pooler on a normalised backbone sits
within a few percent of the mean. Rung 3 was then not "rung 1 plus attention
pooling" but rung 1 with its prototype replaced by one arbitrary positive
example. Zero-initialised scoring weights make the opening *exactly* the parent
rung's average, which is what makes the rung an ablation of one thing.

How fast it moves. Score spread is `||w||` times the embeddings' scale, so an
arm on a large-magnitude backbone departs from the mean tens of times faster
than one on a normalised backbone -- the same architecture dependence, moved
from the opening to the rate. Scoring from `LayerNorm`ed examples removes it.
The pooled *values* stay un-normalised, so the prototype the language model
receives is unchanged.

And the diagnostics, because neither of those is visible in accuracy. A pooler
that stayed at the mean and a pooler that found something look identical in
every other column of metrics.csv.
"""

import math

import pytest
import torch

import _bootstrap  # noqa: F401

import models.sender as S


D_MODEL = 512
N_EXAMPLES = 20  # ten positive, ten negative, as the speaker is handed them

# The two ends of the ablation's feature scales, from the probe of the eight
#     smoke-test runs: Conv4 is the one sender backbone whose output is not
#     scale-normalised, and a post-norm ViT sits near unity.
CONV4_SCALE = 25.8
NORMALISED_SCALE = 0.45


def _examples(scale, batch=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return scale * torch.randn(
        batch, N_EXAMPLES, D_MODEL, generator=generator
    )


def _halves(samples):
    half = samples.size(1) // 2
    return samples[:, :half], samples[:, half:]


# ------------------------------------------------- 1. where the pooler opens --

@pytest.mark.parametrize("scale", [CONV4_SCALE, NORMALISED_SCALE])
def test_attention_pooling_opens_at_the_average(scale):
    """
    The whole point of the zero init: at step zero this rung *is* its parent
    rung, on any backbone. Checked at both extremes of feature scale because
    the failure it replaces is precisely one that only shows at one end.
    """
    prototyper = S.AttentionPrototyper(D_MODEL)
    samples = _examples(scale)
    positive, negative = prototyper(samples)
    positive_examples, negative_examples = _halves(samples)

    assert torch.allclose(positive, positive_examples.mean(1), atol=1e-3)
    assert torch.allclose(negative, negative_examples.mean(1), atol=1e-3)


def test_attention_pooling_matches_the_average_prototyper_exactly():
    """
    Stated against the other class rather than against `mean`, since it is the
    ladder's parent rung that has to be reproduced, not an arithmetic identity.
    """
    samples = _examples(CONV4_SCALE)
    attention = S.AttentionPrototyper(D_MODEL)(samples)
    average = S.AveragePrototyper()(samples)

    for pooled, averaged in zip(attention, average):
        assert torch.allclose(pooled, averaged, atol=1e-3)


def test_zero_initialised_scoring_weights_still_receive_gradient():
    """
    The objection to a zero init, and why it does not apply. There is one
    output unit, so there is no symmetry between units to break, and
    `dL/dW = sum_i (dL/ds_i) x_i` depends on the examples rather than on `W`.
    """
    prototyper = S.AttentionPrototyper(D_MODEL)
    positive, negative = prototyper(_examples(NORMALISED_SCALE))
    (positive.sum() + negative.sum()).backward()

    for pool in (prototyper.pos_pool, prototyper.neg_pool):
        assert pool.attention[0].weight.grad.norm().item() > 0.0


def test_reset_parameters_returns_the_pooler_to_the_average():
    """
    A reset speaker must not keep a trained pooling any more than it keeps a
    trained channel: `reset_parameters` has to reimpose the zero, not just
    delegate to broccoli's own init.
    """
    prototyper = S.AttentionPrototyper(D_MODEL)
    with torch.no_grad():
        prototyper.pos_pool.attention[0].weight.normal_()
        prototyper.neg_pool.attention[0].weight.normal_()

    prototyper.reset_parameters()
    samples = _examples(CONV4_SCALE)
    positive, negative = prototyper(samples)
    positive_examples, negative_examples = _halves(samples)

    assert torch.allclose(positive, positive_examples.mean(1), atol=1e-3)
    assert torch.allclose(negative, negative_examples.mean(1), atol=1e-3)


# ---------------------------------------------- 2. how fast the pooler moves --

def test_score_spread_is_independent_of_the_feature_scale():
    """
    The property the scoring-path `LayerNorm` buys, and the reason it is worth
    a module: the same scoring vector must mean the same pooling on any
    backbone, so that `||w||`'s distance to travel -- and therefore the arm's
    departure from the mean -- is comparable across the ladder. Unnormalised,
    the same vector spans a factor of fifty over these two scales.
    """
    generator = torch.Generator().manual_seed(1)
    direction = torch.randn(D_MODEL, generator=generator)
    direction /= direction.norm()

    spreads = {}
    for scale in (NORMALISED_SCALE, CONV4_SCALE):
        prototyper = S.AttentionPrototyper(D_MODEL)
        with torch.no_grad():
            prototyper.pos_pool.attention[0].weight.copy_(direction.view(1, -1))

        examples, _ = _halves(_examples(scale, batch=64))
        scoring = prototyper.pos_pool.attention[0]
        spreads[scale] = (
            scoring(prototyper.score_norm(examples)).squeeze(-1).std().item()
        )

    ratio = spreads[CONV4_SCALE] / spreads[NORMALISED_SCALE]
    assert 0.9 < ratio < 1.1, spreads


def test_pooled_values_keep_the_backbone_s_magnitude():
    """
    The other half of "on the scoring path only". Normalising the values as
    well would rescale every prototype in the repo, which is not this change's
    business -- `layer_norm_logits` owns what the channel sees, and the
    language model's own weights own the rest.
    """
    samples = _examples(CONV4_SCALE)
    positive, _ = S.AttentionPrototyper(D_MODEL)(samples)
    positive_examples, _ = _halves(samples)

    assert positive.std().item() > 1.0
    assert math.isclose(
        positive.std().item(),
        positive_examples.mean(1).std().item(),
        rel_tol=1e-2,
    )


# ------------------------------------------------------- 3. the diagnostics --

def test_effective_examples_opens_at_the_number_of_examples():
    """
    `1 / sum(p^2)` in examples: uniform pooling over ten of them reads ten, and
    that is the number a fresh run must log. Anything less at epoch zero means
    the pooler did not open at the average.
    """
    prototyper = S.AttentionPrototyper(D_MODEL)
    prototyper(_examples(CONV4_SCALE))

    assert math.isclose(
        prototyper.pool_effective_examples, N_EXAMPLES // 2, rel_tol=1e-3
    )
    assert prototyper.pool_score_norm == 0.0


def test_effective_examples_falls_as_the_pooler_commits():
    """
    The diagnostic has to *move*, or it says nothing about a run. Driving the
    scoring vector up drives the count towards 1 -- one example carrying the
    whole prototype, which is the failure the zero init was introduced to stop
    happening at initialisation.
    """
    prototyper = S.AttentionPrototyper(D_MODEL)
    samples = _examples(NORMALISED_SCALE)

    prototyper(samples)
    opened = prototyper.pool_effective_examples

    generator = torch.Generator().manual_seed(2)
    direction = torch.randn(1, D_MODEL, generator=generator)
    with torch.no_grad():
        prototyper.pos_pool.attention[0].weight.copy_(20.0 * direction)
        prototyper.neg_pool.attention[0].weight.copy_(20.0 * direction)

    prototyper(samples)

    assert opened > prototyper.pool_effective_examples
    assert prototyper.pool_effective_examples < 2.0
    assert prototyper.pool_score_norm > 0.0


def test_average_prototyper_reports_the_same_columns():
    """
    Both arms write the same header, so the ladder's rungs can be read side by
    side. Averaging is uniform pooling, so its effective count is the number of
    examples; it has no scoring vector, so that column is NaN rather than a
    zero that would read as "a pooler that has not moved".
    """
    prototyper = S.AveragePrototyper()
    prototyper(_examples(CONV4_SCALE))

    assert math.isclose(
        prototyper.pool_effective_examples, N_EXAMPLES // 2, rel_tol=1e-9
    )
    assert math.isnan(prototyper.pool_score_norm)


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

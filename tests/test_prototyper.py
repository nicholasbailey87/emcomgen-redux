"""
Tests for the two prototypers in code/models/sender.py.

Runnable without pytest:  python tests/test_prototyper.py

The prototyper is where the referents meet each other and where they are reduced
to the one vector per polarity the language model speaks from.
`AttentionPrototyper` does both: one transformer block over all `2n` referents,
tagged by label, and then a softmax over the examples of each half.

**The pooling opens at the average.** That softmax had the same defect the
softmax over tokens had before `layer_norm_logits`: its input is `w . x`, so both
where it starts and how fast it moves are set by the magnitude of whatever is
upstream. With broccoli's default init the scoring direction is random and the
sharpness of the resulting softmax goes as the between-example standard
deviation of the embeddings, which across the ablation's backbones spans a factor
of fifty -- so a fresh pooler on one backbone would select a single example while
a fresh pooler on another sits within a few percent of the mean. Zero-initialised
scoring weights make the pooling's opening *exactly* the mean. Scoring from
`LayerNorm`ed examples removes the same dependence from the rate of departure;
the pooled *values* stay un-normalised, so the prototype keeps its magnitude.

**The block does not, and that is deliberate.** It is a normally-initialised
DeepNorm residual, so this module opens at a random perturbation of the referents
and then their mean, where until 2026-09-16 the rung opened bit-identically to
its parent. What bought that was `ExampleContrast.contrast_gate`, a lone scalar
at exactly zero with nothing anchoring its sign; the merged module has no gate.
`sender.AttentionPrototyper` carries the argument.

**It cannot read the referent ordering.** Everywhere else in the speaker,
polarity *is* the ordering: the first half of the examples are the positives by
convention. Every positional input to the block is off, so it is
permutation-equivariant and the label tag is the only route by which polarity
reaches it. If a positional embedding is ever added here, the block could infer
polarity without the tag and the tag would stop meaning anything.

**And the shares mean what they say.** A block whose delta is one vector for the
whole game, or one per polarity, can have a large `prototyper_mix_share` while
doing nothing the two separate pools do not already do. Only
`prototyper_within_share` separates that from contrast between examples.

And the diagnostics exist because none of this is visible in accuracy. A pooler
that stayed at the mean and a pooler that found something look identical in every
other column of metrics.csv.
"""

import math

import pytest
import torch

import _bootstrap  # noqa: F401

import models.sender as S


D_MODEL = 64
HEADS = 4
N_EXAMPLES = 20  # ten positive, ten negative, as the speaker is handed them
BATCH = 8

SETTINGS = dict(
    d_model=D_MODEL,
    heads=HEADS,
    ff_inner_size=2 * D_MODEL,
    activation="GELU",
    self_attention_dropout=0.0,
    ff_inner_dropout=0.0,
    ff_outer_dropout=0.0,
)

# The two ends of the ablation's feature scales, from the probe of the eight
#     smoke-test runs: Conv4 is the one sender backbone whose output is not
#     scale-normalised, and a post-norm ViT sits near unity.
CONV4_SCALE = 25.8
NORMALISED_SCALE = 0.45


def _prototyper(seed=0, **overrides):
    torch.manual_seed(seed)
    return S.AttentionPrototyper(**{**SETTINGS, **overrides})


def _flat(prototyper):
    """
    The same module with its block's residual branch zeroed, so the block is the
        identity up to its output norm.

    Reaching into broccoli's block layout, which nothing in `code/` does and
        nothing here should read as licence to: it is the only way to ask "does
        the pooling still open at the mean" separately from "the block moved the
        referents", and the second is what this module deliberately gave up.
    """
    with torch.no_grad():
        for block in prototyper.block.blocks:
            block.beta = 0.0
    return prototyper


def _examples(scale=NORMALISED_SCALE, batch=BATCH, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return scale * torch.randn(batch, N_EXAMPLES, D_MODEL, generator=generator)


def _labels(batch=BATCH):
    """1.0 for the first half, 0.0 for the rest, as `Sender` requires."""
    labels = torch.zeros(batch, N_EXAMPLES)
    labels[:, : N_EXAMPLES // 2] = 1.0
    return labels


def _halves(samples):
    half = samples.size(1) // 2
    return samples[:, :half], samples[:, half:]


def _set_scorer(prototyper, norm, seed=1):
    """
    Point both pools along one random direction of a chosen norm.

    The norm is the knob, not the raw entries: scoring runs off `score_norm`'s
    output, whose projection onto a unit direction is about standard normal, so
    `||w||` *is* the score spread the pools will produce. That is what makes the
    targets below readable as the spreads they are meant to construct.
    """
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(1, prototyper.d_model, generator=generator)
    direction = norm * direction / direction.norm()

    with torch.no_grad():
        prototyper.pos_pool.attention[0].weight.copy_(direction)
        prototyper.neg_pool.attention[0].weight.copy_(direction)


def _direct_score_sd(prototyper, examples):
    """The pre-softmax scores' spread, computed rather than recovered."""
    positive, negative = _halves(examples)
    return torch.cat(
        [
            pool.attention[0](prototyper.score_norm(half)).squeeze(-1)
            for pool, half in (
                (prototyper.pos_pool, positive),
                (prototyper.neg_pool, negative),
            )
        ]
    ).std(-1).mean().item()


# ------------------------------------------------- 1. where the pooler opens --

@pytest.mark.parametrize("scale", [CONV4_SCALE, NORMALISED_SCALE])
def test_the_pooling_opens_at_the_average(scale):
    """
    The whole point of the zero init, asserted on the pooling alone. With the
    block's residual branch zeroed the module *is* its parent rung, on any
    backbone -- and the failure that init replaces is precisely one that only
    shows at one end of the feature scale, so both ends are checked.
    """
    prototyper = _flat(_prototyper())
    samples = _examples(scale)
    mixed = prototyper.block(samples + prototyper.label_embedding[(1.0 - _labels()).long()])
    positive, negative = prototyper(samples, _labels())
    positive_examples, negative_examples = _halves(mixed)

    assert torch.allclose(positive, positive_examples.mean(1), atol=1e-4)
    assert torch.allclose(negative, negative_examples.mean(1), atol=1e-4)


def test_a_flattened_block_reproduces_the_average_prototyper_exactly():
    """
    Stated against the other class rather than against `mean`, since it is the
    ladder's parent rung that has to be reproduced, not an arithmetic identity.

    The block has to be made an identity for this, which is the honest form of
    what used to be free: `AttentionPrototyper` opened at `AveragePrototyper`
    with no help until the contrast stage was merged into it. Here the block is
    zeroed *and* replaced by an identity, so what is left is the claim that the
    pooling and only the pooling is uniform at step 0.
    """
    prototyper = _prototyper()
    prototyper.block = torch.nn.Identity()
    with torch.no_grad():
        prototyper.label_embedding.zero_()

    samples = _examples(CONV4_SCALE)
    pooled = prototyper(samples, _labels())
    averaged = S.AveragePrototyper()(samples)

    for a, b in zip(pooled, averaged):
        assert torch.allclose(a, b, atol=1e-3)


def test_zero_initialised_scoring_weights_still_receive_gradient():
    """
    The objection to a zero init, and why it does not apply. There is one
    output unit, so there is no symmetry between units to break, and
    `dL/dW = sum_i (dL/ds_i) x_i` depends on the examples rather than on `W`.
    """
    prototyper = _prototyper()
    positive, negative = prototyper(_examples(), _labels())
    (positive.sum() + negative.sum()).backward()

    for pool in (prototyper.pos_pool, prototyper.neg_pool):
        assert pool.attention[0].weight.grad.norm().item() > 0.0


def test_the_block_receives_gradient_without_a_gate():
    """
    What replaces `test_the_branch_learns_while_the_gate_is_shut`. There is no
    scalar in front of the block and no `scale_without_attenuating` behind it:
    every parameter is on the path from the first step, which is the whole
    reason the gate could go.
    """
    prototyper = _prototyper()
    positive, negative = prototyper(_examples(), _labels())
    (positive.pow(2).sum() + negative.pow(2).sum()).backward()

    for name, parameter in prototyper.named_parameters():
        assert parameter.grad is not None, name

    assert prototyper.label_embedding.grad.abs().sum().item() > 0.0


def test_reset_parameters_returns_the_pooling_to_the_average():
    """
    A reset speaker must not keep a trained pooling any more than it keeps a
    trained channel: `reset_parameters` has to reimpose the zero, not just
    delegate to broccoli's own init.
    """
    prototyper = _prototyper()
    with torch.no_grad():
        prototyper.pos_pool.attention[0].weight.normal_()
        prototyper.neg_pool.attention[0].weight.normal_()

    prototyper.reset_parameters()

    assert prototyper.pos_pool.attention[0].weight.abs().sum().item() == 0.0
    assert prototyper.neg_pool.attention[0].weight.abs().sum().item() == 0.0

    prototyper(_examples(), _labels())
    assert math.isclose(
        prototyper.pool_effective_examples, N_EXAMPLES // 2, rel_tol=1e-3
    )


# -------------------------------------------------- 2. what the block may see --

def test_the_block_cannot_read_the_referent_ordering():
    """
    Permuting the examples and their labels together permutes the output the
    same way, i.e. the block is equivariant and has no way to tell which half of
    the sequence it is looking at. Every positional argument being off is what
    buys this, and this test is what should fail if a positional embedding is
    ever added.

    Checked on the block rather than on the prototypes, because the pooling is a
    sum over a half and is invariant to a permutation rather than equivariant --
    it would pass this whatever the block did.
    """
    prototyper = _prototyper()
    samples, labels = _examples(), _labels()
    permutation = torch.randperm(
        N_EXAMPLES, generator=torch.Generator().manual_seed(3)
    )

    def mixed(x, y):
        return prototyper.block(x + prototyper.label_embedding[(1.0 - y).long()])

    straight = mixed(samples, labels)[:, permutation]
    permuted = mixed(samples[:, permutation], labels[:, permutation])

    assert torch.allclose(straight, permuted, atol=1e-5)


def test_permuting_within_a_half_leaves_the_prototypes_alone():
    """
    The consequence that matters downstream: the referents of one polarity are a
    set, so reordering them must not change the vector the speaker speaks from.
    Equivariance in the block plus a permutation-invariant pool is what gives
    this, and it is worth pinning as the pair rather than only as the half above.
    """
    prototyper = _prototyper()
    _set_scorer(prototyper, 0.8, seed=7)

    samples, labels = _examples(), _labels()
    half = N_EXAMPLES // 2
    within = torch.cat(
        [
            torch.randperm(half, generator=torch.Generator().manual_seed(4)),
            half + torch.randperm(half, generator=torch.Generator().manual_seed(5)),
        ]
    )

    straight = prototyper(samples, labels)
    shuffled = prototyper(samples[:, within], labels[:, within])

    for a, b in zip(straight, shuffled):
        assert torch.allclose(a, b, atol=1e-5)


def test_the_label_tag_is_load_bearing():
    """
    Since the ordering is unreadable, the tag is the only route polarity has
    into the block: flipping the labels while holding the examples fixed must
    change what comes out. If it does not, polarity is not reaching the block at
    all and the mixing is between examples rather than between classes.
    """
    prototyper = _prototyper()
    samples = _examples()
    flipped = 1.0 - _labels()

    straight = prototyper(samples, _labels())
    swapped = prototyper(samples, flipped)

    assert not torch.allclose(straight[0], swapped[0], atol=1e-5)


def test_the_tag_opens_antipodally_at_unit_scale():
    """
    Row 0 positive, row 1 negative, drawn once and negated -- the same
    initialisation as `SenderTransformerLM.polarity_embedding`, and for the same
    reason: it is added to referents that arrive from a parameter-free norm, so
    unit per-element variance puts it at the scale of what it marks with no
    constant to choose.
    """
    positive, negative = _prototyper().label_embedding

    assert torch.equal(positive, -negative)
    # Three standard errors of the sample standard deviation of `d_model` unit
    #     normals, so this pins the scale without becoming a flaky seed test.
    assert abs(positive.std().item() - 1.0) < 3.0 / math.sqrt(2 * D_MODEL)


def test_the_tag_is_not_named_for_the_speaker_split():
    """
    `SPLIT_LEARNING_RATES` selects parameters by name suffix, so a tag called
    `polarity_embedding` -- or anything ending in it -- would silently join the
    speaker's `polarity_embedding_lr` group. It still has to contain
    "embedding", which is what keeps `gradboard` from decaying it.
    """
    names = [name for name, _ in _prototyper().named_parameters()]

    assert "label_embedding" in names
    assert not any(name.endswith("polarity_embedding") for name in names)
    assert all("embedding" in name for name in names if "label" in name)


# ---------------------------------------------- 3. how fast the pooler moves --

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
        prototyper = _prototyper()
        with torch.no_grad():
            prototyper.pos_pool.attention[0].weight.copy_(direction.view(1, -1))

        examples, _ = _halves(_examples(scale, batch=64))
        scoring = prototyper.pos_pool.attention[0]
        spreads[scale] = (
            scoring(prototyper.score_norm(examples)).squeeze(-1).std().item()
        )

    ratio = spreads[CONV4_SCALE] / spreads[NORMALISED_SCALE]
    assert 0.9 < ratio < 1.1, spreads


def test_pooled_values_keep_the_blocks_magnitude():
    """
    The other half of "on the scoring path only". Normalising the values as
    well would rescale every prototype in the repo, which is not this module's
    business -- `layer_norm_logits` owns what the channel sees, and the
    language model's own weights own the rest.

    Against the block's output rather than the module's input, because the block
    is between them and ends in an `RMSNorm` of its own.
    """
    prototyper = _prototyper()
    samples, labels = _examples(CONV4_SCALE), _labels()
    mixed = prototyper.block(samples + prototyper.label_embedding[(1.0 - labels).long()])

    positive, _ = prototyper(samples, labels)
    positive_examples, _ = _halves(mixed)

    assert math.isclose(
        positive.std().item(),
        positive_examples.mean(1).std().item(),
        rel_tol=1e-2,
    )


# ------------------------------------------------------- 4. the diagnostics --

def test_effective_examples_opens_at_the_number_of_examples():
    """
    `1 / sum(p^2)` in examples: uniform pooling over ten of them reads ten, and
    that is the number a fresh run must log. Anything less at epoch zero means
    the pooling did not open at the average.
    """
    prototyper = _prototyper()
    prototyper(_examples(CONV4_SCALE), _labels())

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
    prototyper = _prototyper()
    samples, labels = _examples(), _labels()

    prototyper(samples, labels)
    opened = prototyper.pool_effective_examples

    generator = torch.Generator().manual_seed(2)
    direction = torch.randn(1, D_MODEL, generator=generator)
    with torch.no_grad():
        prototyper.pos_pool.attention[0].weight.copy_(20.0 * direction)
        prototyper.neg_pool.attention[0].weight.copy_(20.0 * direction)

    prototyper(samples, labels)

    assert opened > prototyper.pool_effective_examples
    assert prototyper.pool_effective_examples < 2.0
    assert prototyper.pool_score_norm > 0.0


def test_pool_score_sd_is_the_scores_own_spread():
    """
    `pool_score_sd` is recovered from the weights rather than recomputed, on the
    identity `log w = s - logsumexp(s)`: an additive constant per game, so the
    standard deviation over examples is the pre-softmax scores' exactly. Checked
    against the scores computed directly, because if that identity were wrong
    the column would be silently wrong too and nothing else would notice.
    """
    prototyper = _prototyper()
    samples, labels = _examples(), _labels()
    mixed = prototyper.block(samples + prototyper.label_embedding[(1.0 - labels).long()])

    _set_scorer(prototyper, 0.8, seed=3)
    prototyper(samples, labels)

    assert prototyper.pool_score_sd == pytest.approx(
        _direct_score_sd(prototyper, mixed), rel=1e-4
    )

    # The recovery is exact only while the softmax has not underflowed. Past a
    # gap of about 87 nats a loser's weight is 0 in fp32 and the unclamped log
    # would give NaN -- which reads as "no pooler at all" rather than as the
    # total commitment it is, so the floor has to hold.
    _set_scorer(prototyper, 300.0, seed=3)
    prototyper(samples, labels)

    assert math.isfinite(prototyper.pool_score_sd)
    assert prototyper.pool_score_sd > 10.0
    assert prototyper.pool_effective_examples < 1.1


def test_pool_score_sd_resolves_where_the_effective_count_cannot():
    """
    The reason for the column. `1 / sum(p^2)` is about `n / (1 + sd^2)`, so near
    its ceiling it compresses brutally: the 2026-08-29 ShapeWorld run read
    9.86323 and 9.99613 in consecutive dead stretches, a 1.3% difference in the
    count that was sixfold in the spread -- and the difference between a
    collapse the speaker climbed out of and one it did not.

    Constructed here at those two spreads, and the assertion is that the counts
    agree to about a percent while the sd column separates them by six.
    """
    prototyper = _prototyper()
    samples, labels = _examples(seed=4), _labels()

    readings = {}
    for name, spread in (("faint", 0.118), ("fainter", 0.020)):
        _set_scorer(prototyper, spread, seed=5)
        prototyper(samples, labels)
        readings[name] = (
            prototyper.pool_effective_examples, prototyper.pool_score_sd
        )

    (count_faint, sd_faint), (count_fainter, sd_fainter) = (
        readings["faint"], readings["fainter"]
    )

    # Both read as "uniform pooling" in the count, within a percent of each
    # other and of the ten-example ceiling.
    assert count_faint / count_fainter > 0.98
    assert count_faint > 0.98 * (N_EXAMPLES // 2)

    # The sd column separates them by the sixfold that is actually there.
    assert sd_faint / sd_fainter > 4.0


def test_the_mix_share_reads_the_volume_of_the_blocks_delta():
    """
    Volume, and only volume. Scaling the block's branch by a factor scales the
    delta by it, so the column moves with the factor and the shape column below
    does not.
    """
    prototyper = _prototyper()
    samples, labels = _examples(), _labels()

    prototyper(samples, labels)
    full = prototyper.prototyper_mix_share

    with torch.no_grad():
        for block in prototyper.block.blocks:
            block.beta = block.beta / 2.0
    prototyper(samples, labels)

    assert full > 0.0
    assert prototyper.prototyper_mix_share < full


def test_the_mix_share_is_zero_for_a_block_that_does_nothing():
    """
    A flattened branch leaves the delta at exactly the post-norm's own
    adjustment, which is what a block contributing nothing looks like -- a
    different row from `AveragePrototyper`'s NaN, which is no block at all.
    """
    prototyper = _flat(_prototyper())
    prototyper.block = torch.nn.Identity()
    prototyper(_examples(), _labels())

    assert prototyper.prototyper_mix_share == 0.0
    assert prototyper.prototyper_within_share == 0.0


@pytest.mark.parametrize("constant", ["game", "polarity"])
def test_within_share_sees_through_a_delta_that_is_not_example_level(constant):
    """
    The failure this column exists to catch. A block whose delta is one vector
    for the whole game shifts both prototypes equally and the language model's
    `LayerNorm` eats most of it; one vector per polarity is a learned "I am
    positive", which the two separate pools already provide. Both can carry a
    large `prototyper_mix_share`, and both must read as approximately no
    example-level mixing at all.
    """
    prototyper = _prototyper()
    generator = torch.Generator().manual_seed(11)

    half = N_EXAMPLES // 2
    if constant == "game":
        delta = torch.randn(BATCH, 1, D_MODEL, generator=generator).expand(
            BATCH, N_EXAMPLES, D_MODEL
        )
    else:
        per_polarity = torch.randn(BATCH, 2, D_MODEL, generator=generator)
        delta = torch.cat(
            (
                per_polarity[:, :1].expand(BATCH, half, D_MODEL),
                per_polarity[:, 1:].expand(BATCH, half, D_MODEL),
            ),
            dim=1,
        )

    samples = _examples()
    weights = torch.full((BATCH, half), 1.0 / half)
    prototyper._record_diagnostics(
        samples, samples, samples + delta, weights, weights
    )

    assert prototyper.prototyper_within_share == pytest.approx(0.0, abs=1e-6)


def test_within_share_is_one_for_a_delta_with_no_shared_component():
    """
    The other end: a delta whose per-polarity means are zero is all
    example-level, so the column reads 1.0. Together with the case above this
    pins the decomposition as a share of the total rather than an arbitrary
    ratio.
    """
    prototyper = _prototyper()
    generator = torch.Generator().manual_seed(12)

    delta = torch.randn(BATCH, N_EXAMPLES, D_MODEL, generator=generator)
    half = N_EXAMPLES // 2
    delta[:, :half] -= delta[:, :half].mean(1, keepdim=True)
    delta[:, half:] -= delta[:, half:].mean(1, keepdim=True)

    samples = _examples()
    weights = torch.full((BATCH, half), 1.0 / half)
    prototyper._record_diagnostics(
        samples, samples, samples + delta, weights, weights
    )

    assert prototyper.prototyper_within_share == pytest.approx(1.0, abs=1e-6)


def test_both_share_columns_are_written_on_a_train_pass_and_are_finite():
    """
    The columns reach metrics.csv off this module directly, with no `hasattr`
    guard in `train.py`, so a fresh module that wrote NaN would put NaN in the
    header's place for the whole run.
    """
    prototyper = _prototyper().train()
    prototyper(_examples(), _labels())

    for column in ("prototyper_mix_share", "prototyper_within_share"):
        value = getattr(prototyper, column)
        assert math.isfinite(value), column
        assert value > 0.0, column


def test_the_spreads_bracket_the_block():
    """
    `referent_spread_backbone` is what the interface handed over and
    `referent_spread` is what the pooling sees, so the two differ exactly
    because the block ran. They live here rather than on `Sender` for that
    reason: with the block inside this module, nothing outside it can see
    between the mixing and the pooling.
    """
    prototyper = _prototyper()

    assert math.isnan(prototyper.referent_spread)
    assert math.isnan(prototyper.referent_spread_backbone)

    prototyper.train()
    prototyper(_examples(), _labels())

    assert prototyper.referent_spread > 0.0
    assert prototyper.referent_spread_backbone > 0.0
    assert prototyper.referent_spread != pytest.approx(
        prototyper.referent_spread_backbone, rel=1e-3
    )


def test_the_spreads_are_not_written_on_an_eval_pass():
    """
    NaN until a train pass, like every other speaker diagnostic, and never
    written on eval -- which samples nothing and so measures nothing.
    """
    prototyper = _prototyper().train()
    prototyper(_examples(seed=5), _labels())
    trained = (prototyper.referent_spread, prototyper.referent_spread_backbone)

    prototyper.eval()
    with torch.no_grad():
        prototyper(_examples(seed=6), _labels())

    assert (
        prototyper.referent_spread, prototyper.referent_spread_backbone
    ) == trained


def test_average_prototyper_reports_the_same_columns():
    """
    Both arms write the same header, so the ladder's rungs can be read side by
    side. Averaging is uniform pooling, so its effective count is the number of
    examples; it has no scoring vector, so that column is NaN rather than a
    zero that would read as "a pooler that has not moved"; and it has no block,
    so both share columns are NaN and the two spreads are equal.
    """
    prototyper = S.AveragePrototyper().train()
    prototyper(_examples(CONV4_SCALE), _labels())

    assert math.isclose(
        prototyper.pool_effective_examples, N_EXAMPLES // 2, rel_tol=1e-9
    )
    assert math.isnan(prototyper.pool_score_norm)
    assert math.isnan(prototyper.pool_score_sd)
    assert math.isnan(prototyper.prototyper_mix_share)
    assert math.isnan(prototyper.prototyper_within_share)
    assert prototyper.referent_spread == prototyper.referent_spread_backbone
    assert prototyper.referent_spread > 0.0


# ----------------------------------------------- 5. the statistic, on its own --

def test_referent_spread_is_the_within_polarity_residual_over_the_means():
    """
    The formula, against an independently computed expectation, because the
    properties below could each hold of a wrong one.

    Pins the polarity handling in particular: positives differ from negatives by
    construction and that is the concept, not a failure to collapse, so anything
    constant *within* a polarity has to leave the numerator. The prototyper pools
    each half separately and never sees the gap between them.
    """
    embedded = _examples(seed=3)
    positive, negative = _halves(embedded)
    means = torch.cat(
        (
            positive.mean(1, keepdim=True).expand_as(positive),
            negative.mean(1, keepdim=True).expand_as(negative),
        ),
        dim=1,
    )
    expected = (
        (embedded - means).pow(2).mean().sqrt() / means.pow(2).mean().sqrt()
    ).item()

    assert S.referent_spread(embedded) == pytest.approx(expected, rel=1e-5)


def test_referent_spread_reads_zero_when_the_referents_collapse():
    """
    The collapse the column exists to name. Every example the same vector means
    nothing to pool, nothing to mix and nothing to say.
    """
    one = torch.randn(1, 1, D_MODEL)
    collapsed = S.referent_spread(one.expand(BATCH, N_EXAMPLES, D_MODEL).contiguous())

    # Not bit-exact: subtracting a mean from the values it was computed over
    # leaves float residue, and it is ~1e-8 against live readings of order 1.
    assert collapsed == pytest.approx(0.0, abs=1e-6)


def test_referent_spread_is_unmoved_by_a_global_rescale():
    """
    A ratio, so the backbone's own magnitude cancels -- which is what lets the
    column be compared across arms whose feature scales differ by fifty.
    """
    embedded = _examples(seed=2)
    reference = S.referent_spread(embedded)

    for scale in (1e-3, 7.0, 1e3):
        assert S.referent_spread(scale * embedded) == pytest.approx(
            reference, rel=1e-4
        )


def test_referent_spread_falls_when_a_common_vector_grows():
    """
    The second reading, and the reason the pair brackets the block: a block
    whose delta is one vector for the whole game raises the mix share while
    adding nothing, and it shows up here as the referents being drowned in what
    they share.
    """
    embedded = _examples(seed=4)
    common = torch.randn(1, 1, D_MODEL)

    quiet = S.referent_spread(embedded + 20.0 * common)
    loud = S.referent_spread(embedded)

    assert quiet < loud / 10.0


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

"""
Tests for `ExampleContrast` in code/models/sender.py.

Runnable without pytest:  python tests/test_contrast.py

The contrast stage runs one self-attention over all the referents -- positive
and negative together -- and adds the result back as a residual, so a positive
example can be represented by what distinguishes it from the negatives rather
than by what it is alone. Four things about it are load-bearing and none of them
is visible in accuracy:

**It opens at the identity.** `contrast_gate` starts at zero, so `contrast =
true` is an ablation of exactly one thing and the rungs stay reproducible with
the flag on. Any drift here turns the arm into "the parent rung, plus a
different opening", which is what rung 3 was before `AttentionPrototyper`'s
scoring weights were zeroed.

**And it can leave.** Opening at zero is only safe if zero is a starting point
rather than a floor to be welded to. It is a gate on a residual and not a
`clamp`, so the gradient there is non-zero -- the same distinction
`AttentionDiscriminator.mix_floor` turns on.

**It cannot read the referent ordering.** Everywhere else in the speaker,
polarity *is* the ordering: the first half of the examples are the positives by
convention. With `rotary_embedding=None` this attention is permutation-
equivariant, so the label tag is the only route by which polarity reaches it. If
a positional embedding is ever added here, the stage could infer polarity
without the tag and the tag would stop meaning anything.

**And the shares mean what they say.** A branch that emits one vector for the
whole game, or one per polarity, can have a large `contrast_share` while doing
nothing that the prototyper's own machinery does not already do. Only
`contrast_within_share` separates that from contrast between examples.
"""

import math

import pytest
import torch

import _bootstrap  # noqa: F401

import models.sender as S


FEAT = 64
D_MODEL = 32
HEADS = 4
N_EXAMPLES = 20  # ten positive, ten negative, as the speaker is handed them
BATCH = 8

SETTINGS = dict(d_model=D_MODEL, heads=HEADS, self_attention_dropout=0.0)


def _contrast(seed=0):
    torch.manual_seed(seed)
    return S.ExampleContrast(FEAT, **SETTINGS)


def _examples(batch=BATCH, seed=0, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return scale * torch.randn(batch, N_EXAMPLES, FEAT, generator=generator)


def _labels(batch=BATCH):
    """1.0 for the first half, 0.0 for the rest, as `Sender` requires."""
    labels = torch.zeros(batch, N_EXAMPLES)
    labels[:, : N_EXAMPLES // 2] = 1.0
    return labels


def test_opens_at_the_identity():
    """
    With the gate shut the referents reach the prototyper untouched -- bitwise,
    not approximately, because the arm is only an ablation of one thing if the
    two speakers start from the same numbers.
    """
    contrast = _contrast()
    samples = _examples()

    assert torch.equal(contrast(samples, _labels()), samples)


def test_the_gate_is_a_starting_point_and_not_a_weld():
    """
    A zero opening is only safe if the gradient at zero is non-zero:
    `dL/dgate = <branch, dL/dout>`. A `clamp` at zero would have no gradient
    there and the stage would never open at all.
    """
    contrast = _contrast()
    contrast(_examples(), _labels()).pow(2).sum().backward()

    assert contrast.contrast_gate.grad is not None
    assert contrast.contrast_gate.grad.abs().item() > 0.0


def test_the_stage_cannot_read_the_referent_ordering():
    """
    Permuting the examples and their labels together permutes the output the
    same way, i.e. the stage is equivariant and has no way to tell which half of
    the sequence it is looking at. `rotary_embedding=None` is what buys this,
    and this test is what should fail if a positional embedding is ever added.
    """
    contrast = _contrast()
    with torch.no_grad():
        contrast.contrast_gate.fill_(1.0)

    samples, labels = _examples(), _labels()
    permutation = torch.randperm(N_EXAMPLES, generator=torch.Generator().manual_seed(3))

    straight = contrast(samples, labels)[:, permutation]
    permuted = contrast(samples[:, permutation], labels[:, permutation])

    assert torch.allclose(straight, permuted, atol=1e-6)


def test_the_label_tag_is_load_bearing():
    """
    Since the ordering is unreadable, the tag is the only route polarity has
    into this stage: flipping the labels while holding the examples fixed must
    change what comes out. If it does not, polarity is not reaching the stage at
    all and the contrast is between examples rather than between classes.
    """
    contrast = _contrast()
    with torch.no_grad():
        contrast.contrast_gate.fill_(1.0)

    samples = _examples()
    flipped = 1.0 - _labels()

    assert not torch.allclose(
        contrast(samples, _labels()), contrast(samples, flipped), atol=1e-6
    )


def test_share_is_zero_while_the_gate_is_shut():
    """
    `contrast_share` measures the *gated* contribution, so it reads zero on a
    stage that is switched on but has not opened -- which is a different row
    from a stage that is not there, where it is NaN.
    """
    contrast = _contrast()
    contrast(_examples(), _labels())

    assert contrast.contrast_share == 0.0


def test_share_tracks_the_gate():
    """
    Volume, and only volume: the branch is unchanged, so opening the gate by a
    factor scales the share by the same factor.
    """
    contrast = _contrast()
    samples, labels = _examples(), _labels()

    with torch.no_grad():
        contrast.contrast_gate.fill_(0.1)
    contrast(samples, labels)
    small = contrast.contrast_share

    with torch.no_grad():
        contrast.contrast_gate.fill_(0.2)
    contrast(samples, labels)

    assert small > 0.0
    assert math.isclose(contrast.contrast_share, 2.0 * small, rel_tol=1e-5)


def test_within_share_is_measured_before_the_gate():
    """
    Shape and volume kept apart, as `logit_spread` and `logit_scale` are on the
    speaker: a well-shaped branch that is still quiet must read as well-shaped,
    or the column could never be interpreted early in a run -- which is the only
    time anyone needs it.
    """
    contrast = _contrast()
    samples, labels = _examples(), _labels()

    contrast(samples, labels)
    shut = contrast.contrast_within_share

    with torch.no_grad():
        contrast.contrast_gate.fill_(0.7)
    contrast(samples, labels)

    assert shut > 0.0
    assert math.isclose(contrast.contrast_within_share, shut, rel_tol=1e-6)


@pytest.mark.parametrize("constant", ["game", "polarity"])
def test_within_share_sees_through_a_branch_that_is_not_example_level(constant):
    """
    The failure this column exists to catch. A branch emitting one vector for
    the whole game shifts both prototypes equally and the language model's
    `LayerNorm` eats most of it; a branch emitting one per polarity is a learned
    "I am positive", which `AttentionPrototyper`'s two separate pools already
    provide. Both can carry a large `contrast_share`, and both must read as
    approximately no example-level contrast at all.
    """
    contrast = _contrast()
    generator = torch.Generator().manual_seed(11)

    half = N_EXAMPLES // 2
    if constant == "game":
        branch = torch.randn(BATCH, 1, FEAT, generator=generator).expand(
            BATCH, N_EXAMPLES, FEAT
        )
    else:
        per_polarity = torch.randn(BATCH, 2, FEAT, generator=generator)
        branch = torch.cat(
            (
                per_polarity[:, :1].expand(BATCH, half, FEAT),
                per_polarity[:, 1:].expand(BATCH, half, FEAT),
            ),
            dim=1,
        )

    contrast._record_diagnostics(_examples(), branch, branch)

    assert contrast.contrast_within_share == pytest.approx(0.0, abs=1e-6)


def test_within_share_is_one_for_a_branch_with_no_shared_component():
    """
    The other end: a branch whose per-polarity means are zero is all
    example-level, so the column reads 1.0. Together with the case above this
    pins the decomposition as a share of the total rather than an arbitrary
    ratio.
    """
    contrast = _contrast()
    generator = torch.Generator().manual_seed(12)

    branch = torch.randn(BATCH, N_EXAMPLES, FEAT, generator=generator)
    half = N_EXAMPLES // 2
    branch[:, :half] -= branch[:, :half].mean(1, keepdim=True)
    branch[:, half:] -= branch[:, half:].mean(1, keepdim=True)

    contrast._record_diagnostics(_examples(), branch, branch)

    assert contrast.contrast_within_share == pytest.approx(1.0, abs=1e-6)


def test_reset_restores_the_opening():
    """
    A reset speaker must open at the identity again. The gate is the whole of
    that guarantee, and it is the one parameter here that is not restored by
    some submodule's own `reset_parameters`.
    """
    contrast = _contrast()
    with torch.no_grad():
        contrast.contrast_gate.fill_(0.9)

    contrast.reset_parameters()

    assert contrast.contrast_gate.item() == 0.0
    assert torch.equal(contrast(_examples(), _labels()), _examples())


def test_the_tag_opens_antipodally_at_unit_scale():
    """
    Row 0 positive, row 1 negative, drawn once and negated -- the same
    initialisation as `SenderTransformerLM.polarity_embedding`, and for the same
    reason: it is added after a parameter-free norm, so unit per-element
    variance puts it at the scale of what it marks with no constant to choose.
    """
    contrast = _contrast()
    positive, negative = contrast.label_embedding

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
    names = [name for name, _ in _contrast().named_parameters()]

    assert "label_embedding" in names
    assert not any(name.endswith("polarity_embedding") for name in names)
    assert all("embedding" in name for name in names if "label" in name)


def test_the_speaker_composes_it_with_either_prototyper():
    """
    The stage returns the backbone's own width, so the prototyper downstream
    neither knows nor cares that it ran -- which is the whole point of putting
    it on the residual.
    """
    samples = _examples()
    labels = _labels()
    contrast = _contrast()
    with torch.no_grad():
        contrast.contrast_gate.fill_(0.5)

    contrasted = contrast(samples, labels)
    assert contrasted.shape == samples.shape

    for prototyper in (S.AveragePrototyper(), S.AttentionPrototyper(FEAT)):
        positive, negative = prototyper(contrasted, labels)
        assert positive.shape == (BATCH, FEAT)
        assert negative.shape == (BATCH, FEAT)


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

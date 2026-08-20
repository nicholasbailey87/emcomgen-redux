"""
Tests for DeepNorm residual scaling, resolved from each stack's own depth.

Runnable without pytest:  python tests/test_residual_scaling.py

`alpha` scales the skip connection and `beta` the branch, and broccoli applies
both at every post-norm block. Left at 1.0 the stacks are vanilla post-LN, which
is the configuration a 4-to-10 layer post-norm transformer is least likely to
train in -- and 1.0 is nowhere near the constants those depths ask for
(1.68/0.42 at four layers, 2.06/0.34 at nine).

Two constants per stack, restated in every config that changes `layers`, is a
standing invitation to leave them behind at values derived for a different
depth. So `"deepnorm"` resolves them at construction from the depth the stack
was actually built with, and these tests pin the resolution rather than the
numbers in DEFAULT.toml: what has to hold is that a stack's scaling follows its
own `layers`, whatever a config says those are.

The encoder constants everywhere, including for the two stacks that have a
cross-attention near them. DeepNorm's decoder form assumes cross-attention
inside every block; here it runs once, outside the residual path whose depth is
being corrected for -- `SenderTransformerLM` uses it to build the sequence its
encoder reads, and `TransformerCrossAttentionComparer` puts it between two
encoder stacks.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import models.model_util as model_util  # noqa: E402
import models.receiver as R  # noqa: E402
import models.sender as S  # noqa: E402
from models.backbone.vision import ViT2  # noqa: E402
from parse_config import get_config  # noqa: E402


def _settings(section, **overrides):
    settings = dict(get_config()[section])
    settings.update(overrides)
    return settings


# ------------------------------------------------------- 1. the arithmetic --

@pytest.mark.parametrize(
    "layers,alpha,beta",
    [(4, 1.682, 0.420), (5, 1.778, 0.398), (9, 2.060, 0.343), (10, 2.115, 0.334)],
)
def test_deepnorm_constants_match_the_paper(layers, alpha, beta):
    """`alpha = (2N)^(1/4)`, `beta = (8N)^(-1/4)`, Wang et al. (2022)."""
    resolved_alpha, resolved_beta = model_util.deepnorm_constants(layers)

    assert abs(resolved_alpha - alpha) < 5e-3
    assert abs(resolved_beta - beta) < 5e-3


def test_a_number_pins_the_scaling():
    """
    The ablation may want to hold these fixed across depths deliberately, so a
    configured number has to survive untouched -- including 1.0, which is the
    behaviour every run before this change had.
    """
    assert model_util.resolve_residual_scaling(1.0, 1.0, 9) == (1.0, 1.0)
    assert model_util.resolve_residual_scaling(2.5, 0.5, 4) == (2.5, 0.5)


def test_one_may_be_pinned_while_the_other_derives():
    derived = model_util.deepnorm_constants(4)[1]
    alpha, beta = model_util.resolve_residual_scaling(1.0, "deepnorm", 4)

    assert alpha == 1.0
    assert beta == derived


def test_a_misspelled_sentinel_is_rejected_at_construction():
    """
    Silence here would be expensive: the string would reach broccoli and
    multiply a tensor, so the run would fail somewhere inside a forward pass,
    or -- worse -- somewhere it is caught and ignored.
    """
    with pytest.raises(ValueError, match="alpha"):
        model_util.resolve_residual_scaling("deepnrom", 1.0, 4)

    with pytest.raises(ValueError, match="beta"):
        model_util.resolve_residual_scaling(1.0, "deep-norm", 4)


def test_a_stack_with_no_blocks_cannot_be_derived_from():
    with pytest.raises(ValueError, match="at least one layer"):
        model_util.deepnorm_constants(0)


# ----------------------------------------------------------- 2. the stacks --

@pytest.mark.parametrize("layers", [2, 4])
def test_vit_resolves_from_its_own_depth(layers):
    backbone = ViT2(
        n_feats=(3, 64, 64),
        **_settings("sender_feature_model", d_model=64, heads=2, layers=layers),
    )
    alpha, beta = model_util.deepnorm_constants(layers)

    assert (backbone.alpha, backbone.beta) == (alpha, beta)


@pytest.mark.parametrize("layers", [1, 4])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_sender_transformer_lm_resolves_from_its_own_depth(layers, bidirectional):
    """
    Depth *and* form. The two arms have different numbers of residual branches
    per block -- the decoder arm cross-attends into the latent memory inside
    every one of them, the parallel arm does not -- and DeepNorm derives
    different constants for the two cases. Reading the encoder constants for a
    three-branch block would open the stack scaled for a residual path shorter
    than the one it has.
    """
    language_model = S.SenderTransformerLM(
        64,
        **_settings(
            "sender_language_model",
            d_model=64,
            token_embedding_size=64,
            heads=4,
            layers=layers,
            bidirectional=bidirectional,
        ),
    )
    alpha, beta = model_util.deepnorm_constants(
        layers, decoder=not bidirectional
    )

    assert (language_model.alpha, language_model.beta) == (alpha, beta)


def test_the_two_deepnorm_forms_are_not_the_same():
    """
    Guards the test above against passing vacuously if the `decoder` flag ever
    stopped doing anything.
    """
    for layers in (1, 4, 10):
        assert model_util.deepnorm_constants(layers) != model_util.deepnorm_constants(
            layers, decoder=True
        )


@pytest.mark.parametrize("layers", [1, 2, 4, 7])
def test_the_comparer_resolves_its_encoder_from_the_configured_depth(layers):
    """
    `layers` is the message encoder's depth and nothing else's, so it derives
    from the whole of it. It used to be a total split between a reading stack
    and a fusion stack, which meant asking for one more block moved two.
    """
    comparer = R.TransformerCrossAttentionComparer(
        64, **_settings("receiver_comparer", d_model=64, heads=4, layers=layers)
    )

    assert (comparer.encoding_alpha, comparer.encoding_beta) == (
        model_util.deepnorm_constants(layers)
    )


@pytest.mark.parametrize("layers", [1, 4, 10])
def test_the_hand_written_residuals_resolve_at_depth_one(layers):
    """
    The three attention stages around the encoder are bare sublayers, not
    `EncoderBlock`s, and DeepNorm's depth argument counts attention-plus-
    feedforward layers. Two attention sublayers with no feedforward are one
    layer's worth of residual path, so these do not move with `layers` -- and
    if they ever start to, the encoder's depth would be silently rescaling
    residuals it is not on.
    """
    comparer = R.TransformerCrossAttentionComparer(
        64, **_settings("receiver_comparer", d_model=64, heads=4, layers=layers)
    )

    assert (comparer.residual_alpha, comparer.residual_beta) == (
        model_util.deepnorm_constants(1)
    )


def test_a_pinned_number_reaches_both_residual_groups():
    """
    Pinning is documented as passing straight through, and there are now two
    places for it to pass through to.
    """
    comparer = R.TransformerCrossAttentionComparer(
        64,
        **_settings(
            "receiver_comparer", d_model=64, heads=4, layers=4, alpha=2.0, beta=0.25
        ),
    )

    assert (comparer.encoding_alpha, comparer.encoding_beta) == (2.0, 0.25)
    assert (comparer.residual_alpha, comparer.residual_beta) == (2.0, 0.25)


def test_the_comparer_still_runs_at_its_resolved_scaling():
    """
    Construction is not the risk on its own -- these multiply tensors inside
    every block, so a resolved value has to survive a forward pass.
    """
    settings = _settings("receiver_comparer", d_model=64, heads=4, layers=4)
    comparer = R.TransformerCrossAttentionComparer(32, **settings)
    scores = comparer(
        torch.randn(2, 6, 32),
        torch.randn(2, settings["message_length"], settings["token_embedding_size"]),
    )

    assert scores.shape == (2, 6)
    assert torch.isfinite(scores).all()


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
        for arguments in ([tuple(b) if isinstance(b, tuple) else (b,) for b in builds] or [()]):
            fn(*arguments)
            passed += 1
        print(f"ok  {fn.__name__}")
    print(f"\n{passed} tests passed")

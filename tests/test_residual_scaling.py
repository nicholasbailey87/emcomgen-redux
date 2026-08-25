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

The encoder constants everywhere, including for the stacks that have a
cross-attention near them. DeepNorm's decoder form assumes cross-attention
inside every block; where it runs once, outside the residual path whose depth is
being corrected for, the encoder form is the right one -- `SenderTransformerLM`
uses it to build the sequence its encoder reads. The listener's two stacks are
built from `DecoderBlock` and do take the decoder form; see
`test_the_listener_asks_for_the_decoder_form`.
"""


import pytest
import torch

import _bootstrap  # noqa: F401
from _bootstrap import build_listener, config_section

import models.model_util as model_util
import models.sender as S
from models.backbone.vision import ViT2


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
        **config_section("sender_feature_model", d_model=64, heads=2, layers=layers),
    )
    alpha, beta = model_util.deepnorm_constants(layers)

    assert (backbone.alpha, backbone.beta) == (alpha, beta)


@pytest.mark.parametrize("layers", [1, 4])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_sender_transformer_lm_resolves_from_its_own_depth(layers, bidirectional):
    """
    Depth, and the encoder form on both arms.

    The arms used to take different DeepNorm constants: the causal arm was a
    `TransformerDecoder` cross-attending into a latent memory inside every block,
    so three residual branches, against the parallel arm's two. Neither arm
    cross-attends from inside a block any more -- the referents arrive as the
    stack's input -- so both are two-branch stacks and both take `decoder=False`.

    Reading the decoder constants for a two-branch block would open the stack
    scaled for a residual path longer than the one it has. The listener still has
    genuine three-branch stacks; `test_the_listener_asks_for_the_decoder_form`
    covers those.
    """
    language_model = S.SenderTransformerLM(
        64,
        **config_section(
            "sender_language_model",
            d_model=64,
            token_embedding_size=64,
            heads=4,
            layers=layers,
            bidirectional=bidirectional,
        ),
    )
    alpha, beta = model_util.deepnorm_constants(layers, decoder=False)

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


def _listener(referent_dim=64, **overrides):
    """
    The attention arm, with each slot's `layers` and residual keys settable
        separately -- which is the whole subject of this section. Keys are
        routed by name because both tables carry `alpha`, `beta` and `layers`
        and mean different stacks by them.
    """
    return build_listener(
        "ReceiverCrossAttentionLM",
        "AttentionDiscriminator",
        referent_dim,
        language_model_overrides=dict(
            d_model=64, heads=4,
            layers=overrides.get("message_layers", 1),
            **{
                key: overrides[key]
                for key in ("alpha", "beta")
                if key in overrides
            },
        ),
        discriminator_overrides=dict(
            d_model=64, heads=4,
            layers=overrides.get("referent_layers", 1),
            **{
                key: overrides[key]
                for key in ("alpha", "beta")
                if key in overrides
            },
        ),
    )


@pytest.mark.parametrize("layers", [1, 2, 4, 7])
def test_the_listener_resolves_each_stack_from_its_own_depth(layers):
    """
    Two stacks, two depths, two pairs. The depth key of one must not reach the
    other's scaling -- there was once a single key that was a total split
    between two stacks, so asking for one more block moved two. They are now in
    separate config tables, which makes the mistake unstateable rather than
    merely untested; the test stays because the tables could be merged again.
    """
    listener = _listener(message_layers=layers, referent_layers=1)
    language_model = listener.language_model
    discriminator = listener.discriminator

    assert (language_model.alpha, language_model.beta) == (
        model_util.deepnorm_constants(layers, decoder=True)
    )
    assert (discriminator.alpha, discriminator.beta) == (
        model_util.deepnorm_constants(1, decoder=True)
    )


@pytest.mark.parametrize("layers", [1, 4, 10])
def test_the_listener_asks_for_the_decoder_form(layers):
    """
    Both stacks are built from `DecoderBlock`, which has three residual branches
    to a block rather than two, so they take `(3N)^0.25` and `(12N)^-0.25`. The
    encoder form would scale their branches as if a block held two sublayers,
    which is the wrong constant by a factor that grows with depth.
    """
    listener = _listener(message_layers=layers, referent_layers=layers)
    language_model = listener.language_model

    assert (language_model.alpha, language_model.beta) != (
        model_util.deepnorm_constants(layers, decoder=False)
    )
    assert language_model.alpha == pytest.approx((3 * layers) ** 0.25)
    assert language_model.beta == pytest.approx((12 * layers) ** -0.25)


def test_a_pinned_number_reaches_both_stacks():
    """
    Pinning is documented as passing straight through, and there are two places
    for it to pass through to.
    """
    listener = _listener(
        message_layers=4, referent_layers=2, alpha=2.0, beta=0.25
    )

    assert (listener.language_model.alpha, listener.language_model.beta) == (
        2.0, 0.25
    )
    assert (listener.discriminator.alpha, listener.discriminator.beta) == (
        2.0, 0.25
    )


def test_the_listener_still_runs_at_its_resolved_scaling():
    """
    Construction is not the risk on its own -- these multiply tensors inside
    every block, so a resolved value has to survive a forward pass.
    """
    listener = _listener(referent_dim=32, message_layers=2, referent_layers=2)
    scores = listener(
        torch.randn(2, 6, 32),
        torch.randn(
            2, listener.message_length, listener.token_embedding_size
        ),
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

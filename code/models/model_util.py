"""
Model building utils
"""

from broccoli.activation import ReLU, GELU, SquaredReLU, SwiGLU


# Name -> broccoli activation, so `activation` can be set from a TOML config.
#     `TransformerEncoder` takes the class directly, so the lookup happens here.
ACTIVATIONS = {
    "ReLU": ReLU,
    "GELU": GELU,
    "SquaredReLU": SquaredReLU,
    "SwiGLU": SwiGLU,
}


def get_activation(name):
    """Look up a broccoli activation by name. See docs/broccoli.md."""
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Valid options: {', '.join(sorted(ACTIVATIONS))}."
        )


# Sentinel for `alpha`/`beta`: resolve them from the stack's own depth. Either
#     may be given as a number instead, which pins it. See docs/broccoli.md.
DEEPNORM = "deepnorm"


def deepnorm_constants(layers, decoder=False):
    """
    DeepNorm's residual scaling for a post-norm stack `layers` deep (Wang et al.
        2022, https://arxiv.org/abs/2203.00555).

    `decoder` selects the three-branch form, i.e. a cross-attention sublayer
        inside every block. See docs/broccoli.md for which stacks are which.

    Returns:
        (alpha, beta)
    """
    if layers < 1:
        raise ValueError(
            f"DeepNorm constants need at least one layer, got {layers}. A stack "
            f"with no blocks has no residual path to scale, so pin `alpha` and "
            f"`beta` to 1.0 instead."
        )

    if decoder:
        return (3.0 * layers) ** 0.25, (12.0 * layers) ** -0.25

    return (2.0 * layers) ** 0.25, (8.0 * layers) ** -0.25


def resolve_residual_scaling(alpha, beta, layers, decoder=False):
    """
    Resolve the configured `alpha` and `beta` against a stack's depth.

    Each is either a number, passed through untouched, or the string `DEEPNORM`,
        replaced by `deepnorm_constants(layers, decoder)`. Mixing is allowed.

    Returns:
        (alpha, beta), both floats
    """
    for name, value in (("alpha", alpha), ("beta", beta)):
        if isinstance(value, str) and value != DEEPNORM:
            raise ValueError(
                f"Unknown {name} setting {value!r}. Give a number to pin it, or "
                f"{DEEPNORM!r} to derive it from the stack's depth."
            )

    derived_alpha, derived_beta = (
        deepnorm_constants(layers, decoder=decoder)
        if DEEPNORM in (alpha, beta)
        else (None, None)
    )

    return (
        derived_alpha if alpha == DEEPNORM else float(alpha),
        derived_beta if beta == DEEPNORM else float(beta),
    )


def scale_without_attenuating(x, scale):
    """
    Multiply `x` by `scale` in the forward pass, and hide `scale` from the
        backward pass: `d/dx` is 1 rather than `scale`, while `d/dscale` is `x`
        as usual.

    Both agents hold the volume of a decision in a lone learned scalar sitting
        at the front of a normalised quantity -- `logit_scale` over the
        speaker's layer-normed vocabulary logits, `score_scale` over the
        listener's per-game standardised candidate scores. A scalar that
        multiplies the forward normally multiplies the backward too, and in both
        places everything upstream of the scale is the machinery that would make
        raising it worthwhile: the speaker's stack and vision model on one side,
        the message, the token embedding and the Gumbel channel on the other.

    That coupling is a loop rather than a cost. BCE's minimiser is `p = 0.5`
        everywhere, so on a message carrying nothing the scale is correct to
        fall; falling then multiplies down the gradient reaching the pair, which
        keeps the message carrying nothing. A listener that quietens starves the
        speaker that would have made it worth listening to. `a9a6a9c` broke the
        loop by deleting the scalars and putting the volume on the weight
        matrices, which stopped the volume moving at all -- 1.3% over thirty
        epochs against the scalar's 59%. This breaks it the other way and keeps
        the scalar.

    The two halves are wanted separately and this is what separates them. Note
        what is *not* given up: the saturation. The gradient reaching `x` is
        `sigma(scale * x) - y` where honest BCE gives `scale * (sigma(scale * x)
        - y)`, so both go quiet on examples already scored correctly once
        `scale` is large, and only the uniform factor differs.

    Pre-multiplying the input by `1 / scale` instead does nothing, because the
        normalisation upstream of the scale is scale-invariant: it annihilates
        the factor forward, and differentiating that same invariance gives
        `J(u / s) = s * J(u)`, which reintroduces exactly what the `1 / s` was
        meant to cancel. A normaliser cannot see a multiplication.

    Args:
        x: the normalised quantity being scaled
        scale: a positive scalar tensor, typically the exp of a learned log

    Returns:
        `scale * x`, with `x`'s gradient path unattenuated.
    """
    return scale * x.detach() + (x - x.detach())

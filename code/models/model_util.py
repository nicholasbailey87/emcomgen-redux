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

"""
Model building utils
"""


import torch.nn as nn

from broccoli.activation import ReLU, GELU, SquaredReLU, SwiGLU


# Name -> broccoli activation, so that `activation` can be set from a TOML
#     config. This mirrors the map broccoli's own `ViT.__init__` applies to
#     string arguments, and deliberately offers the same four options.
#     `TransformerEncoder` has no such map and takes the class (or, for the
#     GLU variants, the factory) directly, so the lookup has to happen here.
ACTIVATIONS = {
    "ReLU": ReLU,
    "GELU": GELU,
    "SquaredReLU": SquaredReLU,
    "SwiGLU": SwiGLU,
}


def get_activation(name):
    """
    Look up a broccoli activation by name.

    Raises
    ------
    ValueError
        If `name` is not a known activation. A config typo should say what the
        valid options are, rather than surfacing as a bare KeyError from deep
        inside model construction.
    """
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Valid options: {', '.join(sorted(ACTIVATIONS))}."
        )


# Sentinel for the `alpha` and `beta` residual-scaling settings: resolve them
#     from the stack's own depth rather than restating two constants in every
#     config that changes `layers`. Either setting may be given as a number
#     instead, which pins it.
DEEPNORM = "deepnorm"


def deepnorm_constants(layers):
    """
    DeepNorm's residual scaling for a post-norm encoder stack `layers` deep:
        `alpha = (2N)^(1/4)` on the skip connection and `beta = (8N)^(-1/4)` on
        the branch, from Wang et al. (2022), *"DeepNet: Scaling Transformers to
        1,000 Layers"* (https://arxiv.org/abs/2203.00555).

    The encoder constants rather than the decoder ones, everywhere this repo
        uses them. DeepNorm's decoder form assumes a cross-attention sublayer
        inside every block; neither stack here is built that way.
        `SenderTransformerLM` runs one cross-attention over the prototypes to
        build the sequence its encoder then reads, and
        `TransformerCrossAttentionComparer` runs one between two encoder
        stacks. In both, the cross-attention sits outside the residual path
        whose depth this is correcting for.

    broccoli applies `beta` as a forward multiplier on the branch rather than
        as an initialisation scaling on its projections, and the post-norm
        `RMSNorm` that follows carries a learnable gain -- so a branch that
        earns it can learn its way back out of the opening ratio. The constants
        set where a stack *starts*, not a ceiling it is held to.

    Args:
        layers: the number of blocks on the residual path being scaled. For a
            stack that is split into sub-stacks, this is the depth of the
            sub-stack, resolved once per sub-stack.

    Returns:
        (alpha, beta)
    """
    if layers < 1:
        raise ValueError(
            f"DeepNorm constants need at least one layer, got {layers}. A stack "
            f"with no blocks has no residual path to scale, so pin `alpha` and "
            f"`beta` to 1.0 instead."
        )

    return (2.0 * layers) ** 0.25, (8.0 * layers) ** -0.25


def resolve_residual_scaling(alpha, beta, layers):
    """
    Resolve the configured `alpha` and `beta` against a stack's depth.

    Each is either a number, which is passed through untouched, or the string
        `DEEPNORM`, which is replaced by `deepnorm_constants(layers)`. Mixing
        the two is allowed: pinning one while deriving the other is a coherent
        thing to want, and refusing it would only push the arithmetic back into
        the config.

    Args:
        alpha: configured skip scaling, a number or `DEEPNORM`
        beta: configured branch scaling, a number or `DEEPNORM`
        layers: depth of the residual path these scale

    Returns:
        (alpha, beta), both floats

    Raises
    ------
    ValueError
        If either is a string other than `DEEPNORM`. A typo should say what it
        should have been rather than reaching broccoli as a string and failing
        somewhere inside a forward pass.
    """
    for name, value in (("alpha", alpha), ("beta", beta)):
        if isinstance(value, str) and value != DEEPNORM:
            raise ValueError(
                f"Unknown {name} setting {value!r}. Give a number to pin it, or "
                f"{DEEPNORM!r} to derive it from the stack's depth."
            )

    derived_alpha, derived_beta = (
        deepnorm_constants(layers)
        if DEEPNORM in (alpha, beta)
        else (None, None)
    )

    return (
        derived_alpha if alpha == DEEPNORM else float(alpha),
        derived_beta if beta == DEEPNORM else float(beta),
    )


def reset_sequential(seq):
    for layer in seq:
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()

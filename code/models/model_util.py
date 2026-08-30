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
    `scale * x` in the forward pass, with `d/dx = 1` rather than `scale`.

    The volume reaches the loss exactly as it always did -- every downstream
        number is `scale * x` -- but the stack behind `x` is no longer
        multiplied by it on the way back. `scale` keeps its true partial,
        `dL/dscale = <dL/dy, x>`, so it learns from an unchanged signal and is
        as free to go quiet as it ever was.

    **This is round nine.** `7b10d47` did it on the speaker and `a9a6a9c` and
        the sixth round of tests/test_score_scale.py's preamble did it here;
        round seven took it out again on the grounds that the coupling never
        reached the optimiser. That argument is right as far as it goes: AdamW
        updates by `m/sqrt(v)`, so a constant factor on a parameter's gradient
        scales numerator and denominator alike and cancels, per parameter and
        independently of every other parameter's gradient size. Do not
        reinstate the "it changes a ratio between modules" reasoning -- AdamW
        normalises each parameter separately, so that ratio is exactly what it
        removes.

    **What it does not cover.** AdamW and `clip_gradients` both act after the
        backward pass. `train.py` runs the forward under `autocast`, so under
        `float16` a gradient the volume has divided down can underflow to zero
        before either of them sees it, and no optimiser recovers a zero. That is
        the failure `docs/anecdotes.md` records as skipped steps. It does not
        arise under `bfloat16`, which has float32's exponent range -- so on a
        GPU that reports `is_bf16_supported()` this function is inert and round
        seven stands. Check the dtype before reading a result as evidence
        either way.

    **The bracketing is load-bearing**, for the reason `Sender.sample`'s
        identity estimator documents: `x - x.detach()` must be formed before it
        is added, or float32 rounds `(scale * x + x) - x` and perturbs the
        forward value this function promises to leave alone.

    Args:
        x: the tensor to scale
        scale: a scalar tensor, the volume

    Returns:
        `scale * x` in value, with the gradient described above
    """
    return scale * x.detach() + (x - x.detach())

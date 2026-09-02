"""
Model building utils
"""

import torch.nn as nn

from broccoli.activation import ReLU, GELU, SquaredReLU, Swish, SwiGLU
from broccoli.transformer import FeedforwardBlock


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


class ReferentAdapter(nn.Module):
    """
    The one stage between a vision backbone and everything that consumes its
        output, on both agents and on every rung.

    **Why it exists.** Until this was added, `feature_model.final_feat_dim` was
        threaded straight into the prototyper, the language model, the contrast
        stage and the discriminator, so a single scalar chosen by the backbone
        set the width of the entire agent. That coupling is what made rung 9
        unreadable as an experiment: `SenderTransformerLM` rejects
        `token_embedding_size != referent_embedding_size` outright, so the
        speaker's language model had to take the ViT's 320 and the ViT had to
        take the language model's, and neither could move without the other.
        The language model is quadratic in width -- 5,854,089 parameters at 320
        against 12,113,481 at 512 -- so 320 was the only width at which rung 9
        was capacity-matched to the GRU baseline it is compared against, and the
        vision model was pinned there by that match rather than by anything
        about vision.

        With this in the path, the backbone emits whatever it emits and the
        agent runs at its language model's `d_model`. Backbone capacity and
        language model capacity become independent variables, which is what a
        comparison across backbones needs.

    **An architectural constant, not a rung.** It is present on both agents at
        every rung, at the same shape, so it is never what a rung is testing.
        The alternative -- introducing it only where a width has to change --
        would put an extra stage on exactly the rungs whose results are being
        compared, which is the confound it exists to remove.

    **Shape.** A `broccoli` `FeedforwardBlock`, SwiGLU, inner size twice the
        output width. The block has no internal residual, so an input width
        different from its output width is native rather than something worked
        around with a projection on a shortcut. Note SwiGLU doubles the up
        projection: at inner size `2 * d_out` the first linear is `4 * d_out`
        wide, so a 512 -> 320 adapter is 862,721 parameters and not the ~500k
        the ratio suggests.

    **The output norm is the block's own.** `FeedforwardBlock.process` already
        ends in `RMSNorm(output_features, elementwise_affine=True)`, so what
        leaves here is normalised with a learnable gain and nothing is stacked
        on top of it. An `nn.LayerNorm` after that would re-centre and re-scale
        what the RMSNorm gain had just set, which is a second normalisation
        rather than the one asked for. The learnable affine is the point: unlike
        `ExampleContrast.adapter`, whose non-affine norm exists so the
        backbone's scale divides out exactly, this stage is a width change that
        downstream modules read as their input distribution, so it is allowed to
        choose that distribution's scale.
    """

    def __init__(self, input_features, output_features, activation="SwiGLU"):
        """
        Args:
            input_features: the backbone's `final_feat_dim`
            output_features: the agent's language model `d_model`
        """
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.block = FeedforwardBlock(
            input_features,
            output_features,
            ratio=2,
            activation=get_activation(activation),
        )

    def forward(self, x):
        return self.block(x)

    def reset_parameters(self):
        """
        `FeedforwardBlock.reset_parameters` walks its `process` sequence and
            calls `reset_parameters` on whatever has one, and broccoli's `Swish`
            does not have one -- so under SwiGLU its `swish_beta` survives a
            reset that is supposed to return the whole stage to its opening
            state. That matters because `receiver_reset_interval` resets the
            listener mid-run, and a parameter that persists across that is a
            parameter the reset was not told about.

        Restored here explicitly rather than by a `hasattr` guard on the walk,
            which is the same choice `Sender.reset_parameters` makes: a guard
            turns a module that has no reset into one that is silently skipped.
            1.0 is `Swish.__init__`'s opening value.
        """
        self.block.reset_parameters()
        for module in self.block.modules():
            if isinstance(module, Swish):
                nn.init.constant_(module.swish_beta, 1.0)

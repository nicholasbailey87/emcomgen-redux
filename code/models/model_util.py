"""
Model building utils
"""

import torch.nn as nn

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


# Well below `F.layer_norm`'s 1e-5 default, and load-bearing wherever it is
#     used: at 1e-5 the normaliser quietly stops normalising once the incoming
#     variance gets small, and whatever it was dividing out goes back to the
#     module upstream. `sender.LAYER_NORM_EPS` and `receiver.LAYER_NORM_EPS`
#     are this constant, re-exported under the names they have always had. See
#     docs/channel.md.
LAYER_NORM_EPS = 1e-12


class LinearInterface(nn.Module):
    """
    How anything reaches a module that declared a width: a plain linear map to
        that width, an affine-free `LayerNorm`, and -- where the consumer asks
        for one -- a dropout mask.

    **One class for both agents.** This is `Sender.adapter`, every one of
        `Receiver.interfaces`, and nothing else. They were two classes briefly
        and the split did not survive contact: a stage that changes width and
        then normalises is one idea, and having a `ReferentAdapter` on the
        speaker beside a listener-only `LinearInterface` made it look like two.

    **The rule, stated once and applied everywhere.**

        Every swappable module declares the widths it wants. The agent brings
        each input to the declared width and hands it over in a stated
        distribution.

            referent interfaces: dropout(norm(adapter(referents)))
            message interfaces:  norm(adapter(message_repr))

        Adapters are plain `nn.Linear`. Norms are affine-free `LayerNorm`.
        Masks are drawn independently per referent interface.

    **Why it exists at all.** Until the first version of this was added,
        `feature_model.final_feat_dim` was threaded straight into the
        prototyper, the language model, the contrast stage and the
        discriminator, so a single scalar chosen by the backbone set the width
        of an entire agent. That coupling is what made rung 9 unreadable as an
        experiment: `SenderTransformerLM` rejects `token_embedding_size !=
        referent_embedding_size` outright, so the speaker's language model had
        to take the ViT's 320 and the ViT had to take the language model's, and
        neither could move without the other. The language model is quadratic
        in width -- 5,854,089 parameters at 320 against 12,113,481 at 512 -- so
        320 was the only width at which rung 9 was capacity-matched to the GRU
        baseline it is compared against, and the vision model was pinned there
        by that match rather than by anything about vision.

        With these in the path the backbone emits whatever it emits and each
        consumer runs at the width it asked for. Backbone capacity and language
        model capacity become independent variables, which is what a comparison
        across backbones needs.

    **An architectural constant, not a rung.** One is present on the speaker at
        every rung and one per declared input on the listener, at the same
        shapes, so an interface is never what a rung is testing. The
        alternative -- introducing one only where a width has to change --
        would put an extra stage on exactly the rungs whose results are being
        compared, which is the confound this exists to remove.

    **A plain `nn.Linear`, and that is the change worth stating.** The
        speaker's was a broccoli `FeedforwardBlock` -- SwiGLU, inner size twice
        the output width, ending in an affine `RMSNorm`, no residual.

        A random linear map approximately preserves inner products, so at
        initialisation the geometry every consumer reads is the backbone's
        geometry. A random SwiGLU block does not: the gate is a multiplication,
        so angles are scrambled before anything has learned to unscramble them.
        The listener's entire early signal is angular -- magnitude cannot flip
        the sign of a bilinear score under BCE, only direction can -- so that
        was a cost paid at exactly the moment a run decides whether to ignite.
        A linear map lets the information through and leaves the departure from
        it to be learned rather than assumed.

    **The norm is unconditional, and it is what the block's `RMSNorm` used to
        do.** That norm was the only thing between a backbone and its
        downstream stages that bounded the feature scale, and dropping it with
        the block would have left the speaker's prototyper reading whatever
        magnitude its vision model happened to emit -- differently per batch on
        a `BatchNorm` trunk, and differently again at eval. So the norm stays;
        what goes is the learnable gain on it. A gain is a route to global
        magnitude, and this stage is not where an agent should be choosing one.

        It is also unconditional in the config sense. The listener's operand
        norms used to be gated by a `[receiver_discriminator]` key, which made
        "deliver an input in a stated distribution" a thing a rung could switch
        off. Nothing in that table reaches them now: `scale_score` and
        `bias_score` build one `ScoreVolume` scalar each and nothing else, and
        the `1/sqrt(d)` calibration below them is unconditional too. See
        `receiver.BilinearDiscriminator`.

    **`bias=False` by default, and load-bearing rather than tidy.**
        `LN(W(cx)) = LN(cW(x)) = LN(W(x))` exactly, because the norm divides
        out any common factor; `LN(W(cx) + b)` does not collapse, because as
        `c` moves the bias's share of the pre-norm vector moves with it. So a
        bias on a referent interface would break the agent's invariance to
        whatever scale its backbone happens to emit at -- not only at
        initialisation, which a zero init would cover, but for the whole run,
        during which a `BatchNorm` trunk's output scale really does drift. That
        invariance is pinned over seven orders of magnitude by
        `test_scores_are_independent_of_the_referent_magnitude`.

        A bias would also buy very little. The norm subtracts the mean, so a
        uniform bias is annihilated outright, and what survives is a constant
        direction added to every candidate before normalising -- a component
        common across candidates, which is what `ScoreVolume.score_bias`
        already is and what `ScoreVolume`'s docstring argues should not be
        spent out of the discriminative capacity.

        Message interfaces pass `bias=True`. No cross-backbone scale claim
        rides on the message side: it arrives through the Gumbel channel rather
        than off a vision model.

    **No dropout unless the consumer asks for one**, and on the listener only
        the referent interfaces do. The message already arrives through the
        Gumbel channel, whose noise `uniform_weight` calibrates, so a mask on
        top is a second and uncalibrated perturbation of a signal that already
        has one.
    """

    def __init__(self, input_size, output_size, bias=False, dropout=None):
        """
        Args:
            input_size: the producer's width -- a backbone's `final_feat_dim`
                on a referent interface, a language model's `output_size` on a
                message one
            output_size: the width the consumer declared
            bias: `False` on referent interfaces, `True` on message ones; see
                the class docstring for why that asymmetry is deliberate
            dropout: the mask rate, or `None` for no mask at all. `None` rather
                than 0.0 so that "this interface is never masked" is a
                structural fact about the interface and not a rate a config
                could accidentally set.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.adapter = nn.Linear(input_size, output_size, bias=bias)
        self.norm = nn.LayerNorm(
            output_size, elementwise_affine=False, eps=LAYER_NORM_EPS
        )
        self.dropout = None if dropout is None else nn.Dropout(p=dropout)

    def forward(self, x):
        adapted = self.norm(self.adapter(x))
        return adapted if self.dropout is None else self.dropout(adapted)

    def reset_parameters(self):
        """
        Two submodules, one of which holds nothing. The norm is reset anyway so
            that turning `elementwise_affine` back on cannot leave a reset agent
            holding trained gains.

        This used to re-initialise `Swish.swish_beta` by hand as well, because
            `FeedforwardBlock.reset_parameters` walks its `process` sequence and
            broccoli's `Swish` has no `reset_parameters` for the walk to find --
            so under SwiGLU that parameter survived a reset which was supposed
            to return the whole stage to its opening state, and
            `receiver_reset_interval` resets an agent mid-run. There is no
            `Swish` here any more, and nothing left for that fix-up to
            re-initialise.
        """
        self.adapter.reset_parameters()
        self.norm.reset_parameters()

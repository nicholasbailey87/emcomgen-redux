# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import functools
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

from broccoli.transformer import FeedforwardBlock
from broccoli.vit import ViT, SequencePoolClassificationHead

from ..model_util import get_activation, resolve_residual_scaling

# The ViT's patch grid is this many patches on a side, whatever the image size,
#     so every dataset runs the same sequence length. See `ViT2.__init__` for
#     why the grid is the fixed quantity and the patch size the derived one.
PATCH_GRID = 16

# The patch-embedding stems `ViT2` can build; `[sender_feature_model] stem`.
STEMS = ("feedforward", "cnn")

# The ViT's feedforward linears, bias-free since 2026-09-27. Attention was
#     already bias-free inside broccoli; these were the last biases in the
#     transformer body.
#
# Why. ShapeWorld arrives unnormalised with ~94% of every image at exactly 0.0,
#     so most patches tokenise to the zero vector. With biases, the embedding
#     feedforward turns every one of those into the same learned constant, and
#     ~240 copies of one key outbid the content tokens in every head by about
#     log 240 while handing back a value that says nothing about the image.
#     `lr_sweep_4_sender_transformer_lm` measured the consequence:
#     `train_referent_spread_backbone` at 0.014 for twenty epochs on ShapeWorld
#     against 0.59 on birds. Without biases a zero patch stays exactly zero into
#     the first block, where it scores 0 against every query and returns a zero
#     value, so `out_norm` rescales it out of the content tokens' reading.
#
# Not a complete fix. Past block 1 the background tokens are ~240 identical
#     copies of the image's mean again, and a mean keeps colour and loses shape.
#     Dropping all-zero patches from the sequence is the version that removes
#     the vote entirely.
#
# `functools.partial` rather than a wrapper that swallows `bias`: broccoli calls
#     these positionally as `(in, out)` and never passes `bias`, so the two
#     only differ for a caller who asks for a bias explicitly.
BIAS_FREE_LINEAR = functools.partial(nn.Linear, bias=False)


class ViT2(nn.Module):
    def __init__(
            self,
            n_feats=(3, 64, 64),
            **kwargs
        ):
        super().__init__()
        
        self.d_model = kwargs["d_model"]

        # The patch-embedding stem: "feedforward" (ShapeWorld) or "cnn"
        #     (CUB). See `[sender_feature_model] stem` for why they differ.
        self.stem = kwargs["stem"]
        if self.stem not in STEMS:
            raise ValueError(
                f"Unknown stem {self.stem!r}. Valid options: "
                f"{', '.join(STEMS)}."
            )
        cnn_stem = self.stem == "cnn"

        # The CNN stem runs with the raw patches in the stream, as
        #     vit-representation-learning's `9-bigger.json` does, and broccoli
        #     then scales the stem by `beta` as one more residual branch.
        self.alpha, self.beta = resolve_residual_scaling(
            kwargs["alpha"], kwargs["beta"], kwargs["layers"],
            stem_branch=cnn_stem,
        )

        self.image_max_side = max(n_feats[1:])

        # The patch grid, fixed at 16x16 and derived from the image size.
        #
        # **256 tokens on every dataset since 2026-09-09**, where the rule used
        #     to fix the patch *size* at three 32nds of the image and let the
        #     token count fall out: 6px patches on an 11x11 grid at
        #     ShapeWorld's 64px, 20px on a 12x12 grid at CUB's 224px. Fixing
        #     the grid instead makes the sequence the constant and the patch
        #     the consequence -- 4px patches at 64px, 14px at 224px.
        #
        # Why. ShapeWorld's concepts are shapes, and a shape's identity is in
        #     its outline. The CNN the ViT rung is compared against is a
        #     `CifarResNet`: a 3x3 stride-1 stem at full resolution, so its
        #     first look at a boundary is a 3px window moving one pixel at a
        #     time. A 6px non-overlapping patch is the opposite -- an edge
        #     falling inside one is never seen against its neighbourhood, and
        #     the model has to put the outline back together from the
        #     positional grid. `lr_sweep_2_sender_vit` sat in the colour-only
        #     minimum at all five rates while the CNN under the same recipe
        #     learned shape, and colour is the feature that survives any
        #     tiling. This is a hypothesis about that result, not a measurement
        #     of it.
        #
        # The tiling still does not overlap. `pooling_type` is `"concat"`, so
        #     the tokenizer is a space-to-depth and at stride = kernel every
        #     pixel reaches the transformer exactly once; the old stride =
        #     kernel/2 duplicated each pixel four times rather than adding
        #     information. Finer patches buy the locality that overlap was
        #     bought for, and buy it without the duplication.
        #
        # What it costs. Tokens go 121 -> 256 at 64px, and the attention term
        #     is quadratic in them -- but at `d_model` 128 that term is a
        #     minority of the arithmetic: per token per layer the projections
        #     are 4d^2 and the feedforward 4*d*ff against a score/AV pair of
        #     2*n*d, so ~14% at 121 tokens and ~25% at 256. Per-token cost
        #     rises about 1.15x and the sequence 2.1x. `scripts/vit_geometry_
        #     sweep.py` puts 16x16 at 2.95 GMAC/img against 1.30 at 11x11, and
        #     it is the harness for timing rather than counting them.
        #
        # The patch size is the patch-embedding feedforward's input width:
        #     4*4*3 = 48 at 64px and 14*14*3 = 588 at 224px. Since broccoli 31
        #     the stem reads that width as it is; 30.x zero-padded it up to
        #     `d_model` first when it was narrower.
        #
        # `pooling_padding` is whatever makes the tiling cover the image, split
        #     symmetrically. It is 0 at both live sizes, which divide by 16
        #     exactly; the formula is kept for a size that does not.
        self.pooling_kernel_size = math.ceil(self.image_max_side / PATCH_GRID)
        self.pooling_kernel_stride = self.pooling_kernel_size
        self.pooling_padding = (
            PATCH_GRID * self.pooling_kernel_size - self.image_max_side + 1
        ) // 2

        # Every broccoli argument is set explicitly, including the inert ones.
        #     See docs/broccoli.md.
        self.backbone = ViT(
            input_size=n_feats[1:],
            image_classes=self.d_model, # Just return an overall embedding
            in_channels=n_feats[0],
            # No `BatchNorm2d(3)` over the raw image, on either dataset.
            #     ShapeWorld lost it on 2026-09-10 and birds on 2026-10-25;
            #     see `ShapeWorldViT` and `BirdsViT` for why. Not the only
            #     possible BatchNorm: the CNN stem below brings its own
            #     (`cnn_batch_norm`), which birds still runs.
            initial_batch_norm=False,
            # The CNN stem is a linear patch embedding: one convolution at the
            #     patch grid's kernel and stride, `d_model` channels, no
            #     activation, after `9-bigger.json`. broccoli gives the
            #     convolution a bias and follows it with a BatchNorm, neither
            #     of which can be turned off -- which is why ShapeWorld does
            #     not run it. Inert under the feedforward stem.
            cnn=cnn_stem,
            cnn_out_channels=self.d_model,
            cnn_kernel_size=self.pooling_kernel_size,
            cnn_kernel_stride=self.pooling_kernel_stride,
            cnn_padding=self.pooling_padding,
            cnn_kernel_dilation=1,
            cnn_kernel_groups=1,
            cnn_activation=nn.Identity,
            cnn_activation_kwargs=None,
            cnn_dropout=0.,
            # No pooling after the CNN stem, which already sets the grid.
            pooling_type=None if cnn_stem else kwargs["pooling_type"],
            # Derived from the image size, not configured: these size the patch
            #     grid, and so the transformer's source_size, from the data.
            pooling_kernel_size=self.pooling_kernel_size,
            pooling_kernel_stride=self.pooling_kernel_stride,
            pooling_padding=self.pooling_padding,
            # With the CNN stem, the raw patch (resized to fit `d_model` if
            #     larger, then zero-padded) is added to the stream beside the
            #     stem, as in `9-bigger.json`. Off with the feedforward stem,
            #     as in `5b-ff-stem-normed-gelu.json`. broccoli 31 removed the
            #     stem's residual path; this is what replaced it. See
            #     docs/broccoli.md.
            images_in_stream=cnn_stem,
            # The feedforward stem broccoli builds here is replaced below, to
            #     give it its own inner size; these arguments still set what it
            #     would be.
            transformer_feedforward_first=not cnn_stem,
            transformer_initial_ff_activation=get_activation(
                kwargs["stem_activation"]
            ),
            transformer_initial_ff_activation_kwargs=None,
            transformer_initial_ff_linear_module_up=None,
            transformer_initial_ff_linear_module_down=None,
            # None means "fall back to the corresponding `transformer_ff_*`
            #     value", which is 0. in each case — not "no dropout arg".
            transformer_initial_ff_dropout=None,
            transformer_initial_ff_inner_dropout=None,
            transformer_initial_ff_outer_dropout=None,
            # Bias-free, and the two `transformer_initial_ff_*` modules above
            #     fall back to these, so every feedforward in the stack --
            #     patch embedding included -- runs without biases. See
            #     `BIAS_FREE_LINEAR`.
            transformer_ff_linear_module_up=BIAS_FREE_LINEAR,
            transformer_ff_linear_module_down=BIAS_FREE_LINEAR,
            transformer_pre_norm=kwargs["pre_norm"],
            transformer_post_norm=kwargs["post_norm"],
            # Pinned False, and no longer a config option; every stack here
            #     runs rotary. See docs/broccoli.md.
            transformer_absolute_position_embedding=False,
            transformer_relative_position_embedding=kwargs[
                "relative_position_embedding"
            ],
            # Pinned at 1.0, so every head receives axial RoPE, and no longer
            #     configurable. See docs/broccoli.md.
            transformer_positional_heads=1.0,
            transformer_embedding_size=self.d_model,
            transformer_layers=kwargs["layers"],
            transformer_heads=kwargs["heads"],
            # `ff_ratio` must be None here or it wins -- `ViT` resolves the two
            #     in the *opposite* order to `FeedforwardBlock`. See
            #     docs/broccoli.md.
            transformer_ff_ratio=None,
            transformer_ff_inner_size=kwargs["ff_inner_size"],
            transformer_bos_tokens=kwargs["utility_tokens"],
            transformer_knocking_heads=kwargs["knocking_heads"],
            transformer_return_bos_tokens=kwargs["return_bos_tokens"],
            transformer_activation=get_activation(kwargs["activation"]),
            transformer_activation_kwargs=None,
            transformer_msa_scaling="d",
            # Pinned rather than promoted: this argument can never take effect.
            #     Use the inner/outer knobs instead. See docs/broccoli.md.
            transformer_ff_dropout=0.,
            transformer_ff_inner_dropout=kwargs["ff_inner_dropout"],
            transformer_ff_outer_dropout=kwargs["ff_outer_dropout"],
            transformer_msa_dropout=kwargs["self_attention_dropout"],
            transformer_stochastic_depth=kwargs["stochastic_depth"],
            transformer_depthwise_linear_stochastic_depth=kwargs[
                "depthwise_linear_stochastic_depth"
            ],
            # Pinned False, and deliberately not a config key. This backbone's
            #     output is left unnormalised on purpose; whichever consumer
            #     needs a controlled magnitude normalises it where the score is
            #     formed. See docs/broccoli.md.
            batch_norm_logits=False,
            logit_projection_layer=nn.Linear,
            linear_module=nn.Linear,
            head=SequencePoolClassificationHead,
            # Residual branch scaling, resolved against this stack's depth when
            #     the config asks for `"deepnorm"` or `"branches"`. See
            #     docs/broccoli.md.
            alpha=self.alpha,
            beta=self.beta,
        )

        # The patch-embedding stem, rebuilt at `stem_ff_inner_size`. broccoli's
        #     `ViT` gives the stem `transformer_ff_inner_size`, the body's, and
        #     has no argument to separate them; ShapeWorld's stem is wider
        #     than its body (see `[sender_feature_model]`). Every other
        #     argument is the one broccoli passed, so only the width differs.
        if not cnn_stem:
            stem = self.backbone.encoder.initial_ff
            self.backbone.encoder.initial_ff = FeedforwardBlock(
                stem.linear_in.in_features,
                self.d_model,
                ratio=None,
                inner_size=kwargs["stem_ff_inner_size"],
                activation=get_activation(kwargs["stem_activation"]),
                activation_kwargs=None,
                dropout=0.,
                inner_dropout=kwargs["ff_inner_dropout"],
                outer_dropout=kwargs["ff_outer_dropout"],
                linear_module_up=BIAS_FREE_LINEAR,
                linear_module_down=BIAS_FREE_LINEAR,
            )
        self.final_feat_dim = self.d_model

    def forward(self, x):
        return self.backbone(x)

    def reset_parameters(self):
        """
        Delegate to broccoli's `ViT`, which resets its own encoder and head --
            the CNN stem's BatchNorm included since broccoli 8da3992.
        """
        self.backbone.reset_parameters()


def ShapeWorldViT(*args, **kwargs):
    """
    `ViT2` under ShapeWorld's name: 128 wide, 6
        layers, 4 heads, `NormedSwiGLU` at inner 170 behind a `NormedGELU`
        stem at inner 256 -- 851,763 parameters against `ResNet56`'s 852,368,
        since 2026-10-22 (874,417 before, at inner 256 with GELU).

    The size is not here. It is in `[sender_feature_model]` and
        `[receiver_feature_model]`, which is where every backbone's
        hyperparameters live. Since 2026-10-25 this factory, like `BirdsViT`,
        does nothing beyond naming the class: a pure alias, the way `ResNet56`
        and `ResNet18` are over `ResNet`. From 2026-09-10 to then it passed
        `initial_batch_norm=False` where `BirdsViT` passed `True`, and the 6
        parameters between 876,593 and 876,599 were that layer's weight and
        bias. `ViT2` now builds no input BatchNorm on either dataset.

    **Why it came off here first.** `ViT2` opened with an
        `nn.BatchNorm2d(3)` over the raw image on both datasets until
        2026-09-10. ShapeWorld runs the feedforward stem, so on ShapeWorld it
        was the only BatchNorm in the stack (birds' CNN stem adds a second,
        `cnn_batch_norm`). ShapeWorld's images arrive as [0, 1] with a black
        background at exactly 0.0 (`train.prepare_batch` divides the uint8
        store by 255), so that layer
        was doing two things at once: standing in for the input normalisation
        ShapeWorld has none of, and making the background nonzero.

    The second is the problem. `CifarResNet` opens with a bias-free 3x3
        convolution on the same [0, 1] tensor, and a conv's weight gradient is
        `sum(x * dL/dy)` -- so a pixel at exactly 0.0 contributes nothing, and
        roughly 94% of a ShapeWorld image is background. Under a BatchNorm the
        background is mapped to `-mean_c/sd_c` instead, which is nonzero,
        channel-selective, and a function of the batch's colour composition.
        That gives colour a route into the transformer's first layer, across
        the whole image area, that the CNN it is size-matched against does not
        have; shape has no equivalent, since no per-channel batch statistic
        encodes an outline. On ShapeWorld, where the standing failure is a
        colour-only minimum, that is worth removing.

    CUB kept it until 2026-10-25, because none of that reaches birds: there
        is no black background to lift. See `BirdsViT` for why it went anyway.

    What the name buys is a key. `[optimiser.implementation_lr]` is keyed by
        the class named in the config, and both datasets run the same `ViT2`
        code at sizes chosen against their own baseline CNNs -- so one name
        could hold one rate for two architectures that have no reason to want
        the same one. `ResNet18` and `ResNet18SmallInput` are the precedent:
        two factories over one `ResNet` class, told apart because
        `GROUP_IMPLEMENTATION` reads the config string rather than
        `type(module).__name__`.

    `parse_config.validate_config` refuses a `ShapeWorldViT` on a `cub`
        dataset. With the two names aliases again, that check guards a label
        -- but the label is what selects the learning rate, so a wrong one is
        a run at the wrong rate under a name saying otherwise.
    """
    return ViT2(*args, **kwargs)


def BirdsViT(*args, **kwargs):
    """
    `ViT2` under CUB's name: 320 wide, 10 layers, 5 heads at head_dim 64,
        `NormedSwiGLU` at inner 704 behind a CNN stem with the raw patches in
        the stream -- 11,168,091 parameters against `ResNet18`'s 11,176,512,
        since the input BatchNorm came off on 2026-10-25 (11,168,097 with it
        earlier that day; 10,612,078 before 2026-10-22, at inner 576 with
        SwiGLU and a feedforward stem).

    Sized by `[birds.sender_feature_model]` and
        `[birds.receiver_feature_model]`, which the dataset name selects. See
        `ShapeWorldViT` for why the two stacks have two names when they are one
        class: the learning-rate table is keyed by name, so the two can take
        different rates. DEFAULT.toml currently gives both 5e-5 (speaker from
        `experiments/lr_sweep_2_sender_vit/`, listener from
        `experiments/lr_sweep_5_receiver_vit/`).

    **No `BatchNorm2d(3)` over the raw image since 2026-10-25.** It was kept
        here when it came off ShapeWorld because the argument against it there
        -- a black background lifted off zero, colour-selectively -- does not
        reach CUB. But nothing argued *for* it either. `image_util.TransformLoader`
        already ends in a torchvision ImageNet normalise, so the layer had
        little left to do beyond swapping ImageNet's channel statistics for
        CUB's batch ones and adding a per-channel affine, at the price of
        another train/eval gap between batch and running statistics. The CNN
        stem's own BatchNorm (`cnn_batch_norm`) is unaffected. See
        `ShapeWorldViT`.
    """
    return ViT2(*args, **kwargs)


# Basic ResNet model
def init_layer(L):
    # He initialisation using fan-out (kernel area * out_channels)
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# Simple Conv Block
class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]

        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        self.reset_parameters()

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out

    def reset_parameters(self):
        # Reproduce construction exactly: PyTorch's own initialisation, then
        #     `init_layer` overriding the weights. See docs/anecdotes.md.
        for layer in self.parametrized_layers:
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                layer.reset_parameters()
                init_layer(layer)


# Simple ResNet Block
class SimpleBlock(nn.Module):
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(
            indim,
            outdim,
            kernel_size=3,
            stride=2 if half_res else 1,
            padding=1,
            bias=False,
        )
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            self.shortcut = nn.Conv2d(
                indim, outdim, 1, 2 if half_res else 1, bias=False
            )
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = "1x1"
        else:
            self.shortcut_type = "identity"

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = (
            x if self.shortcut_type == "identity" else self.BNshortcut(self.shortcut(x))
        )
        out = out + short_out
        out = self.relu2(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1024

    def forward(self, x):
        out = self.trunk(x)
        return out

    def reset_parameters(self):
        for layer in self.trunk:
            if isinstance(layer, ConvBlock):
                layer.reset_parameters()


def Conv4(
    **kwargs
):
    return ConvNet(4)

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        list_of_num_layers,
        list_of_out_dims,
        flatten=True,
        small_input_stem=False,
    ):
        """
        Args:
            small_input_stem: replace the ImageNet stem -- 7x7 stride 2 followed
                by a 3x3 stride-2 maxpool -- with a 3x3 stride-1 convolution and
                no pooling, as SimCLR does for CIFAR-10 (Chen et al. 2020,
                arXiv:2002.05709). See docs/architecture.md.
        """
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        assert len(list_of_num_layers) == 4, "Can have only four stages"

        if small_input_stem:
            conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu]

        if not small_input_stem:
            trunk.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            # Adaptive rather than `AvgPool2d(7)`, which hardcodes a 224px input.
            #     Numerically identical at 224. See docs/architecture.md.
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out

    def reset_parameters(self):
        """
        Re-initialise every layer exactly as `__init__` did, buffers included.
            Recursing over `self.modules()` is what reaches the residual blocks;
            see docs/anecdotes.md.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                init_layer(module)
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()

def ResNet18(*args, **kwargs):
    rn18 = ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten=True)
    return rn18


class CifarBlock(nn.Module):
    """
    He et al. 2015's CIFAR residual block: two 3x3 convolutions and an
        *option A* shortcut.

    A sibling of `SimpleBlock` rather than a configuration of it, because the
        two differ in the one place a channel list cannot express. `SimpleBlock`
        projects a widening shortcut through a 1x1 convolution and a BatchNorm
        -- option B -- where this one subsamples the spatial axes by taking
        every other pixel and zero-pads the channel axis. Option A carries no
        parameters at all, which is why He et al. chose it for CIFAR: the
        residual net then has exactly the parameter count of the plain net it is
        being compared against, so the comparison is about the shortcut and not
        about capacity. See section 4.2.

    Args:
        indim: input channels
        outdim: output channels; may only be `indim` or a multiple of it
        half_res: stride 2 on the first convolution, and on the shortcut
    """

    def __init__(self, indim, outdim, half_res):
        super(CifarBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.half_res = half_res

        self.C1 = nn.Conv2d(
            indim,
            outdim,
            kernel_size=3,
            stride=2 if half_res else 1,
            padding=1,
            bias=False,
        )
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.shortcut_type = (
            "identity" if (indim == outdim and not half_res) else "zero_pad"
        )

        for layer in self.parametrized_layers:
            init_layer(layer)

    def shortcut(self, x):
        """
        The identity, brought to the block's output shape without parameters.

        Stride-2 subsampling rather than pooling, and zero padding split evenly
            across the channel axis, which is what option A is. `F.pad`'s pad
            list runs from the last axis backwards, so the channel pair is the
            fifth and sixth entries.
        """
        if self.shortcut_type == "identity":
            return x

        if self.half_res:
            x = x[:, :, ::2, ::2]

        missing = self.outdim - self.indim

        return F.pad(x, (0, 0, 0, 0, missing // 2, missing - missing // 2))

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = out + self.shortcut(x)
        out = self.relu2(out)
        return out


class CifarResNet(nn.Module):
    """
    He et al. 2015 section 4.2's CIFAR-10 network at any depth `6n + 2`.

    A separate class from `ResNet`, which is the ImageNet family and asserts
        four stages. This one is three stages of `n` blocks at 16, 32 and 64
        channels on a 3x3 stride-1 stem, with stride 2 at the first block of the
        second and third stages, and a global average pool. There is no maxpool
        and no widening beyond 64: the whole network is `6n + 2` weighted layers
        and, at n = 9, 852,368 parameters.

    Args:
        n: blocks per stage. 9 gives ResNet-56.
    """

    def __init__(self, n):
        super(CifarResNet, self).__init__()

        conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(16)
        relu = nn.ReLU()

        trunk = [conv1, bn1, relu]

        indim = 16
        for stage, outdim in enumerate([16, 32, 64]):
            for block in range(n):
                trunk.append(
                    CifarBlock(indim, outdim, half_res=(stage >= 1 and block == 0))
                )
                indim = outdim

        # As `ResNet`: adaptive, so nothing pins this backbone to one input
        #     resolution. It runs on ShapeWorld's 64px images, where the last
        #     stage is a 16x16 map.
        trunk.append(nn.AdaptiveAvgPool2d((1, 1)))
        trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = indim

        self.reset_parameters()

    def forward(self, x):
        return self.trunk(x)

    def reset_parameters(self):
        """
        Re-initialise every layer exactly as `__init__` did, buffers included.

        Recursing over `self.modules()` rather than over `self.trunk` is what
            reaches the fifty-four convolutions inside the blocks; the same
            mistake in `ResNet` left 11.1M of 11.18M parameters untouched by a
            reset. See docs/anecdotes.md.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                init_layer(module)
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()


def ResNet56(*args, **kwargs):
    """
    `CifarResNet` at n = 9: 56 weighted layers, 852,368 parameters,
        `final_feat_dim` 64.

    ShapeWorld's backbone on both agents. A factory that swallows its arguments,
        as every backbone factory here does, because the config selects a
        backbone by name and a name is the whole of the registration.
    """
    return CifarResNet(9)

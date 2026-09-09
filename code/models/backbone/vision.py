# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

from broccoli.activation import ReLU
from broccoli.vit import ViT, SequencePoolClassificationHead

from ..model_util import get_activation, resolve_residual_scaling

# The ViT's patch grid is this many patches on a side, whatever the image size,
#     so every dataset runs the same sequence length. See `ViT2.__init__` for
#     why the grid is the fixed quantity and the patch size the derived one.
PATCH_GRID = 16


class ViT2(nn.Module):
    def __init__(
            self,
            n_feats=(3, 64, 64),
            **kwargs
        ):
        super().__init__()
        
        self.d_model = kwargs["d_model"]


        self.alpha, self.beta = resolve_residual_scaling(
            kwargs["alpha"], kwargs["beta"], kwargs["layers"]
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
        # Stride and kernel appear in no weight shape at 64px, so ShapeWorld's
        #     876,599 parameters are untouched and the ablation's
        #     size-matching claim against `ResNet56` still holds. CUB's moves,
        #     because `ResizeAndPadPatches` carries a patch of 14*14*3 = 588
        #     values where it carried 20*20*3 = 1,200.
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
            initial_batch_norm=True,
            # The whole `cnn_*` group is inert while `cnn` is False, and pinned
            #     so that flipping `cnn` on is a deliberate act.
            cnn=False,
            cnn_out_channels=16,
            cnn_kernel_size=3,
            cnn_kernel_stride=1,
            cnn_padding="same",
            cnn_kernel_dilation=1,
            cnn_kernel_groups=1,
            cnn_activation=ReLU,
            cnn_activation_kwargs=None,
            cnn_dropout=0.,
            pooling_type=kwargs["pooling_type"],
            # Derived from the image size, not configured: these size the patch
            #     grid, and so the transformer's source_size, from the data.
            pooling_kernel_size=self.pooling_kernel_size,
            pooling_kernel_stride=self.pooling_kernel_stride,
            pooling_padding=self.pooling_padding,
            transformer_feedforward_first=True,
            # On: broccoli 30.1.0 carries the residual with
            #     `ResizeAndPadPatches`, so `d_model` is no longer tied to the
            #     patch size. See docs/broccoli.md.
            transformer_initial_ff_residual_path=True,
            transformer_initial_ff_linear_module_up=None,
            transformer_initial_ff_linear_module_down=None,
            # None means "fall back to the corresponding `transformer_ff_*`
            #     value", which is 0. in each case — not "no dropout arg".
            transformer_initial_ff_dropout=None,
            transformer_initial_ff_inner_dropout=None,
            transformer_initial_ff_outer_dropout=None,
            transformer_ff_linear_module_up=None,
            transformer_ff_linear_module_down=None,
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
            #     the config asks for `"deepnorm"`. See docs/broccoli.md.
            alpha=self.alpha,
            beta=self.beta,
        )
        self.final_feat_dim = self.d_model

    def forward(self, x):
        return self.backbone(x)

    def reset_parameters(self):
        """Delegate to broccoli's `ViT`, which resets its own encoder and head."""
        self.backbone.reset_parameters()


def ShapeWorldViT(*args, **kwargs):
    """
    `ViT2` under ShapeWorld's name: 128 wide, 6 layers, 4 heads, feedforward
        inner 256, GELU -- 876,599 parameters against `ResNet56`'s 852,368.

    The size is not here. It is in `[sender_feature_model]` and
        `[receiver_feature_model]`, which is where every backbone's
        hyperparameters live; this factory swallows its arguments and returns
        the class, exactly as `ResNet56` and `ResNet18` do.

    What the name buys is a key. `[optimiser.implementation_lr]` is keyed by
        the class named in the config, and both datasets run the same `ViT2`
        code at sizes chosen against their own baseline CNNs -- so one name
        could hold one rate for two architectures that have no reason to want
        the same one. `ResNet18` and `ResNet18SmallInput` are the precedent:
        two factories over one `ResNet` class, told apart because
        `GROUP_IMPLEMENTATION` reads the config string rather than
        `type(module).__name__`.

    `parse_config.validate_config` refuses a `ShapeWorldViT` on a `cub`
        dataset, so the label cannot drift from the block that sizes it.
    """
    return ViT2(*args, **kwargs)


def BirdsViT(*args, **kwargs):
    """
    `ViT2` under CUB's name: 320 wide, 10 layers, 5 heads at head_dim 64,
        feedforward inner 576, SwiGLU -- 10,626,990 parameters against
        `ResNet18`'s 11,176,512.

    Sized by `[birds.sender_feature_model]` and
        `[birds.receiver_feature_model]`, which the dataset name selects. See
        `ShapeWorldViT` for why the two stacks have two names when they are one
        class: the learning-rate table is keyed by name, and the birds arm of
        `experiments/lr_sweep_2_sender_vit/` chose 2e-5 for this one alone.
    """
    return ViT2(*args, **kwargs)


# Basic ResNet model
def init_layer(L):
    # Initialization using fan-in
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

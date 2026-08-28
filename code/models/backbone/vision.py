# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

from broccoli.activation import ReLU
from broccoli.vit import ViT, SequencePoolClassificationHead

from ..model_util import get_activation, resolve_residual_scaling

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

        def close_even_number(x):
            return int(x) if int(x) % 2 == 0 else int(x) - 1

        # The patch grid, derived from the image size rather than configured.
        #
        # The tiling is non-overlapping: `pooling_type` is `"concat"`, so the
        #     tokenizer is a space-to-depth, and at stride = kernel it is an
        #     exact tiling in which every pixel reaches the transformer exactly
        #     once. The previous geometry ran stride = kernel/2, which does not
        #     add information -- it duplicates each pixel four times. What the
        #     overlap bought was a locality prior and a finer positional grid,
        #     and it cost 289 tokens against 121 here.
        #
        # It cost more than the arithmetic suggests. Measured on an A100 at 640
        #     images of 64px, fwd+bwd, bf16, compiled: 303ms at the old
        #     geometry against 118ms at this one, where the
        #     `ResNet18SmallInput` these backbones are compared against runs in
        #     81ms. The ViT was 3.75x the baseline's wall clock and is now
        #     1.46x. See `scripts/vit_geometry_sweep.py`, which is the harness
        #     those numbers came from and can re-derive them.
        #
        # The `x 3` was `x 4`. Together with the stride it puts ShapeWorld's
        #     64px on an 11x11 grid of 6px patches and CUB's 224px on a 12x12
        #     grid of 20px ones. Stride appears in no weight shape, so the
        #     ShapeWorld parameter count is untouched at 10,319,266, or 92% of
        #     ResNet18's; CUB's moves, because a 20px patch is 1,200 values and
        #     a 28px one 2,352, and above `d_model` that difference is carried
        #     by `ResizeAndPadPatches`. It moves the right way -- 101% of
        #     ResNet18 where the old geometry was 113%.
        #
        # `pooling_padding` is whatever makes the tiling cover the image, split
        #     symmetrically: 1 each side at 64px, 8 at 224px. Without it the
        #     final partial patch is silently cropped, which is a strip of the
        #     image the model cannot see.
        self.pooling_kernel_size = close_even_number((self.image_max_side / 32) * 3)
        self.pooling_kernel_stride = self.pooling_kernel_size
        patch_grid = math.ceil(self.image_max_side / self.pooling_kernel_size)
        self.pooling_padding = (
            patch_grid * self.pooling_kernel_size - self.image_max_side + 1
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


def ResNet18SmallInput(*args, **kwargs):
    """
    `ResNet18` with SimCLR's small-image stem -- see `ResNet.__init__`.

    A separate factory rather than a flag because the backbone is selected by
        name from the config, so a name is the whole of the registration. Both
        factories swallow their arguments, as every backbone factory here does.
    """
    return ResNet(
        SimpleBlock,
        [2, 2, 2, 2],
        [64, 128, 256, 512],
        flatten=True,
        small_input_stem=True,
    )


# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import torchvision.models as models

from broccoli.activation import ReLU
from broccoli.vit import ViT, SequencePoolClassificationHead

from ..model_util import get_activation

class ViT2(nn.Module):
    def __init__(
            self,
            n_feats=(3, 64, 64),
            **kwargs
        ):
        super().__init__()
        
        self.d_model = kwargs["d_model"]

        self.image_max_side = max(n_feats[1:])

        def close_even_number(x):
            return int(x) if int(x) % 2 == 0 else int(x) - 1
        
        self.pooling_kernel_size = close_even_number((self.image_max_side / 32) * 4)
        self.pooling_kernel_stride = int(self.pooling_kernel_size / 2)
        self.pooling_padding = self.pooling_kernel_stride

        # As in `receiver.py` and `sender.py`, every broccoli argument is set
        #     explicitly, including the inert ones, because broccoli's defaults
        #     have changed underneath this repository before. See the note at
        #     the top of `receiver.py`.
        self.backbone = ViT(
            input_size=n_feats[1:],
            image_classes=self.d_model, # Just return an overall embedding
            in_channels=n_feats[0],
            initial_batch_norm=True,
            # The whole `cnn_*` group is inert while `cnn` is False: broccoli
            #     swaps in an Identity and the image goes straight to pooling.
            #     Pinned so that flipping `cnn` on is a deliberate act with
            #     visible settings, rather than silently adopting defaults.
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
            transformer_initial_ff_residual_path=False, # So that d_model can be as small as we like
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
            transformer_absolute_position_embedding=kwargs[
                "absolute_position_embedding"
            ],
            transformer_relative_position_embedding=kwargs[
                "relative_position_embedding"
            ],
            # Live: with relative position embedding on, this decides how many
            #     heads receive axial RoPE.
            transformer_positional_heads=kwargs["positional_heads"],
            transformer_embedding_size=self.d_model,
            transformer_layers=kwargs["layers"],
            transformer_heads=kwargs["heads"],
            transformer_ff_ratio=kwargs["ff_ratio"],
            transformer_ff_inner_size=None, # inert: `ff_ratio` sizes the block
            transformer_bos_tokens=kwargs["utility_tokens"],
            transformer_knocking_heads=kwargs["knocking_heads"],
            transformer_return_bos_tokens=kwargs["return_bos_tokens"],
            transformer_activation=get_activation(kwargs["activation"]),
            transformer_activation_kwargs=None,
            transformer_msa_scaling="d",
            # Not configurable, and pinned rather than promoted: broccoli's
            # `FeedforwardBlock` uses this only as a fallback --
            # `inner_dropout if inner_dropout is not None else dropout` -- and
            # `TransformerEncoder` always forwards `ff_inner_dropout` and
            # `ff_outer_dropout`, which default to 0.0 rather than None. So this
            # argument can never take effect, and TOML has no way to write the
            # None that would let it. Use the inner/outer knobs instead.
            transformer_ff_dropout=0.,
            transformer_ff_inner_dropout=kwargs["ff_inner_dropout"],
            transformer_ff_outer_dropout=kwargs["ff_outer_dropout"],
            transformer_msa_dropout=kwargs["self_attention_dropout"],
            transformer_stochastic_depth=kwargs["stochastic_depth"],
            transformer_depthwise_linear_stochastic_depth=kwargs[
                "depthwise_linear_stochastic_depth"
            ],
            batch_norm_logits=kwargs["batch_norm_logits"],
            logit_projection_layer=nn.Linear,
            linear_module=nn.Linear,
            head=SequencePoolClassificationHead,
            # Residual branch scaling. broccoli moved away from deepnorm at
            #     30.0.0, and 1.0 is the no-scaling identity either way.
            alpha=kwargs["alpha"],
            beta=kwargs["beta"],
        )
        self.final_feat_dim = self.d_model

    def forward(self, x):
        return self.backbone(x)

    def reset_parameters(self):
        """
        Delegate to broccoli's `ViT`, which resets its own encoder and head.

        Without this, `Receiver.reset_parameters` raises `AttributeError` for
            any rung using a ViT listener, and `Sender.reset_parameters` --
            which guards on `hasattr` -- silently left the whole speaker
            backbone untouched instead. Every other feature model here defines
            the method, so its absence was the odd one out rather than a
            deliberate opt-out.
        """
        self.backbone.reset_parameters()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Basic ResNet model
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        WeightNorm.apply(
            self.L, "weight", dim=0
        )  # split the weight update component to direction and norm
        self.relu = nn.ReLU()

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        L_norm = (
            torch.norm(self.L.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.L.weight.data)
        )
        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)  # matrix product by forward function
        scores = 10 * (
            cos_dist
        )  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax

        return scores


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super(Conv2d_fw, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(
                    x, self.weight.fast, None, stride=self.stride, padding=self.padding
                )
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(
                    x,
                    self.weight.fast,
                    self.bias.fast,
                    stride=self.stride,
                    padding=self.padding,
                )
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).to(x.device)
        running_var = torch.ones(x.data.size()[1]).to(x.device)
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight.fast,
                self.bias.fast,
                training=True,
                momentum=1,
            )
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                training=True,
                momentum=1,
            )
        return out


# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
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
        # Reproduce construction exactly, which is PyTorch's own initialisation
        #     followed by `init_layer` overriding the weights -- `__init__`
        #     builds the layers (so `nn.Conv2d.__init__` seeds weight *and*
        #     bias) and only then calls this method. Going straight to
        #     `init_layer` skipped the conv biases, which it does not touch, and
        #     the BatchNorm running statistics, which are buffers rather than
        #     parameters. Both were then carried across a reset.
        # `parametrized_layers` also holds the ReLU and (when pooling) the
        #     MaxPool2d, neither of which has parameters or a
        #     `reset_parameters`, so select the two types that do rather than
        #     excluding types one at a time.
        for layer in self.parametrized_layers:
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                layer.reset_parameters()
                init_layer(layer)


# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(
                indim,
                outdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
                bias=False,
            )
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
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
            if self.maml:
                self.shortcut = Conv2d_fw(
                    indim, outdim, 1, 2 if half_res else 1, bias=False
                )
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
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


# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim / 4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1, bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(
                bottleneckdim,
                bottleneckdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
            )
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1, bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(
                bottleneckdim,
                bottleneckdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
            )
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [
            self.C1,
            self.BN1,
            self.C2,
            self.BN2,
            self.C3,
            self.BN3,
        ]
        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(
                    indim, outdim, 1, stride=2 if half_res else 1, bias=False
                )
            else:
                self.shortcut = nn.Conv2d(
                    indim, outdim, 1, stride=2 if half_res else 1, bias=False
                )

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = "1x1"
        else:
            self.shortcut_type = "identity"

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):

        short_out = x if self.shortcut_type == "identity" else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
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

# MODELS BELOW HERE SHOULD NOT BE USED

# class ConvNetNopool(
#     nn.Module
# ):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
#     def __init__(self, depth, flatten=False):
#         super(ConvNetNopool, self).__init__()
#         trunk = []
#         for i in range(depth):
#             indim = 3 if i == 0 else 64
#             outdim = 64
#             B = ConvBlock(
#                 indim, outdim, pool=(i in [0, 1]), padding=0 if i in [0, 1] else 1
#             )  # only first two layer has pooling and no padding
#             trunk.append(B)

#         if flatten:
#             trunk.append(Flatten())

#         self.trunk = nn.Sequential(*trunk)
#         if flatten:
#             # FIXME: This dimension is for conv4 only
#             self.final_feat_dim = 12544
#         else:
#             self.final_feat_dim = [64, 19, 19]

#     def forward(self, x):
#         out = self.trunk(x)
#         return out


# class ConvNetS(nn.Module):  # For omniglot, only 1 input channel, output dim is 64
#     def __init__(self, depth, flatten=True):
#         super(ConvNetS, self).__init__()
#         trunk = []
#         for i in range(depth):
#             indim = 1 if i == 0 else 64
#             outdim = 64
#             B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
#             trunk.append(B)

#         if flatten:
#             trunk.append(Flatten())

#         self.trunk = nn.Sequential(*trunk)
#         self.final_feat_dim = 64

#     def forward(self, x):
#         out = x[:, 0:1, :, :]  # only use the first dimension
#         out = self.trunk(out)
#         return out


# class ConvNetSNopool(
#     nn.Module
# ):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
#     def __init__(self, depth):
#         super(ConvNetSNopool, self).__init__()
#         trunk = []
#         for i in range(depth):
#             indim = 1 if i == 0 else 64
#             outdim = 64
#             B = ConvBlock(
#                 indim, outdim, pool=(i in [0, 1]), padding=0 if i in [0, 1] else 1
#             )  # only first two layer has pooling and no padding
#             trunk.append(B)

#         self.trunk = nn.Sequential(*trunk)
#         self.final_feat_dim = [64, 5, 5]

#     def forward(self, x):
#         out = x[:, 0:1, :, :]  # only use the first dimension
#         out = self.trunk(out)
#         return out


class ResNet(nn.Module):
    maml = False  # Default

    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        assert len(list_of_num_layers) == 4, "Can have only four stages"
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            # Adaptive rather than `AvgPool2d(7)`, which hardcodes a 224px input.
            #     Below that the pooling window is larger than the feature map
            #     and the forward pass errors; above it a single 7x7 window
            #     silently *crops* the map rather than pooling it (at 320px the
            #     map is 10x10 and three rows and columns are discarded), which
            #     also leaves `final_feat_dim` wrong. Numerically identical at
            #     224, where the map is exactly 7x7. This matches torchvision's
            #     `resnet18`, which is otherwise this network exactly: same
            #     layout, same stride placement, same fan-out init.
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
        Re-initialise every layer exactly as `__init__` did.

        This used to walk `self.trunk` and call `reset_parameters()` on anything
            that had one. `SimpleBlock` defines no such method, so the eight
            residual blocks -- 11.1M of the 11.18M parameters -- were skipped
            entirely, and the two layers that *were* reached (the stem conv and
            BN) got PyTorch's defaults, which for `Conv2d` is kaiming *uniform*
            rather than the fan-out normal `init_layer` applies at construction.
            One tensor of sixty was reset, with the wrong distribution.

        Recursing over `self.modules()` reaches the blocks, and going through
            `init_layer` keeps a reset indistinguishable from a fresh build.

        BatchNorm running statistics are buffers rather than parameters, so they
            are reset too: leaving them would carry the pre-reset feature
            distribution across the reset, which is not what
            `receiver_reset_interval` is asking for.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                init_layer(module)
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()

# def Conv6():
#     return ConvNet(6)


# def Conv4NP():
#     return ConvNetNopool(4, flatten=True)


# def Conv6NP():
#     return ConvNetNopool(6)


# def Conv4S():
#     return ConvNetS(4)


# def Conv4SNP():
#     return ConvNetSNopool(4)


# def ResNet10(flatten=True):
#     return ResNet(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten)


# def reset_parameters(model):
#     def weight_reset(m):
#         if (
#             isinstance(m, nn.Conv1d)
#             or isinstance(m, nn.Conv2d)
#             or isinstance(m, nn.Linear)
#             or isinstance(m, nn.Conv3d)
#             or isinstance(m, nn.ConvTranspose1d)
#             or isinstance(m, nn.ConvTranspose2d)
#             or isinstance(m, nn.ConvTranspose3d)
#             or isinstance(m, nn.BatchNorm1d)
#             or isinstance(m, nn.BatchNorm2d)
#             or isinstance(m, nn.BatchNorm3d)
#             or isinstance(m, nn.GroupNorm)
#         ):
#             m.reset_parameters()

#     model.apply(weight_reset)


def ResNet18(*args, **kwargs):
    rn18 = ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten=True)
    return rn18


# def PretrainedResNet18():
#     rn18 = models.resnet18(pretrained=True)
#     rn18.final_feat_dim = 512
#     rn18.fc = Identity()  # We don't use final fc
#     # Define reset parameters on resnet18
#     rn18.reset_parameters = reset_parameters.__get__(rn18)
#     return rn18


# def ResNet34(flatten=True):
#     return ResNet(SimpleBlock, [3, 4, 6, 3], [64, 128, 256, 512], flatten)


# def ResNet50(flatten=True):
#     return ResNet(BottleneckBlock, [3, 4, 6, 3], [256, 512, 1024, 2048], flatten)


# def ResNet101(flatten=True):
#     return ResNet(BottleneckBlock, [3, 4, 23, 3], [256, 512, 1024, 2048], flatten)

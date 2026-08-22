"""
Tests for the vision backbones, `reset_parameters`, and per-module gradient
clipping.

Runnable without pytest:  python tests/test_backbones.py

None of this was covered before. An edit that broke `Conv4` construction
outright -- every ShapeWorld rung, and both agents of it -- left the suite fully
green, because nothing in `tests/` ever built a backbone. The first two tests
here exist so that that specific silence cannot happen again.

The rest pin the three properties that were actually wrong in the code, each of
which fails quietly rather than loudly:

`reset_parameters` walked `self.trunk` calling `reset_parameters()` on whatever
had one. `SimpleBlock` has none, so 11.1M of `ResNet`'s 11.18M parameters were
skipped and the two layers that were reached got PyTorch's kaiming *uniform*
instead of `init_layer`'s fan-out normal. `ConvBlock` went straight to
`init_layer`, which touches neither conv biases nor BatchNorm running statistics,
so both survived a reset. A reset that silently resets almost nothing looks
exactly like a reset that worked, which is why `test_reset_parameters_*` asserts
on coverage and on the resulting *distribution* rather than on the call
returning.

`AvgPool2d(7)` hardcoded a 224px input: below that it errored, and above it a
single 7x7 window cropped the feature map rather than pooling it (at 320px the
map is 10x10 and three rows and columns were discarded), returning a
plausible-looking vector that is not a global pool.
`test_resnet_is_resolution_independent` covers both directions, and
`test_resnet_matches_torchvision_resnet18` pins the claim that this network *is*
torchvision's `resnet18` -- same layout, same fan-out init, same block order --
so the pretrained weights remain a drop-in.

`clip_grad_norm_` scales every gradient by one factor derived from a norm taken
across the whole pair, and the listener's comparer supplies ~90% of that norm.
So a global clip handed the speaker's vision model a coefficient set by the
comparer's batch-to-batch fluctuation. `test_clip_gradients_*` pins the two
properties that make the per-module version correct: that the groups partition
the pair, so nothing can silently go unclipped, and that each module is bounded
by its own gradient alone.
"""

import math
import os
import tempfile

import torch
import torch.nn as nn
import torchvision

import _bootstrap  # noqa: F401

import parse_config
import models.builder
import train
from models.backbone import vision

SHAPEWORLD_FEATS = (3, 64, 64)
BIRDS_FEATS = (3, 224, 224)

# Large enough that a parameter carrying it is unmistakably untouched: every
# initialiser in the codebase produces values far below this.
PERTURBATION = 100.0


def _backbones():
    """
    One instance of each vision backbone, with the feature size it advertises
    and an input shape it is valid at. `ViT2` takes its whole argument list from
    the config, so it is built the way `models.builder` builds it.
    """
    config = parse_config.get_config()  # plain defaults, i.e. ShapeWorld
    # DEFAULT.toml now carries a runnable ViT -- 320 wide, 5 heads, head_dim 64
    # -- so this no longer has to repair it. Only `layers` is overridden, and
    # only to keep the test cheap: a 10-layer stack is built and forwarded in
    # every one of these cases and none of them is about depth.
    config["sender_feature_model"].update(layers=2)
    return [
        ("Conv4", vision.Conv4(), SHAPEWORLD_FEATS),
        ("ResNet18", vision.ResNet18(), BIRDS_FEATS),
        ("ResNet18SmallInput", vision.ResNet18SmallInput(), SHAPEWORLD_FEATS),
        (
            "ViT2",
            vision.ViT2(n_feats=SHAPEWORLD_FEATS, **config["sender_feature_model"]),
            SHAPEWORLD_FEATS,
        ),
    ]


def _pair(dataset, n_feats, name):
    """A sender/receiver pair built through `models.builder`, as training does."""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(f'name = "test"\n[data]\ndataset = "{dataset}"\n')
        path = f.name
    try:
        config = parse_config.get_config(path)
    finally:
        os.unlink(path)
    config["cuda"] = False

    class _Dataset:
        pass

    _Dataset.n_feats = n_feats
    _Dataset.name = name

    class _Loader:
        dataset = _Dataset()

    return config, models.builder.build_models({"train": _Loader()}, config)["pair"]


def _perturb(module):
    """Move every parameter and float buffer somewhere no initialiser would."""
    with torch.no_grad():
        for p in module.parameters():
            p.add_(PERTURBATION)
        for b in module.buffers():
            if b.is_floating_point():
                b.add_(PERTURBATION)


def _still_perturbed(module):
    """Names of tensors that `reset_parameters` left holding the perturbation."""
    stale = [
        n
        for n, p in module.named_parameters()
        if (p.detach().abs() > PERTURBATION / 2).all()
    ]
    stale += [
        n
        for n, b in module.named_buffers()
        if b.is_floating_point() and (b.abs() > PERTURBATION / 2).all()
    ]
    return stale


def test_resnet18_small_input_stem():
    """
    `ResNet18SmallInput` is `ResNet18` with SimCLR's CIFAR stem, and the reason
    for it is resolution rather than parameters.

    The stock stem downsamples 4x before any residual block -- 7x7 stride 2 then
    a 3x3 stride-2 maxpool -- which on ShapeWorld's 64px images leaves a 2x2 map
    at the end for the adaptive pool to average. Colour survives that; shape does
    not. With the small-input stem the same input reaches the last stage as 8x8.
    """
    stock = vision.ResNet18()
    small = vision.ResNet18SmallInput()

    stem = small.trunk[0]
    assert isinstance(stem, torch.nn.Conv2d)
    assert (stem.kernel_size, stem.stride, stem.padding) == ((3, 3), (1, 1), (1, 1))
    assert not any(isinstance(m, torch.nn.MaxPool2d) for m in small.modules())
    assert any(isinstance(m, torch.nn.MaxPool2d) for m in stock.modules())

    # 3*3*3*64 against 7*7*3*64, and the maxpool has none.
    assert sum(p.numel() for p in stock.parameters()) == 11_176_512
    assert sum(p.numel() for p in small.parameters()) == 11_168_832

    # Same depth, so `ResNet.reset_parameters`' 20-convolution expectation and
    # everything keyed to it still hold.
    convs = [m for m in small.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 20

    assert small.final_feat_dim == stock.final_feat_dim == 512


def test_small_input_stem_keeps_more_of_a_small_image():
    """The claim the stem swap is actually for, measured rather than asserted."""
    x = torch.randn(2, 3, *SHAPEWORLD_FEATS[1:])

    def pre_pool(model):
        # Everything up to, but not including, the adaptive pool and flatten.
        trunk = torch.nn.Sequential(*list(model.trunk)[:-2])
        with torch.no_grad():
            return trunk(x).shape[-2:]

    assert tuple(pre_pool(vision.ResNet18())) == (2, 2)
    assert tuple(pre_pool(vision.ResNet18SmallInput())) == (8, 8)


def test_small_input_stem_is_still_resolution_independent():
    """It is the ShapeWorld backbone, but nothing should pin it to 64px."""
    model = vision.ResNet18SmallInput().eval()
    for size in (64, 112, 224):
        with torch.no_grad():
            out = model(torch.randn(1, 3, size, size))
        assert out.shape == (1, model.final_feat_dim)


def test_every_backbone_constructs_and_forwards():
    """
    The gap that let a broken `Conv4` through a green suite: nothing built one.
    """
    for name, backbone, feats in _backbones():
        out = backbone(torch.randn(2, *feats))
        assert out.ndim == 2, f"{name} returned {out.ndim} dims, expected 2"
        assert out.shape[0] == 2, name
        assert out.shape[1] == backbone.final_feat_dim, (
            f"{name} emits {out.shape[1]} features but advertises "
            f"final_feat_dim={backbone.final_feat_dim}; the agents size their "
            f"input layers off the advertised value"
        )


def test_resnet_is_resolution_independent():
    """
    `AdaptiveAvgPool2d` rather than `AvgPool2d(7)`. Below 224 the fixed pool
    errored; above it, it silently cropped the feature map and `final_feat_dim`
    became a lie.
    """
    backbone = vision.ResNet18().eval()
    for size in (112, 160, 224, 320):
        with torch.no_grad():
            out = backbone(torch.randn(1, 3, size, size))
        assert out.shape == (1, backbone.final_feat_dim), (
            f"at {size}px got {tuple(out.shape)}, expected "
            f"(1, {backbone.final_feat_dim})"
        )


def test_resnet_matches_torchvision_resnet18():
    """
    Same architecture, tensor for tensor and in the same order, so ImageNet
    weights load positionally and the pretrained backbone stays a drop-in.
    """
    mine = vision.ResNet18().eval()
    theirs = torchvision.models.resnet18(weights=None)
    theirs.fc = nn.Identity()
    theirs.eval()

    mine_shapes = [tuple(t.shape) for t in mine.state_dict().values()]
    their_shapes = [tuple(t.shape) for t in theirs.state_dict().values()]
    assert mine_shapes == their_shapes, "state_dict shapes or ordering diverge"

    theirs.load_state_dict(dict(zip(theirs.state_dict(), mine.state_dict().values())))
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(mine(x), theirs(x), atol=1e-5)


def test_reset_parameters_restores_every_backbone():
    """
    Coverage, not just completion: the old `ResNet.reset_parameters` ran without
    error while resetting one tensor of sixty.
    """
    for name, backbone, _ in _backbones():
        _perturb(backbone)
        backbone.reset_parameters()
        stale = _still_perturbed(backbone)
        # broccoli's RoPE frequency tables are deterministic constants
        # (requires_grad=False) that it does not recompute on reset. Nothing
        # downstream reads them as learned state.
        stale = [n for n in stale if "rotary_embedding" not in n]
        if name == "ViT2":
            # broccoli does not restore these two, which is an upstream gap
            # rather than one this repository can fix from here.
            stale = [
                n for n in stale if not (n.endswith("swish_beta") or "norm." in n)
            ]
        assert not stale, f"{name}.reset_parameters left {len(stale)} stale: {stale[:5]}"


def test_reset_parameters_uses_the_construction_time_init():
    """
    `init_layer`'s fan-out normal, not PyTorch's kaiming uniform. Both produce
    plausible weights, so only the distribution distinguishes them.
    """
    backbone = vision.ResNet18()
    _perturb(backbone)
    backbone.reset_parameters()

    convs = [m for m in backbone.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 20, f"expected 20 convs in ResNet-18, found {len(convs)}"
    for conv in convs:
        fan_out = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
        expected = math.sqrt(2.0 / fan_out)
        assert math.isclose(conv.weight.std().item(), expected, rel_tol=0.15), (
            f"conv std {conv.weight.std().item():.5f} against init_layer's "
            f"{expected:.5f} -- kaiming uniform would read ~"
            f"{expected * math.sqrt(1 / 3) * 2:.5f}"
        )
        assert abs(conv.weight.mean().item()) < expected


def test_reset_parameters_restores_normalisation_state():
    """
    BatchNorm affine parameters *and* running statistics. The statistics are
    buffers, so an initialiser-only reset carried the pre-reset feature
    distribution across the reset.
    """
    for _, backbone, _ in _backbones():
        norms = [m for m in backbone.modules() if isinstance(m, nn.BatchNorm2d)]
        if not norms:
            continue
        with torch.no_grad():
            for m in norms:
                m.weight.fill_(5.0)
                m.bias.fill_(9.0)
                m.running_mean.fill_(3.0)
                m.running_var.fill_(7.0)
                m.num_batches_tracked.fill_(11)
        backbone.reset_parameters()
        for m in norms:
            assert (m.weight == 1).all() and (m.bias == 0).all()
            assert (m.running_mean == 0).all() and (m.running_var == 1).all()
            assert int(m.num_batches_tracked) == 0


def test_conv_block_reset_restores_conv_bias():
    """
    `ConvBlock`'s convs carry a bias (unlike `ResNet`'s, which are bias=False),
    and `init_layer` does not touch it, so it used to survive a reset.
    """
    block = vision.ConvBlock(3, 64)
    with torch.no_grad():
        block.C.bias.fill_(PERTURBATION)
    block.reset_parameters()
    assert block.C.bias.abs().max().item() < 1.0


def test_agent_reset_parameters_covers_every_parameter():
    """
    Whole-agent coverage for the two rungs built entirely from this repository's
    own modules. `receiver_reset_interval` drives
    `Receiver.reset_parameters`, and a reset that misses a submodule leaves the
    listener holding what it had learned.
    """
    for dataset, feats, name in (
        ("../data/shapeworld_40", SHAPEWORLD_FEATS, "shapeworld"),
        ("../data/cub", BIRDS_FEATS, "cub"),
    ):
        _, pair = _pair(dataset, feats, name)
        _perturb(pair)
        pair.sender.reset_parameters()
        pair.receiver.reset_parameters()
        stale = _still_perturbed(pair)
        assert not stale, f"{name}: {len(stale)} tensors not reset: {stale[:5]}"


def test_reset_parameters_clears_the_speakers_measured_survival():
    """
    `realised_survival` is measured off the logits a particular set of weights
    produced, so it is meaningless for freshly drawn ones and must not be
    carried across a reset into the first epoch of a re-initialised speaker.

    The `logit_scale` it was measured at is deliberately *not* reset: it is a
    constant resolved from the config and the vocabulary, so it does not depend
    on the weights and there is nothing about it to restore.
    """
    _, pair = _pair("../data/cub", BIRDS_FEATS, "cub")
    speaker = pair.sender.language_model
    speaker.realised_survival = 0.5
    speaker.logit_spread = 0.9
    scale = speaker.logit_scale

    pair.sender.reset_parameters()

    assert math.isnan(speaker.realised_survival)
    assert math.isnan(speaker.logit_spread)
    assert speaker.logit_scale == scale


def test_cross_attention_comparer_reset_covers_its_adapters():
    """
    The baseline rungs use `BilinearGRUComparer`, so the pair-level test above
    never reaches this class. Its `reset_parameters` used to omit both adapters
    and the referent norm -- i.e. everything mapping the listener's two inputs
    into `d_model` -- while re-drawing everything downstream of them.
    """
    import models.receiver as receiver

    config = parse_config.get_config()
    kwargs = dict(config["receiver_comparer"])
    kwargs["layers"] = 2  # so the fusion stack is non-empty
    # `d_model` has to be overridden, as it is everywhere this class is built
    # outside a rung config. DEFAULT's 1024 is `BilinearGRUComparer`'s hidden
    # size, and the section's `heads` describes this class instead -- the two
    # numbers belong to different modules, so the pair does not satisfy
    # broccoli's `d_model % n_heads == 0`. 320 is the width rung 11 uses.
    kwargs["d_model"] = 320
    comparer = receiver.TransformerCrossAttentionComparer(512, **kwargs)

    _perturb(comparer)
    comparer.reset_parameters()

    for name in (
        "referent_adapter",
        "message_adapter",
        "referent_layer_norm",
        "message_cross_attention",
        "message_residual_norm",
        "referent_cross_attention",
        "referent_residual_norm",
        "referent_self_attention",
        "referent_self_attention_norm",
        "decision_layer_norm",
        "decision",
    ):
        module = getattr(comparer, name)
        stale = _still_perturbed(module)
        assert not stale, f"{name} not reset: {stale}"


def _pair_with_gradients():
    torch.manual_seed(0)
    config, pair = _pair("../data/shapeworld_40", SHAPEWORLD_FEATS, "shapeworld")
    n_examples = config["data"]["n_examples"]
    inputs = torch.randn(2, n_examples, *SHAPEWORLD_FEATS)
    targets = torch.zeros(2, n_examples)
    targets[:, : n_examples // 2] = 1.0
    pair.train()
    messages, _ = pair.sender(inputs, targets)
    nn.BCEWithLogitsLoss()(pair.receiver(inputs, messages), targets).backward()
    return pair


def test_clip_gradients_partitions_every_parameter():
    """
    `CLIP_GROUPS` must cover the pair. The `other` fallback catches anything a
    future architecture adds, so its presence is the alarm, not the fix.
    """
    pair = _pair_with_gradients()
    grouped = set()
    for _, select in train.CLIP_GROUPS:
        grouped.update(id(p) for p in select(pair).parameters())
    missing = [n for n, p in pair.named_parameters() if id(p) not in grouped]
    assert not missing, f"outside CLIP_GROUPS: {missing[:5]}"

    assert "other" not in train.clip_gradients(pair, 1.0)


def test_clip_gradients_bounds_each_module_independently():
    """
    The point of clipping per module: a module under the ceiling is left alone
    however large another module's gradient is. Under one global norm the
    speaker's vision model was scaled by ~86x because of the comparer.
    """
    pair = _pair_with_gradients()
    before = train.clip_gradients(pair, 1.0)
    assert before, "no gradients to clip"

    for name, select in train.CLIP_GROUPS:
        grads = [p.grad for p in select(pair).parameters() if p.grad is not None]
        if not grads:
            continue
        after = torch.norm(torch.stack([g.norm() for g in grads])).item()
        assert after <= 1.0 + 1e-4, f"{name} left at {after}"
        if before.get(name, 0.0) <= 1.0:
            # Was already inside the ceiling, so it must not have been touched.
            assert math.isclose(after, before[name], rel_tol=1e-4), (
                f"{name} was {before[name]} before clipping and {after} after, "
                f"despite never reaching the ceiling"
            )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")

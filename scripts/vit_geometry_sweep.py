"""
Wall-clock cost of the ViT patch geometry, on a real GPU.

`ViT2` derives its patch grid from the image size rather than from config:

    pooling_kernel_size   = largest even number <= (max_side / 32) x 4
    pooling_kernel_stride = kernel_size / 2
    pooling_padding       = stride

At 64px that is kernel 8, stride 4, padding 4 -- a 2x-overlapping tiling giving
a 17x17 grid, so 289 tokens. Since `pooling_type` is `"concat"`, the tokenizer
is a space-to-depth: at stride = kernel it is an exact tiling and every pixel
reaches the transformer exactly once, so the overlap is duplicating each pixel
four times rather than adding information. Dropping it takes the sequence to 64
tokens, and nothing else about the model changes -- the parameter count is
identical either way (10,319,266 at 64px), because stride does not appear in any
weight shape.

This script times the candidates against each other and, with
`--include-resnet`, against `ResNet18SmallInput`, which is the backbone the ViT
rungs are replacing and so the wall-clock number to beat.

    kernel stride pad   grid  tokens   MACs/img   what it is
    ------ ------ ---   ----  ------   --------   ----------------------------
       8      4     4   17x17    289   3.43 G     current
       4      4     0   16x16    256   2.95 G     fine patches, no overlap
       6      6     1   11x11    121   1.30 G     middle, one row of padding
       8      8     0    8x8      64   0.67 G     standard ViT tokenization

`ResNet18SmallInput` at 64px is 2.22 GMAC, for reference. Note the current ViT
is only ~1.5x the ResNet's arithmetic, so if it is much slower than that in
practice the gap is throughput rather than work -- which is what the TFLOP/s
column here is for.

Run from the repository root:

    python scripts/vit_geometry_sweep.py
    python scripts/vit_geometry_sweep.py --include-resnet
    python scripts/vit_geometry_sweep.py --image-size 224 \
        --geometries 28,14,14 28,28,0 16,16,0

Forward *and* backward, in `train()` mode, under the same bf16 autocast as
`train.py`, both with and without `torch.compile` since `compile = true` in
`DEFAULT.toml`. A few minutes, most of it compile warm-up. Nothing is written
and no config is read.
"""

import argparse
import time

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import models.backbone.vision as vision  # noqa: E402
from models.backbone.vision import ViT2, ResNet18SmallInput  # noqa: E402


# [sender_feature_model] in DEFAULT.toml, verbatim. The geometry is *not* here,
# because it is derived inside `ViT2` rather than configured -- overriding it is
# what `geometry_override` below exists for.
SPEC = dict(
    d_model=320,
    layers=10,
    heads=5,
    utility_tokens=0,
    ff_inner_size=576,
    stochastic_depth=0.1,
    depthwise_linear_stochastic_depth=True,
    activation="SwiGLU",
    relative_position_embedding=True,
    pre_norm=False,
    post_norm=True,
    return_bos_tokens=False,
    knocking_heads=False,
    pooling_type="concat",
    ff_inner_dropout=0.0,
    ff_outer_dropout=0.0,
    self_attention_dropout=0.0,
    alpha="deepnorm",
    beta="deepnorm",
)

# 20,000 games x 20 examples x 2 agents, the ShapeWorld epoch, as
# `vit_throughput.py` uses it. Only turns a per-step time into a per-epoch one.
# Rungs 11 and 13 put a `ViT2` on *both* agents, so both halves of that product
# are ViT work for them; rungs 5 and 9 have a ResNet on the other side and see
# roughly half the benefit.
IMAGES_PER_EPOCH = 20_000 * 20 * 2


class geometry_override:
    """Force `ViT2`'s derived patch geometry to a given (kernel, stride, pad).

    `ViT2.__init__` computes the three pooling arguments from the image size and
    passes them straight into broccoli's `ViT`, with no way in between to say
    otherwise. Rather than fork the backbone to sweep it, this intercepts the
    constructor call at the point the arguments are handed over. Nothing is left
    patched once the block exits.
    """

    def __init__(self, kernel, stride, padding):
        self.kernel, self.stride, self.padding = kernel, stride, padding

    def __enter__(self):
        self._original = vision.ViT
        kernel, stride, padding = self.kernel, self.stride, self.padding
        original = self._original

        def patched(*args, **kwargs):
            kwargs["pooling_kernel_size"] = kernel
            kwargs["pooling_kernel_stride"] = stride
            kwargs["pooling_padding"] = padding
            return original(*args, **kwargs)

        vision.ViT = patched
        return self

    def __exit__(self, *exc):
        vision.ViT = self._original
        return False


def token_count(image_size, kernel, stride, padding):
    side = (image_size + 2 * padding - kernel) // stride + 1
    return side * side, side


def flops_per_image(build, image_size):
    """FLOPs of one fwd+bwd for one image, counted on the *uncompiled* module.

    The counter sees through eager dispatch, and `torch.compile` changes which
    kernels run but not the arithmetic they do.
    """
    from torch.utils.flop_counter import FlopCounterMode

    model = build().cuda()
    model.train()
    probe = torch.randn(2, 3, image_size, image_size, device="cuda")
    counter = FlopCounterMode(display=False)
    with counter:
        model(probe).float().sum().backward()
    total = counter.get_total_flops() / 2
    del model
    torch.cuda.empty_cache()
    return total


def measure(build, image_size, batch_images, compile_model, steps):
    """Median seconds per fwd+bwd step."""
    model = build().cuda()
    model.train()

    if compile_model:
        model = torch.compile(model)

    x = torch.randn(batch_images, 3, image_size, image_size, device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out.float().sum().backward()
        model.zero_grad(set_to_none=True)

    # Warm-up: the first steps pay for autotuning and, under compile, for
    # tracing. Not the steady state being measured.
    for _ in range(5):
        step()
    torch.cuda.synchronize()

    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()

    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return times[len(times) // 2], peak


def parse_geometry(text):
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"geometry must be kernel,stride,padding -- got {text!r}"
        )
    return tuple(int(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--batch-images",
        type=int,
        default=32 * 20,
        help="images per forward; the ShapeWorld rungs send 32 games x 20 examples",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--geometries",
        type=parse_geometry,
        nargs="+",
        default=[(8, 4, 4), (4, 4, 0), (6, 6, 1), (8, 8, 0)],
        help="kernel,stride,padding triples. The default set is for 64px; "
             "the first is the geometry the rungs currently run.",
    )
    parser.add_argument(
        "--include-resnet",
        action="store_true",
        help="also time ResNet18SmallInput, the backbone the ViT rungs replace",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip the compiled rows; `compile = true` in DEFAULT.toml, so the "
             "compiled number is the one the rungs actually get",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible -- this number is meaningless on CPU.")

    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    print(
        f"{args.batch_images} images of {args.image_size}x{args.image_size}, "
        f"bf16 autocast, fwd+bwd, median of {args.steps}\n"
    )

    header = (
        f"{'backbone':>22} {'tokens':>7} {'params':>11} {'compile':>8} "
        f"{'ms/step':>9} {'TFLOP/s':>9} {'peak GiB':>9} {'epoch':>8} {'vs now':>7}"
    )
    print(header)
    print("-" * len(header))

    compile_modes = (False,) if args.no_compile else (False, True)
    baseline = {}

    for kernel, stride, padding in args.geometries:
        tokens, side = token_count(args.image_size, kernel, stride, padding)

        def build(kernel=kernel, stride=stride, padding=padding):
            with geometry_override(kernel, stride, padding):
                return ViT2(n_feats=(3, args.image_size, args.image_size), **SPEC)

        params = sum(p.numel() for p in build().parameters())
        flops = flops_per_image(build, args.image_size)

        for compile_model in compile_modes:
            seconds, peak = measure(
                build, args.image_size, args.batch_images, compile_model, args.steps
            )
            tflops = flops * args.batch_images / seconds / 1e12
            epoch_minutes = seconds * (IMAGES_PER_EPOCH / args.batch_images) / 60
            baseline.setdefault(compile_model, epoch_minutes)
            speedup = baseline[compile_model] / epoch_minutes

            print(
                f"{f'ViT k{kernel} s{stride} p{padding}':>22} "
                f"{f'{side}x{side}':>7} {params:>11,} {str(compile_model):>8} "
                f"{seconds * 1e3:>9.1f} {tflops:>9.1f} {peak:>9.2f} "
                f"{epoch_minutes:>7.1f}m {speedup:>6.2f}x"
            )

    if args.include_resnet:
        print()
        def build_resnet():
            return ResNet18SmallInput()

        params = sum(p.numel() for p in build_resnet().parameters())
        flops = flops_per_image(build_resnet, args.image_size)
        for compile_model in compile_modes:
            seconds, peak = measure(
                build_resnet, args.image_size, args.batch_images,
                compile_model, args.steps,
            )
            tflops = flops * args.batch_images / seconds / 1e12
            epoch_minutes = seconds * (IMAGES_PER_EPOCH / args.batch_images) / 60
            speedup = baseline[compile_model] / epoch_minutes
            print(
                f"{'ResNet18SmallInput':>22} {'-':>7} {params:>11,} "
                f"{str(compile_model):>8} {seconds * 1e3:>9.1f} {tflops:>9.1f} "
                f"{peak:>9.2f} {epoch_minutes:>7.1f}m {speedup:>6.2f}x"
            )

    print(
        "\n`epoch` is this backbone alone over 20,000 x 20 x 2 images, not a "
        "whole rung.\n`vs now` is against the first geometry listed, at the "
        "same compile setting."
    )


if __name__ == "__main__":
    main()

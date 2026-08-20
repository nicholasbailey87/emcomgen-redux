"""
Microbenchmark for the one shared ViT specification, on a real GPU.

Answers a narrow question: how much of the ablation's ~14 TFLOP/s comes back
from `torch.compile` and from aligning `ff_inner_size`, measured rather than
argued. It builds `ViT2` exactly as `[sender_feature_model]` specifies and runs
the batch shape a ShapeWorld rung actually sees -- 32 games x 20 examples = 640
images at 64x64 -- under the same bf16 autocast as `train.py`.

Forward *and* backward, in `train()` mode. Both matter: eval() with an untrained
BatchNorm is not the regime, and a forward-only number would miss where most of
the time goes.

Run from the repository root:

    python scripts/vit_throughput.py

Roughly two minutes, most of it `torch.compile` warm-up. Nothing is written and
no config is read, so it is safe to run on a login node if the queue is long --
though the number is only meaningful on the GPU the rungs actually get.
"""

import argparse
import time

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from models.backbone.vision import ViT2  # noqa: E402


# [sender_feature_model] in DEFAULT.toml, verbatim apart from `ff_inner_size`,
# which is what the sweep varies.
SPEC = dict(
    d_model=320,
    layers=10,
    heads=5,
    utility_tokens=0,
    stochastic_depth=0.1,
    depthwise_linear_stochastic_depth=True,
    activation="SwiGLU",
    absolute_position_embedding=True,
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

# 20,000 games x 20 examples x 2 agents, the ShapeWorld epoch. Used only to turn
# a per-step time into a per-epoch one, so the output is in the units the
# decision is actually made in.
IMAGES_PER_EPOCH = 20_000 * 20 * 2


def measure(ff_inner_size, image_size, batch_images, compile_model, steps):
    """Median seconds per fwd+bwd step, and the FLOPs that step costs."""
    model = ViT2(
        n_feats=(3, image_size, image_size), ff_inner_size=ff_inner_size, **SPEC
    ).cuda()
    model.train()

    if compile_model:
        model = torch.compile(model)

    x = torch.randn(batch_images, 3, image_size, image_size, device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out.float().sum().backward()
        model.zero_grad(set_to_none=True)

    # Warm-up. The first few steps pay for autotuning and, under compile, for
    # tracing the graph; they are not the steady state being measured.
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

    # FLOPs are counted on the uncompiled module: the counter sees through eager
    # dispatch, and compile does not change the arithmetic, only the kernels.
    from torch.utils.flop_counter import FlopCounterMode

    counter_model = ViT2(
        n_feats=(3, image_size, image_size), ff_inner_size=ff_inner_size, **SPEC
    ).cuda()
    counter_model.train()
    probe = torch.randn(2, 3, image_size, image_size, device="cuda")
    counter = FlopCounterMode(display=False)
    with counter:
        counter_model(probe).float().sum().backward()
    flops_per_image = counter.get_total_flops() / 2

    del model, counter_model
    torch.cuda.empty_cache()

    return times[len(times) // 2], flops_per_image


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
        "--ff-inner-sizes",
        type=int,
        nargs="+",
        default=[554, 576],
        help="554 is the configured value; 576 is the nearest multiple of 64",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible -- this number is meaningless on CPU.")

    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    print(
        f"{args.batch_images} images of {args.image_size}x{args.image_size}, "
        f"bf16 autocast, fwd+bwd, median of {args.steps}\n"
    )
    header = f"{'ff_inner':>9} {'compile':>8} {'ms/step':>9} {'TFLOP/s':>9} {'epoch':>9}"
    print(header)
    print("-" * len(header))

    for ff_inner_size in args.ff_inner_sizes:
        for compile_model in (False, True):
            seconds, flops_per_image = measure(
                ff_inner_size,
                args.image_size,
                args.batch_images,
                compile_model,
                args.steps,
            )
            tflops = flops_per_image * args.batch_images / seconds / 1e12
            epoch_minutes = (
                flops_per_image * IMAGES_PER_EPOCH / (tflops * 1e12) / 60
            )
            print(
                f"{ff_inner_size:>9} {str(compile_model):>8} "
                f"{seconds * 1e3:>9.1f} {tflops:>9.1f} {epoch_minutes:>8.1f}m"
            )


if __name__ == "__main__":
    main()

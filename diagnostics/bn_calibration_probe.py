"""
How many clean batches does a BatchNorm re-estimate need?

    python diagnostics/bn_calibration_probe.py \
        --checkpoint <run>/checkpoint_last.pt \
        --config <run's toml> \
        --n 0,1,2,4,8,16,32,64,128

Silhouetting is training-time only: `shapeworld.load` zeroes both rates and
passes `augment=False` on every split but `train`. So every BatchNorm in the
pair gathers its running mean and variance over a silhouetted/augmented
*mixture* and then applies them to clean images at eval. docs/data.md names the
consequence for the receiver's input layer -- a fixed offset of
`(mu_clean - mu_running) / sigma` on every eval activation, which the learned
affine cannot absorb because at train time that offset is zero -- and the same
argument holds for every deeper BatchNorm.

`train.calibrate_batch_norm` fixes that by re-estimating the statistics on the
`train_clean` loader before each eval pass. What it cannot decide for itself is
how many batches that estimate needs, which is what this script measures and
what `bn_calibration_batches` is then set to.

Method. Rebuild the run from its own config with `silhouette_p_receiver = 0.5`,
load its weights, and **train one epoch at that rate before sweeping**. The
pollution has to be there to be corrected: a checkpoint's statistics were
gathered under whatever rate that run used, and the question is how many batches
undo the damage at 0.5. Then, from those same post-train weights, calibrate at
each N and score `test` and `test_same`.

N = 0 is the uncalibrated baseline. It is also the number every silhouetted run
on record was scored at, including all ten runs of the titration -- so the
column to read the rest against is the first one.

Two things are reported per N and they are meant to agree:

  * `acc`, `acc_md_shape` and `acc_md_color` on both splits, and
  * `stat drift`, the largest relative change in any `running_mean` or
    `running_var` between this N and the previous one.

Where they disagree, **believe the statistics**. Accuracy on one seed is noisy
-- `silhouette_shape_probe.py` swung 0.560 to 0.403 between identical re-runs of
the same arm, on the same GPU, at the same seed -- and a plateau in the drift is
a statement about the estimator rather than about one fit. `--seeds` exists for
the same reason it does there: the accuracies are reported as a mean over that
many draws of the calibration batches and the eval passes, with every individual
fit printed underneath.

The train epoch is run once and shared by every seed and every N. Only the
calibration draw and the eval passes vary within the table, which is the
variance the seeds are here for; re-training per seed would fold a different
question into the same column.
"""
import argparse
import copy
import os
import sys

# Before `torch`, and before any CUDA context exists: cuBLAS reads this at
#     initialisation and `use_deterministic_algorithms` raises without it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

# The same flags `silhouette_shape_probe.py` sets, and for the reason its
#     docstring gives: convolution backward on cuDNN accumulates with atomics,
#     and two runs of that script at one seed disagreed by more than the effect
#     being read.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
)

import data  # noqa: E402
import models.builder  # noqa: E402
import parse_config  # noqa: E402
import paths  # noqa: E402
import train  # noqa: E402

from gradboard.scheduler import PASS  # noqa: E402
from torch.amp import GradScaler  # noqa: E402


REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "bn_calibration_probe")


def build_config(config_path, silhouette_p_receiver, exp_dir):
    """
    The run's own config, with the receiver's silhouette rate overridden.

    The dataset paths are remapped through `paths.data_fast_storage()` exactly
        as `train.py`'s entry point does it, and for the same reason: a config
        names a dataset, and where that dataset is staged is a property of the
        machine.
    """
    config = parse_config.get_config(config_path)

    config['data']['silhouette_p_receiver'] = silhouette_p_receiver

    emcomgen_data = paths.data_fast_storage() / "emcomgen" / "data"
    for key in ('dataset', 'ref_dataset'):
        if key in config['data']:
            basename = os.path.basename(config['data'][key])
            config['data'][key] = str(emcomgen_data / basename)

    # `run` reads this only when `[vis]` is on, but it reads it unguarded from
    #     the config dict, so it has to be a real directory.
    config['exp_dir'] = exp_dir

    return config


def load_weights(pair, scheduler, checkpoint_path):
    """
    Restore either of the two files a run leaves behind.

    `checkpoint_last.pt` holds `epoch` and `scheduler_state` only -- gradboard's
        `PASS` carries the model inside its own state, which is why this goes
        through the scheduler rather than through `pair`. A finished run also
        leaves a plain `pair.state_dict()` at `final_model.pt`.

    Returns:
        a one-line description of what was loaded, for the output header
    """
    blob = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    if isinstance(blob, dict) and "scheduler_state" in blob:
        scheduler.load_state_dict(blob["scheduler_state"])
        return (f"{checkpoint_path} (PASS checkpoint, "
                f"trained {blob.get('epoch', '?')} epochs)")

    pair.load_state_dict(blob)
    return f"{checkpoint_path} (bare pair.state_dict)"


def batch_norm_statistics(pair):
    """Every `_BatchNorm`'s running mean and variance, detached, on the CPU."""
    return [
        (module.running_mean.detach().cpu().clone(),
         module.running_var.detach().cpu().clone())
        for module in pair.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and module.running_mean is not None
    ]


def max_relative_change(statistics, previous):
    """
    The largest change between two sets of statistics, relative to the scale of
        the previous one.

    Per tensor, `max|new - old| / max(|old|)`, then the maximum over tensors.
        Normalising by the largest entry rather than elementwise is deliberate:
        a running mean has entries near zero, and an elementwise ratio would
        report a huge relative change on a channel that barely moved in absolute
        terms. This is the change the *worst* channel makes as a fraction of the
        layer's own scale.

    Returns `nan` when there is nothing to compare against, i.e. at the first N.
    """
    if previous is None:
        return float("nan")

    worst = 0.0
    for (mean, var), (was_mean, was_var) in zip(statistics, previous):
        for new, old in ((mean, was_mean), (var, was_var)):
            scale = old.abs().max().item()
            if scale == 0.0:
                scale = 1.0
            worst = max(worst, (new - old).abs().max().item() / scale)

    return worst


def main(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_batches = [int(n) for n in args.n.split(",")]
    if 0 not in n_batches:
        # Every existing silhouetted run was scored here; a table without it is
        #     a table with no baseline.
        n_batches = [0] + n_batches
    n_batches = sorted(set(n_batches))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    config = build_config(args.config, args.silhouette, RESULTS_DIR)
    dataloaders = data.loader.load_dataloaders(config)
    model_config = models.builder.build_models(dataloaders, config)
    pair = model_config['pair']

    scaler = GradScaler()
    training_examples = len(dataloaders['train'].dataset)
    batch_size = (
        config['data']['batch_size'] * config['optimiser']['accumulator_steps']
    )
    scheduler = PASS(
        train.build_lr_schedule(config, training_examples, batch_size),
        pair,
        model_config['optimiser'],
        scaler=scaler,
        range_test=config['scheduler']['range_test'],
        cool_point_multiplier=0.0,
    )

    loaded = load_weights(pair, scheduler, args.checkpoint)

    if "train_clean" not in dataloaders:
        raise RuntimeError(
            "This dataset has no `train_clean` loader, so there is nothing to "
            "calibrate on. Only ShapeWorld builds one -- see "
            "`data/shapeworld.py`."
        )

    run_args = (
        pair,
        model_config['optimiser'],
        dataloaders,
        scheduler,
        scaler,
        config,
    )

    splits = [s for s in ("test", "test_same") if s in dataloaders]

    print("=" * 78)
    print(f"checkpoint  {loaded}")
    print(f"config      {args.config}")
    print(f"silhouette  p_receiver = {args.silhouette} "
          f"(p_sender left at {config['data']['silhouette_p_sender']})")
    print(f"sweep       N = {n_batches}  |  {args.seeds} seeds  |  "
          f"splits {splits}")
    print(f"batch size  {config['data']['batch_size']} games")
    print("=" * 78, flush=True)

    # One epoch at the swept rate, so the statistics under test are polluted the
    #     way a silhouetted run's are. Shared by every seed and every N; see the
    #     module docstring.
    print(f"\nTraining one epoch at silhouette_p_receiver = {args.silhouette} "
          f"to pollute the statistics...", flush=True)
    train.run("train", 0, *run_args)

    # Every N starts here. `calibrate_batch_norm` writes the running statistics
    #     in place, and they live in `state_dict`, so without this the sweep
    #     would be cumulative rather than a comparison.
    polluted = copy.deepcopy(pair.state_dict())

    metrics_of_interest = ("acc", "acc_md_shape", "acc_md_color")
    results = {}   # (n, split, metric) -> list over seeds
    drift = {}     # n -> list over seeds

    for seed in range(args.seeds):
        previous = None

        for n in n_batches:
            torch.manual_seed(args.seed + seed)
            torch.cuda.manual_seed_all(args.seed + seed)
            np.random.seed(args.seed + seed)

            pair.load_state_dict(polluted)
            train.calibrate_batch_norm(
                pair, dataloaders['train_clean'], config, n
            )

            statistics = batch_norm_statistics(pair)
            drift.setdefault(n, []).append(
                max_relative_change(statistics, previous)
            )
            previous = statistics

            for split in splits:
                metrics, _ = train.run(split, n, *run_args)
                for metric in metrics_of_interest:
                    results.setdefault((n, split, metric), []).append(
                        metrics.get(metric, float("nan"))
                    )

    for split in splits:
        print(f"\n{split}")
        header = (f"  {'N':>5}{'acc':>9}{'sd':>7}"
                  f"{'shape':>9}{'sd':>7}{'colour':>9}{'sd':>7}"
                  f"{'stat drift':>13}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for n in n_batches:
            row = f"  {n:>5}"
            for metric in metrics_of_interest:
                values = np.array(results[(n, split, metric)])
                row += f"{np.nanmean(values):>9.3f}{np.nanstd(values):>7.3f}"
            row += f"{np.nanmean(drift[n]):>13.2e}"
            print(row, flush=True)

    # Every fit, so a reader can take a median or look at the shape of the
    #     numbers without re-running a job that trains an epoch first.
    print(f"\nper seed (0..{args.seeds - 1})")
    for split in splits:
        for metric in metrics_of_interest:
            print(f"  {split} {metric}")
            for n in n_batches:
                values = results[(n, split, metric)]
                print(f"    N={n:<5}" + " ".join(f"{v:.3f}" for v in values))

    print("\nstat drift is the largest relative change in any running_mean or "
          "running_var\nbetween this N and the previous one. Where it and the "
          "accuracies disagree,\nthe drift is the honest answer: one fit's "
          "accuracy is noisy.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="<run>/checkpoint_last.pt or <run>/final_model.pt")
    p.add_argument("--config", required=True,
                   help="the TOML that run was trained from")
    p.add_argument("--n", default="0,1,2,4,8,16,32,64,128",
                   help="calibration batch counts to sweep. 0 is added if "
                        "absent: it is the uncalibrated baseline")
    p.add_argument("--seeds", type=int, default=3,
                   help="draws of the calibration batches and the eval passes "
                        "per N. One fit is not a reading: see the module "
                        "docstring")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed, for the build and the polluting epoch")
    p.add_argument("--silhouette", type=float, default=0.5,
                   help="`[data] silhouette_p_receiver` to pollute and then "
                        "correct at")
    main(p.parse_args())

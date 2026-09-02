"""
Why some rungs ignite and some do not, measured on the gradient rather than
inferred from the metrics.

The observation this exists to explain. In the August 2026 ablation every
ShapeWorld rung's speaker plateaus in the first five epochs and then either
restarts or does not. `pool_score_norm` -- the `AttentionPrototyper` scoring
weight, which `reset_parameters` sets to exactly zero -- shows it most cleanly,
because it has nowhere to go but up:

    rung   ep0      ep5      ep10     ep29     outcome
    ----   ------   ------   ------   ------   ---------------------------
    05     0.0102   0.0111   0.1022   0.0558   escaped at ~epoch 10
    07     0.0112   0.0124   0.0848   0.1501   escaped at ~epoch 10
    09     0.0117   0.0125   0.0125   0.0127   never left the plateau
    11     0.0064   0.0069   0.0069   0.0069   never left the plateau
    13     0.0099   0.0121   0.0121   0.2121   escaped late, ~epoch 20

`logit_scale`, `contrast_gate` and `pool_score_norm` used to leave the plateau
in the same epoch, so this is one event and not three: the speaker sharpens
because the listener started to decode, and the listener decodes because the
speaker sharpened. Ignition, not learning. (The channel scale was a constant
between 2026-08-30 and 2026-08-31 and is a learned parameter again, so all three
move -- but it now opens at 1.0 and is bounded at 2.0 rather than opening where
`init_energy` put it. Do not compare its plateau against the numbers above; see
docs/channel.md.)

Two readings fit the table equally well and the metrics cannot separate them:

  (a) *No gradient reaches the speaker.* Something about `SenderTransformerLM`
      or the modules under it attenuates the path from the loss back to the
      prototyper, so the speaker is stationary because it is being told nothing.

  (b) *Gradient reaches it and cancels.* AdamW normalises by the second moment,
      so a parameter moves about `lr` per step whatever its gradient's size, and
      its travel over a run is `lr * steps` times the *net sign consistency* of
      that gradient. A sign that flips batch to batch produces a stationary
      parameter out of a perfectly healthy gradient.

Nothing logged per epoch separates them, because a per-epoch column shows where
a parameter got to and not what it was told. This script records the per-group
gradient norms at every optimiser step, read after `scaler.unscale_` so they are
in loss units and comparable across configs. Under (a) the norms into
`sender.prototyper` and `sender.language_model` are orders of magnitude below the
rungs that ignite; under (b) they are comparable and the parameters are still
not moving.

This used to carry a third reading off `log_logit_scale`'s gradient -- its sign
consistency step by step, which was the sharpest single number here. It was
dropped when the parameter was, and the parameter came back on 2026-08-31 while
the reading did not; the `scale_grad` columns are still gone. Restoring them is
the obvious next thing to do here, and the question about any other lone scalar
-- `log_score_scale`, `contrast_gate` -- would be
asked the same way, and this script does not currently ask it.

Rung 13 is the control that makes this worth running. It has the *same speaker
as rung 09* -- `SenderTransformerLM`, 320 wide, four layers -- and differs only
in the listener's discriminator, and it ignites. So whatever kills 09 is not
the speaker's architecture on its own, and a per-group gradient norm should say
whether the listener is failing to supply signal or the speaker is failing to
use it.

    python scripts/ignition_audit.py \
        --configs experiments/ablation/configs/07_shapeworld_sender_contrast.toml \
                  experiments/ablation/configs/09_shapeworld_sender_transformer_lm.toml \
                  experiments/ablation/configs/13_shapeworld_attention_discriminator.toml \
        --steps 800 --out results/ignition_audit

Real data, real optimiser, real `accumulator_steps`, and the same bf16 autocast
and `GradScaler` as `train.py`, so the trajectory is the run's trajectory and
not a proxy for it -- gradients are read after `scaler.unscale_`, so the norms
are in loss units and comparable across configs. Nothing is checkpointed and no
`metrics.csv` is touched; one `<config stem>.csv` per config goes to `--out`.

The ablation runs 156 optimiser steps per epoch, so `--steps` converts at
roughly that. 800 stays inside the plateau, which is where the two readings
above already differ; 1800 reaches epoch ~11 and so covers the epoch-10
ignition in rungs 05 and 07, which is the useful setting if you want to watch
the sign flip rather than infer it. At the ablation's measured 3.3-6.5 minutes
an epoch, 1800 steps is roughly 40-75 minutes for one config.

`compile` is honoured exactly as `train.py` honours it -- the two feature
models, compiled in place after building. `Module.compile()` leaves the
parameter objects and their names alone, so reading `.grad` by name every step
still works, and the ViT backbones are ~90% of a rung's device time. Off it,
this script would be roughly 3x slower than the run it is auditing rather than
comparable to it.
"""

import argparse
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import data.loader  # noqa: E402
import models.builder  # noqa: E402
import parse_config  # noqa: E402
import paths  # noqa: E402


# Logical parameter groups, in the order they appear in the CSV. A parameter is
# assigned to the *first* prefix that matches, so a more specific entry has to
# precede the module it lives on.
GROUPS = OrderedDict(
    [
        ("sender_vision", ("sender.feat_model",)),
        ("sender_prototyper", ("sender.prototyper",)),
        ("sender_contrast", ("sender.contrast",)),
        ("sender_language_model", ("sender.language_model",)),
        ("receiver_vision", ("receiver.feature_model",)),
        ("receiver_token_embedding", ("receiver.token_embedding",)),
        ("receiver_language_model", ("receiver.language_model",)),
        ("receiver_discriminator", ("receiver.discriminator",)),
    ]
)


def group_of(name):
    """The first group whose prefix matches `name`, or None."""
    for group, prefixes in GROUPS.items():
        if any(name.startswith(prefix) for prefix in prefixes):
            return group
    return None


def resolve_dataset_paths(config):
    """
    Rewrite the dataset keys onto fast storage exactly as `train.py`'s main
        block does, and for the same reason: `get_config` has already branched
        on the original string to choose the birds defaults, so this cannot
        happen any earlier.
    """
    emcomgen_data = paths.data_fast_storage() / "emcomgen" / "data"
    for key in ("dataset", "ref_dataset"):
        if key in config["data"]:
            basename = Path(config["data"][key]).name
            config["data"][key] = str(emcomgen_data / basename)
    return config


def grad_norms(pair):
    """
    Per-group gradient norm, as the square root of the summed squares over the
        group's parameters. Called after `scaler.unscale_`, so these are the
        gradients the optimiser is about to act on, in loss units.

    A parameter whose `.grad` is None contributes nothing, which is the honest
        reading: the backward pass never reached it.
    """
    squares = {group: 0.0 for group in GROUPS}
    for name, parameter in pair.named_parameters():
        group = group_of(name)
        if group is None or parameter.grad is None:
            continue
        squares[group] += parameter.grad.detach().float().pow(2).sum().item()
    return {group: math.sqrt(value) for group, value in squares.items()}


def diagnostics(pair):
    """
    The same speaker columns `train.py` writes to `metrics.csv`, read at the
        same point in the step, so a row here can be compared against a row
        there directly. NaN where the architecture does not have the quantity.
    """
    language_model = pair.sender.language_model
    prototyper = pair.sender.prototyper
    contrast = pair.sender.contrast
    unmeasured = float("nan")

    return {
        "realised_survival": language_model.realised_survival,
        "logit_spread": language_model.logit_spread,
        "logit_scale": (
            language_model.logit_scale.item()
            if language_model.normalises_logits else unmeasured
        ),
        "pool_effective_examples": getattr(
            prototyper, "pool_effective_examples", unmeasured
        ),
        "pool_score_norm": getattr(prototyper, "pool_score_norm", unmeasured),
        "polarity_separation": getattr(
            language_model, "polarity_separation", unmeasured
        ),
        "contrast_gate": (
            contrast.contrast_gate.item() if contrast is not None else unmeasured
        ),
    }


def audit(config_path, steps, seed, out_dir, device):
    """
    Train one rung from scratch for `steps` optimiser steps, recording the
        gradient at each one.

    Returns the path of the CSV written.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = parse_config.get_config(config_path)
    config = resolve_dataset_paths(config)
    config["resume"] = False

    dataloaders = data.loader.load_dataloaders(config)
    built = models.builder.build_models(dataloaders, config)
    pair, optimiser = built["pair"], built["optimiser"]

    # The two feature models only, in place, matching `train.py`. See the
    #     module docstring for why this is safe to do while reading `.grad`.
    if config["compile"]:
        pair.sender.feat_model.compile()
        pair.receiver.feature_model.compile()

    accumulator_steps = config["optimiser"]["accumulator_steps"]
    reference_game_xent = config["reference_game_xent"]
    bce = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    # Disabled off-GPU, where there is no fp16 overflow to guard against and
    #     `unscale_` has nothing to undo.
    scaler = GradScaler(enabled=(device.type == "cuda"))

    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    stem = Path(config_path).stem
    out_path = Path(out_dir) / f"{stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = (
        ["step", "loss", "acc"]
        + [f"grad_{group}" for group in GROUPS]
        + sorted(diagnostics(pair))
    )

    pair.train()
    taken = 0
    micro = 0

    with open(out_path, "w") as handle:
        handle.write(",".join(columns) + "\n")

        optimiser.zero_grad(set_to_none=True)

        # `steps` counts optimiser steps, so the epoch loop keeps handing out
        #     batches until that many have been taken.
        while taken < steps:
            for batch in dataloaders["train"]:
                spk_inp, spk_y, lis_inp, lis_y, _true_lang, _md, _idx = batch

                if dataloaders["train"].dataset.name == "shapeworld":
                    spk_inp = spk_inp.float() / 255
                    lis_inp = lis_inp.float() / 255
                else:
                    spk_inp = spk_inp.float()
                    lis_inp = lis_inp.float()

                spk_inp = spk_inp.to(device)
                spk_y = spk_y.float().to(device)
                lis_inp = lis_inp.to(device)
                lis_y = lis_y.float().to(device)

                with autocast(device_type=device.type, dtype=autocast_dtype):
                    lang, _concepts = pair.sender(spk_inp, spk_y)
                    scores = pair.receiver(lis_inp, lang)

                    if reference_game_xent:
                        midpoint = scores.shape[1] // 2
                        selected = torch.cat(
                            (scores[:, :1], scores[:, midpoint:]), 1
                        )
                        targets = torch.zeros(
                            scores.shape[0], dtype=torch.int64, device=device
                        )
                        loss = xent(selected, targets)
                        accuracy = (selected.argmax(1) == 0).float().mean().item()
                    else:
                        loss = bce(scores, lis_y)
                        predicted = (scores > 0).float()
                        accuracy = (predicted == lis_y).float().mean().item()

                scaler.scale(loss / accumulator_steps).backward()
                micro += 1

                if micro % accumulator_steps != 0:
                    continue

                # Unscale before reading anything: `GradScaler` multiplies the
                #     loss by a factor that changes over the run, so scaled
                #     gradients are not comparable step to step, let alone
                #     across configs.
                scaler.unscale_(optimiser)

                norms = grad_norms(pair)
                measured = diagnostics(pair)

                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

                taken += 1

                row = (
                    [taken, loss.item(), accuracy]
                    + [norms[group] for group in GROUPS]
                    + [measured[key] for key in sorted(measured)]
                )
                handle.write(",".join(f"{value}" for value in row) + "\n")

                if taken % 25 == 0:
                    handle.flush()
                    print(
                        f"  {stem} step {taken:>5}/{steps}  "
                        f"loss={loss.item():.4f}  "
                        f"speaker_grad={norms['sender_language_model']:.3e}  "
                        f"survival={measured['realised_survival']:.3f}",
                        flush=True,
                    )

                if taken >= steps:
                    break

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Record the gradient reaching the speaker over the plateau, to "
            "separate 'no gradient arrives' from 'gradient arrives and "
            "cancels'."
        )
    )
    parser.add_argument(
        "--configs", required=True, nargs="+", type=str,
        help="One or more experiment TOMLs, audited in turn."
    )
    parser.add_argument(
        "--steps", type=int, default=800,
        help="Optimiser steps per config. 156 steps is one ablation epoch."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed, matching train.py's --seed so a row here lines up with a "
             "row in that seed's metrics.csv."
    )
    parser.add_argument(
        "--out", type=str, default="results/ignition_audit",
        help="Directory for the per-config CSVs."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(
            "WARNING: no GPU visible. This runs the real model on real data "
            "and will be very slow on CPU.",
            flush=True,
        )

    for config_path in args.configs:
        print(f"\n=== {config_path} ===", flush=True)
        written = audit(config_path, args.steps, args.seed, args.out, device)
        print(f"  wrote {written}", flush=True)


if __name__ == "__main__":
    main()

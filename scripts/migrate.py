#!/usr/bin/env python3
"""
Rename result directories that a config *reordering* has orphaned.

``train.py`` writes each run to ``<output_root>/<experiment>/<stem>_seed<N>/``,
where ``stem`` is the config filename without ``.toml``, and
``job_utils.get_incomplete_jobs`` looks for exactly that path when it decides
what still needs running. So renaming a config renames the directory the
results are expected in, and a finished run silently reads as missing.

That is what happens when a sweep is renumbered -- ``01_birds_1e-5.toml``
becomes ``06_birds_1e-5.toml`` and the ``01_birds_1e-5_seed0`` directory on
disk no longer corresponds to any config. The runs are fine; only the number
in front of them is stale.

This script fixes exactly that case and no other. A stale directory is
migrated only when stripping the leading ``NN_`` from its stem leaves something
that matches a current config's stem stripped the same way, and when that match
is unique. Anything else -- a directory whose identity has no config at all, or
one that could be two different configs -- is reported and left alone, because
guessing there would be attaching real results to the wrong arm.

Renames are staged through temporary names so that a pure permutation (an arm
moving into a slot another arm is vacating) cannot collide part-way through.

Dry run by default; pass ``--apply`` to move anything::

    python scripts/migrate.py                          # show the plan
    python scripts/migrate.py --experiment lr_sweep_3_attention_prototyper
    python scripts/migrate.py --apply

See docs/training.md.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import toml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

import paths  # noqa: E402  (needs the sys.path line above)

# A result directory is `<config stem>_seed<N>`.
RESULT_DIR = re.compile(r"^(?P<stem>.+)_seed(?P<seed>\d+)$")

# The ordering prefix a renumbering moves, and nothing else in the name.
ORDER_PREFIX = re.compile(r"^\d+_")


def identity(stem: str) -> str:
    """The part of a config stem that a renumbering leaves alone.

    ``05_birds_2e-4`` and ``10_birds_2e-4`` are the same arm in two slots, so
    both have the identity ``birds_2e-4``. A stem with no numeric prefix is its
    own identity.
    """
    return ORDER_PREFIX.sub("", stem)


def current_stems(experiment: str) -> List[str]:
    """Config stems the repo currently defines for ``experiment``, sorted."""
    config_folder = REPO_ROOT / "experiments" / experiment / "configs"
    return sorted(p.stem for p in config_folder.glob("*.toml"))


def experiments_with_configs() -> List[str]:
    """Every experiment folder in the repo that has configs in it."""
    root = REPO_ROOT / "experiments"
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and any((d / "configs").glob("*.toml"))
    )


def expected_epochs(experiment: str, stem: str) -> int:
    """``[scheduler].epochs`` for one arm, read as ``job_utils`` reads it."""
    config = toml.load(
        REPO_ROOT / "experiments" / experiment / "configs" / f"{stem}.toml"
    )
    default = toml.load(REPO_ROOT / "DEFAULT.toml")["scheduler"]["epochs"]
    return config.get("scheduler", {}).get("epochs", default)


def metrics_rows(run_dir: Path) -> Optional[int]:
    """Rows in ``run_dir/metrics.csv``, or ``None`` if there is no such file."""
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return None
    with open(metrics, "r") as f:
        return sum(1 for _ in csv.DictReader(f))


def plan_experiment(
    experiment: str, results_dir: Path
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, str]]]:
    """Work out what to rename under one experiment's results directory.

    Returns ``(renames, skipped)``, where ``renames`` are ``(source, target)``
    pairs and ``skipped`` are ``(directory, reason)`` pairs for everything the
    script will not touch. Nothing is moved here.
    """
    stems = current_stems(experiment)

    # Identity -> current stem. A folder with two configs of the same identity
    # is not a reordering, so those identities are dropped as ambiguous.
    by_identity: Dict[str, List[str]] = {}
    for stem in stems:
        by_identity.setdefault(identity(stem), []).append(stem)

    renames: List[Tuple[Path, Path]] = []
    skipped: List[Tuple[Path, str]] = []

    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        match = RESULT_DIR.match(run_dir.name)
        if not match:
            skipped.append((run_dir, "not a `<stem>_seed<N>` directory"))
            continue

        stem, seed = match.group("stem"), match.group("seed")
        if stem in stems:
            continue  # already where the repo expects it

        candidates = by_identity.get(identity(stem), [])
        if not candidates:
            skipped.append(
                (run_dir, f"no current config has the identity `{identity(stem)}`")
            )
            continue
        if len(candidates) > 1:
            skipped.append(
                (run_dir, "identity matches " + ", ".join(f"`{c}`" for c in candidates))
            )
            continue

        renames.append((run_dir, results_dir / f"{candidates[0]}_seed{seed}"))

    # A target that already exists is only safe if it is itself being moved out
    # of the way in this same plan. Anything else is a real collision, and
    # overwriting it would destroy a run.
    sources = {src for src, _ in renames}
    checked: List[Tuple[Path, Path]] = []
    for src, dst in renames:
        if dst.exists() and dst not in sources:
            skipped.append((src, f"`{dst.name}` already exists and is not being moved"))
        else:
            checked.append((src, dst))

    return checked, skipped


def apply_renames(renames: List[Tuple[Path, Path]]) -> None:
    """Rename in two passes, so a permutation cannot collide half-way.

    Every source goes to a temporary name first, then each temporary name to
    its target. Without this, moving ``01_birds`` to ``06_birds`` while
    ``06_shapeworld`` moves to ``01_shapeworld`` is safe only because the
    dataset is in the name -- and that is a property of today's filenames, not
    something to depend on.
    """
    staged: List[Tuple[Path, Path]] = []
    for src, dst in renames:
        tmp = src.with_name(f".migrate-tmp-{src.name}")
        if tmp.exists():
            raise FileExistsError(f"stale temporary directory in the way: {tmp}")
        src.rename(tmp)
        staged.append((tmp, dst))
    for tmp, dst in staged:
        tmp.rename(dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="rename result directories orphaned by a config renumbering"
    )
    parser.add_argument(
        "--experiment", "-e",
        action="append",
        default=None,
        help="limit to this experiment; repeatable (default: all of them)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually rename (default is a dry run)",
    )
    args = parser.parse_args()

    output_root = paths.output_root()
    if not output_root.exists():
        print(f"No results root at {output_root}; nothing to do.")
        return 0

    experiments = args.experiment or experiments_with_configs()

    total_renames = 0
    total_skipped = 0
    short_runs: List[str] = []

    for experiment in experiments:
        config_folder = REPO_ROOT / "experiments" / experiment / "configs"
        if not any(config_folder.glob("*.toml")):
            print(f"{experiment}: no configs in the repo -- skipped.")
            continue

        results_dir = output_root / experiment
        if not results_dir.is_dir():
            continue  # never run; nothing on disk to migrate

        renames, skipped = plan_experiment(experiment, results_dir)
        if not renames and not skipped:
            continue

        print(f"\n{experiment}  ({results_dir})")
        for src, dst in renames:
            print(f"  {src.name}  ->  {dst.name}")
        for run_dir, reason in skipped:
            print(f"  LEFT ALONE  {run_dir.name}: {reason}")

        if args.apply and renames:
            apply_renames(renames)
            print(f"  renamed {len(renames)}")

        total_renames += len(renames)
        total_skipped += len(skipped)

        # Informational, and unrelated to the renaming: a run whose metrics.csv
        # is shorter than its config's `epochs` reads as incomplete, which is
        # what happens when a budget is raised after the fact. That is a resume
        # and not a re-run -- `train.py` restores `checkpoint_last.pt`, starts
        # at the epoch it records and appends to the existing metrics.csv -- so
        # only the missing epochs cost anything. It is worth seeing the list
        # before submitting all the same.
        for stem in current_stems(experiment):
            expected = expected_epochs(experiment, stem)
            for run_dir in sorted(results_dir.glob(f"{stem}_seed*")):
                rows = metrics_rows(run_dir)
                if rows is not None and rows < expected:
                    short_runs.append(
                        f"  {experiment}/{run_dir.name}: "
                        f"{rows} epochs on disk, config wants {expected}"
                    )

    if total_renames == 0 and total_skipped == 0:
        print("Every result directory already matches a current config.")
    elif not args.apply and total_renames:
        print(f"\nDry run: {total_renames} would be renamed. Pass --apply to do it.")

    if short_runs:
        print(
            "\nSeparately, these runs are shorter than their config's `epochs`, "
            "so `job_utils` will treat them as incomplete. They resume from "
            "`checkpoint_last.pt` rather than starting over, so only the "
            "missing epochs cost anything:"
        )
        for line in short_runs:
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

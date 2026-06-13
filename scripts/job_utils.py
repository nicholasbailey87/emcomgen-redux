#!/usr/bin/env python3
"""
Because we typically want to set up one SLURM job per repeat,
    per configuration, but the ideal etiquette is to send one
    request encompassing all jobs (using a SLURM array to
    spawn multiple queue items), we need

    * To be able to generate a SLURM array from all the configs
        and repeats defined for an experiment
    * To be able to identify the config name and repeat number
        for each item in a SLURM array

Here are some utils for that.

Configs are TOML, and a job is "complete" when its ``metrics.csv`` exists with
at least ``[scheduler].epochs`` rows (emcomgen writes one row per epoch).
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import toml


def enumerate_jobs(experiment: str) -> list:
    config_folder = Path("experiments") / experiment / "configs"
    config_files = sorted(config_folder.glob("*.toml"))
    configs_with_repeats = []
    max_repeats = 0
    for config_file in config_files:
        repeats = toml.load(config_file)["experiment"]["repeats"]
        configs_with_repeats.append((config_file, repeats))
        max_repeats = max(max_repeats, repeats)
    jobs = []
    for repeat in range(max_repeats):
        for config_file, repeats in configs_with_repeats:
            if repeat < repeats:
                jobs.append((config_file.name, repeat))
    return jobs


def describe_index(experiment: str, index: int) -> Tuple[str, int]:
    jobs = enumerate_jobs(experiment)
    return jobs[index]


def get_incomplete_jobs(experiment: str, output_root: str) -> List[int]:
    jobs = enumerate_jobs(experiment)
    incomplete = []
    for i, (config_file, seed) in enumerate(jobs):
        config_path = Path("experiments") / experiment / "configs" / config_file
        config = toml.load(config_path)
        expected_epochs = config["scheduler"]["epochs"]
        config_stem = Path(config_file).stem
        results_path = (
            Path(output_root).expanduser()
            / experiment
            / f"{config_stem}_seed{seed}"
            / "metrics.csv"
        )
        if not results_path.exists():
            incomplete.append(i)
            continue
        # emcomgen writes one row per completed epoch. A finished run has at
        # least `expected_epochs` rows; a crash mid-run leaves fewer.
        with open(results_path, "r") as f:
            reader = csv.DictReader(f)
            n_rows = sum(1 for _ in reader)
        if n_rows < expected_epochs:
            incomplete.append(i)
    return incomplete


def main():
    parser = argparse.ArgumentParser(
        description="work out the required length for a SLURM array, "
        "or pinpoint the run to which a SLURM array index refers"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Name of the experiment"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--count", "-c",
        action="store_true",
        help="Print total number of jobs"
    )
    group.add_argument(
        "--index", "-i",
        type=int,
        help="Get config and seed for this job index"
    )
    group.add_argument(
        "--incomplete",
        action="store_true",
        help="Print comma-separated list of incomplete job indices"
    )

    parser.add_argument(
        "--output-root",
        type=str,
        help="Root output directory (required with --incomplete)"
    )

    args = parser.parse_args()

    jobs = enumerate_jobs(args.experiment)

    if args.count:
        print(len(jobs))
    elif args.incomplete:
        if not args.output_root:
            raise ValueError("--output-root is required with --incomplete")
        incomplete = get_incomplete_jobs(args.experiment, args.output_root)
        print(",".join(str(i) for i in incomplete))
    else:
        if args.index < 0 or args.index >= len(jobs):
            raise ValueError(
                f"Index {args.index} out of range [0, {len(jobs)-1}]"
            )

        config_file, seed = describe_index(args.experiment, args.index)
        print(f"{config_file} {seed}")


if __name__ == "__main__":
    main()

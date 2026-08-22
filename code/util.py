"""
utils
"""


import os
import subprocess
import json
import importlib.metadata
import toml
import warnings

import data.language
from collections import defaultdict

# Dependencies whose version changes the model, and so has to be recorded
#     alongside the results. See docs/training.md.
PINNED_DEPENDENCIES = ("broccoli-ml", "gradboard", "torch", "torchvision")


def current_git_hash():
    """
    The hash of the latest commit in this repository, and whether the working
        tree is dirty. ``(None, None)`` if git could not be reached.
    """
    unstaged_changes = False
    try:
        subprocess.check_output(["git", "diff-index", "--quiet", "HEAD", "--"])
    except subprocess.CalledProcessError as grepexc:
        if grepexc.returncode == 1:
            warnings.warn("Running experiments with unstaged changes.")
            unstaged_changes = True
    except FileNotFoundError:
        warnings.warn("Git not found")
    try:
        git_hash = (
            subprocess.check_output(["git", "describe", "--always"])
            .strip()
            .decode("utf-8")
        )
        return git_hash, unstaged_changes
    except subprocess.CalledProcessError:
        return None, None


class Statistics:
    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, batch_size=1, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v, batch_size)

    def averages(self):
        """Averages of every meter, always as ``float``."""
        metrics = {m: vs.avg for m, vs in self.meters.items()}
        metrics = {
            m: v if isinstance(v, float) else v.item() for m, v in metrics.items()
        }
        return metrics

    def __str__(self):
        meter_str = ", ".join(f"{k}={v}" for k, v in self.meters.items())
        return f"Statistics({meter_str})"


class AverageMeter:
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"AverageMeter(mean={self.avg:f}, count={self.count:d})"

    def __repr__(self):
        return str(self)


def dependency_versions(packages=PINNED_DEPENDENCIES):
    """
    Installed version of every dependency that can change the model. See
        docs/training.md.

    Returns:
        Package name -> ``{"version": str}``, plus ``"commit"`` where the package
            was installed from a VCS. Values are always strings, since this ends
            up in ``config.toml`` and TOML cannot represent None.
    """
    versions = {}
    for package in packages:
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = {"version": "not installed"}
            continue

        record = {"version": distribution.version}

        # `read_text` returns None for a missing file, but raises for a
        #     distribution installed without a metadata directory.
        try:
            direct_url = distribution.read_text("direct_url.json")
        except OSError:
            direct_url = None

        if direct_url:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
            if commit:
                record["commit"] = commit

        versions[package] = record

    return versions


def save_args(args_dict, exp_dir):
    # Output to both json and toml to support original and new functionality.
    args_dict["git_hash"], args_dict["git_unstaged_changes"] = current_git_hash()
    args_dict["dependency_versions"] = dependency_versions()
    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4, separators=(",", ": "), sort_keys=True)
    with open(os.path.join(exp_dir, "config.toml"), "w") as f:
        toml.dump(args_dict, f)


def update_with_prefix(d, new_d, prefix):
    d.update({f"{prefix}_{k}": v for k, v in new_d.items()})


def to_emergent_text(idxs, join=False, eos=None):
    """`eos` is inclusive: the terminator is kept and the row stops after it."""
    def tokenize(lang):
        toks = []
        for i in lang:
            i_item = i.item()
            toks.append(str(i_item))
            if eos is not None and i_item == eos:
                break
        return toks

    return data.language.rows_to_text(idxs, tokenize, join=join)
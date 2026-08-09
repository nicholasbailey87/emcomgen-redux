"""
utils
"""


import os
import subprocess
import json
import importlib.metadata
import toml
import warnings
from collections import defaultdict

# Dependencies whose version changes the model, and so has to be recorded
#     alongside the results. broccoli and gradboard are installed from git.
PINNED_DEPENDENCIES = ("broccoli-ml", "gradboard", "torch", "torchvision")


def current_git_hash():
    """
    Get the hash of the latest commit in this repository. Does not account for unstaged changes.
    Returns
    -------
    git_hash : ``str``, optional
        The string corresponding to the current git hash if known, else ``None`` if something failed.
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
        """
        Compute averages from meters. Handle tensors vs floats (always return a
        float)

        Parameters
        ----------
        meters : Dict[str, util.AverageMeter]
            Dict of average meters, whose averages may be of type ``float`` or ``torch.Tensor``

        Returns
        -------
        metrics : Dict[str, float]
            Average value of each metric
        """
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
    Record the installed version of every dependency that can change the model,
        so a finished run says which architecture produced it.

    `current_git_hash` covers this repository only, which is not enough: the
        model is mostly broccoli's, and broccoli's version has moved
        underneath this repository before without any commit here to show for
        it. For anything installed from git, pip records the resolved commit
        in the distribution's `direct_url.json`, so the exact source is
        recoverable even if the requirement was written as a branch or a tag.

    Returns
    -------
    versions : ``dict``
        Package name -> ``{"version": str}``, plus ``"commit"`` where the
        package was installed from a VCS. Values are always strings, since
        this ends up in `config.toml` and TOML cannot represent None.
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
        #     distribution installed in a layout without a metadata directory.
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
    # Note: no longer need `args_dict = vars(args)` as args will now already
    # be a dict, see train.py. Also we output to moth json and toml to support
    # original and new functionality.
    args_dict["git_hash"], args_dict["git_unstaged_changes"] = current_git_hash()
    args_dict["dependency_versions"] = dependency_versions()
    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4, separators=(",", ": "), sort_keys=True)
    with open(os.path.join(exp_dir, "config.toml"), "w") as f:
        toml.dump(args_dict, f)


def update_with_prefix(d, new_d, prefix):
    d.update({f"{prefix}_{k}": v for k, v in new_d.items()})


def to_emergent_text(idxs, join=False, eos=None):
    texts = []
    for lang in idxs:
        toks = []
        for i in lang:
            i_item = i.item()
            i = str(i_item)
            toks.append(i)
            if eos is not None and i_item == eos:
                break
        if join:
            texts.append(" ".join(toks))
        else:
            texts.append(toks)
    return texts
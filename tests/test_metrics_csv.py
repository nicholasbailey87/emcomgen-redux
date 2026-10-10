"""
`train.append_metrics_row`, which appends each epoch's row to `metrics.csv` and
refuses a row whose columns do not match the header already on disk.

Runnable without pytest:  python tests/test_metrics_csv.py
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

import _bootstrap  # noqa: F401
import train


def test_rows_land_under_the_header_in_its_order(tmp_path):
    path = str(tmp_path / "metrics.csv")
    train.append_metrics_row(path, {"epoch": 0, "a": 1.0, "b": 2.0})
    train.append_metrics_row(path, {"b": 4.0, "epoch": 1, "a": 3.0})

    frame = pd.read_csv(path)
    assert list(frame.columns) == ["epoch", "a", "b"]
    assert frame.to_dict("list") == {
        "epoch": [0, 1], "a": [1.0, 3.0], "b": [2.0, 4.0]
    }


def test_a_metric_missing_from_a_later_epoch_raises(tmp_path):
    """The header was written by this same process in epoch 0, not a resume."""
    path = str(tmp_path / "metrics.csv")
    train.append_metrics_row(path, {"epoch": 0, "a": 1.0, "b": 2.0})

    with pytest.raises(RuntimeError, match=r"On disk but not measured now: \['b'\]"):
        train.append_metrics_row(path, {"epoch": 1, "a": 3.0})

    assert len(pd.read_csv(path)) == 1


def test_a_metric_new_in_a_later_epoch_raises(tmp_path):
    path = str(tmp_path / "metrics.csv")
    train.append_metrics_row(path, {"epoch": 0, "a": 1.0})

    with pytest.raises(RuntimeError, match=r"Measured now but not on disk: \['b'\]"):
        train.append_metrics_row(path, {"epoch": 1, "a": 3.0, "b": 4.0})

    assert len(pd.read_csv(path)) == 1


if __name__ == "__main__":
    for test in (
        test_rows_land_under_the_header_in_its_order,
        test_a_metric_missing_from_a_later_epoch_raises,
        test_a_metric_new_in_a_later_epoch_raises,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
    print("ok")

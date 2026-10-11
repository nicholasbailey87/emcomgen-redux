"""
`parse_config.validate_config`'s checks on four keys that `train.py` and the
data loaders read as counts or fractions.

Runnable without pytest:  python tests/test_config_counts.py

    * `[optimiser] accumulator_steps` -- a positive integer. `train.py` divides
      each micro-batch loss by it, so 0 was a `ZeroDivisionError` mid-epoch.
    * `bn_calibration_batches` -- a non-negative integer, compared against a
      batch index in `train.calibrate_batch_norm`.
    * `[data] percent_novel` -- a number in [0, 1].
    * `receiver_reset_interval` -- a non-negative whole number of epochs. It is
      used as `epoch % interval`, so a fractional value would run but reset at
      only the epochs that are whole multiples of it.

None of the four can be *missing* from a config `get_config` returns: it merges
DEFAULT.toml first, which sets all four, and a TOML file cannot delete a key.
So the checks are about type and range, and a `None` is rejected as "not a
number" for the case where `validate_config` is handed a dict built elsewhere.

TOML reads `1` as an int and `1.0` as a float, so the two fractional keys accept
either; `bool` is an `int` subclass in Python and is rejected everywhere.
"""

import _bootstrap  # noqa: F401

import math
import os
import tempfile

import pytest

import parse_config


def _config(dataset="../data/shapeworld"):
    """`get_config` for one dataset, so the `[birds.*]` overlay is applied."""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(f'name = "test"\n[data]\ndataset = "{dataset}"\n')
        path = f.name
    try:
        return parse_config.get_config(path)
    finally:
        os.unlink(path)


def _set(config, key, value):
    """Set a dotted key (`table.key` or a top-level `key`) on `config`."""
    if "." in key:
        table, key = key.split(".")
        config[table][key] = value
    else:
        config[key] = value


BAD_COUNT = [None, "4", 4.0, True, False]

CASES = [
    # (key, accepted, rejected)
    ("optimiser.accumulator_steps", [1, 2, 4], [0, -1, 2.5] + BAD_COUNT),
    ("bn_calibration_batches", [0, 1, 8], [-1, 2.5] + BAD_COUNT),
    (
        "data.percent_novel",
        [0, 1, 0.0, 0.5, 1.0],
        [-0.1, 1.1, math.nan, math.inf, None, "1.0", True, False],
    ),
    (
        "receiver_reset_interval",
        [0, 0.0, 1, 5, 5.0],
        [-1, -1.0, 0.5, 1.5, math.inf, math.nan, None, "5", True, False],
    ),
]


@pytest.mark.parametrize(
    "key,value",
    [(key, value) for key, accepted, _ in CASES for value in accepted],
)
def test_a_valid_value_is_accepted(key, value):
    config = _config()
    _set(config, key, value)
    parse_config.validate_config(config)


@pytest.mark.parametrize(
    "key,value",
    [(key, value) for key, _, rejected in CASES for value in rejected],
)
def test_an_invalid_value_is_rejected(key, value):
    config = _config()
    _set(config, key, value)
    with pytest.raises(parse_config.InvalidConfig, match=key.split(".")[-1]):
        parse_config.validate_config(config)


@pytest.mark.parametrize("dataset", ["../data/shapeworld", "../data/cub"])
def test_the_defaults_validate_on_both_datasets(dataset):
    """
    Including CUB's `[birds.optimiser] accumulator_steps = 1`, and
        `receiver_reset_interval = 0.0`, which DEFAULT.toml writes as a float.
    """
    config = _config(dataset)
    assert config["receiver_reset_interval"] == 0.0
    assert config["optimiser"]["accumulator_steps"] == (
        1 if dataset.endswith("cub") else 4
    )


if __name__ == "__main__":
    for key, accepted, rejected in CASES:
        for value in accepted:
            test_a_valid_value_is_accepted(key, value)
        for value in rejected:
            test_an_invalid_value_is_rejected(key, value)
    for dataset in ["../data/shapeworld", "../data/cub"]:
        test_the_defaults_validate_on_both_datasets(dataset)
    print("ok")

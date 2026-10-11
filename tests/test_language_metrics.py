"""
Tests for ``train.compute_language_metrics``: the per-game topsim report an eval
pass writes into ``metrics.csv``.

The distances and the bootstrap are tested in tests/test_emergence.py. What is
checked here is the wiring -- that a synthetic eval pass produces every column,
that each interval brackets its value, that ``test_avg`` keeps only the values,
and that the seed alone fixes the output.
"""


import numpy as np

import _bootstrap  # noqa: F401
import emergence as E
import train


def _eval_pass(n_games=200, seed=0):
    """Games whose message, embeddings and concept vector all follow a formula."""
    rng = np.random.default_rng(seed)
    colours = ["red", "blue", "green", "white"]
    shapes = ["square", "circle", "triangle"]
    symbol_table = rng.standard_normal((4 + 20, 16))
    colour_vectors = rng.standard_normal((len(colours), 24))
    shape_vectors = rng.standard_normal((len(shapes), 24))

    messages, embeddings, concepts, keys = [], [], [], []
    for _ in range(n_games):
        colour = rng.integers(len(colours))
        shape = rng.integers(len(shapes))
        keys.append(f"and {colours[colour]} {shapes[shape]}")
        # Mostly compositional, with some noise symbols so games of one concept
        # do not all say the same thing.
        message = [4 + colour, 8 + shape] + list(rng.integers(4, 24, size=3))
        messages.append(message)
        embeddings.append(
            symbol_table[message] + 0.1 * rng.standard_normal((5, 16))
        )
        concepts.append(
            colour_vectors[colour] + shape_vectors[shape]
            + 0.1 * rng.standard_normal(24)
        )
    return messages, embeddings, np.stack(concepts), keys


def _metrics(ground_truth_meaning=True, seed=0, n_games=200):
    messages, embeddings, concepts, keys = _eval_pass(n_games)
    return train.compute_language_metrics(
        messages,
        embeddings,
        concepts,
        keys,
        n_pairs=3000,
        replicates=200,
        seed=seed,
        ground_truth_meaning=ground_truth_meaning,
    )


def test_every_column_with_an_interval_that_brackets_it():
    metrics = _metrics()
    assert set(metrics) == set(E.REPORT_KEYS) | {"s6_saturated"}
    assert 0.0 <= metrics["s6_saturated"] <= 1.0
    for key in E.REPORT_KEYS:
        if key.endswith(E.BOOTSTRAP_SUFFIXES):
            continue
        low, value, high = (
            metrics[f"{key}_low"], metrics[key], metrics[f"{key}_high"]
        )
        assert np.isfinite(value), key
        assert low <= value <= high, f"{key}: {low} <= {value} <= {high}"
        assert metrics[f"{key}_degenerate"] == 0.0, key


def test_cub_has_no_ground_truth_columns():
    metrics = _metrics(ground_truth_meaning=False)
    assert len(metrics) == 36 + 1
    assert not any(key.startswith("topsim_gt_") for key in metrics)


def test_the_seed_fixes_the_output():
    first, second = _metrics(seed=3), _metrics(seed=3)
    assert first.keys() == second.keys()
    for key in first:
        assert first[key] == second[key] or (
            np.isnan(first[key]) and np.isnan(second[key])
        ), key

    other = _metrics(seed=4)
    assert any(first[key] != other[key] for key in first)


def test_too_few_games_reports_nothing():
    assert _metrics(n_games=2) == {}


def test_split_average_keeps_values_and_drops_bootstrap_columns():
    """
    The mean of two splits' bounds is not an interval for the mean of their
    values, so `test_avg` carries the topsim values alone.
    """
    test, test_same = _metrics(seed=0), _metrics(seed=1)
    split_metrics = {key: [test[key], test_same[key]] for key in test}
    split_metrics["acc"] = [0.5, 0.7]

    average = train.average_splits(split_metrics)
    values = [
        key for key in E.REPORT_KEYS if not key.endswith(E.BOOTSTRAP_SUFFIXES)
    ]
    assert set(average) == set(values) | {"s6_saturated", "acc"}
    assert len(average) == 18 + 2
    assert np.isclose(average["acc"], 0.6)
    for key in values:
        assert np.isclose(average[key], (test[key] + test_same[key]) / 2), key

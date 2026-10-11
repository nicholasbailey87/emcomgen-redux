"""
`train.complete_partial_accumulation`, which makes the trailing partial
accumulation of an epoch a mean over its microbatches like any other step.

Each microbatch's loss is divided by `accumulator_steps` before `backward`. On a
full accumulation that is the mean; on the k < `accumulator_steps` microbatches
an epoch can end on, it was k / `accumulator_steps` of the mean until this
helper. These tests drive the helper through the same sequence `train.run`
does -- scaled backward per microbatch, rescale, `scaler.unscale_` -- under an
enabled `GradScaler`, and compare against the mean gradient computed directly.

Runnable without pytest:  python tests/test_gradient_accumulation.py
"""

import _bootstrap  # noqa: F401

import pytest
import torch
import torch.nn as nn
from torch.amp import GradScaler

import train

ACCUMULATOR_STEPS = 4

# Not 1.0, so a step that skipped or doubled the unscale would not pass.
INIT_SCALE = 1024.0


def _microbatches(n):
    generator = torch.Generator().manual_seed(0)
    return [
        (torch.randn(8, 5, generator=generator),
         torch.randn(8, 3, generator=generator))
        for _ in range(n)
    ]


def _model():
    torch.manual_seed(0)
    return nn.Linear(5, 3)


def _loss(model, x, y):
    return nn.functional.mse_loss(model(x), y)


def _accumulated_gradient(k):
    """The unscaled gradient `train.run` would clip after k microbatches."""
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = GradScaler("cpu", init_scale=INIT_SCALE)

    for x, y in _microbatches(k):
        scaler.scale(_loss(model, x, y) / ACCUMULATOR_STEPS).backward()

    train.complete_partial_accumulation(model, k, ACCUMULATOR_STEPS)
    scaler.unscale_(optimizer)
    return [p.grad.clone() for p in model.parameters()]


def _mean_gradient(k):
    """The mean over the same k microbatches, with no scaler involved."""
    model = _model()
    sum(_loss(model, x, y) for x, y in _microbatches(k)).div(k).backward()
    return [p.grad.clone() for p in model.parameters()]


@pytest.mark.parametrize("k", range(1, ACCUMULATOR_STEPS + 1))
def test_an_accumulation_of_k_microbatches_is_their_mean(k):
    """Partial (k < 4) and full (k = 4) alike."""
    for got, want in zip(_accumulated_gradient(k), _mean_gradient(k)):
        torch.testing.assert_close(got, want)


def test_a_full_accumulation_is_left_untouched():
    """The full path is a no-op, so every step but an epoch's last is as it was."""
    model = _model()
    for x, y in _microbatches(ACCUMULATOR_STEPS):
        (_loss(model, x, y) / ACCUMULATOR_STEPS).backward()
    before = [p.grad.clone() for p in model.parameters()]

    train.complete_partial_accumulation(
        model, ACCUMULATOR_STEPS, ACCUMULATOR_STEPS
    )

    for got, want in zip((p.grad for p in model.parameters()), before):
        assert torch.equal(got, want)


if __name__ == "__main__":
    for k in range(1, ACCUMULATOR_STEPS + 1):
        test_an_accumulation_of_k_microbatches_is_their_mean(k)
    test_a_full_accumulation_is_left_untouched()
    print("ok")

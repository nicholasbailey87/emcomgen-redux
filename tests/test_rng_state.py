"""
`train.rng_state` and `train.restore_rng_state`, which carry the generator
states through the checkpoint so that a resumed run draws what an uninterrupted
one would have.

Runnable without pytest:  python tests/test_rng_state.py
"""

import random

import numpy as np
import torch

import _bootstrap  # noqa: F401
import train


def _draw():
    """One draw from each generator the checkpoint covers."""
    draws = [
        torch.rand(4).tolist(),
        np.random.rand(4).tolist(),
        [random.random() for _ in range(4)],
    ]
    if torch.cuda.is_available():
        draws.append(torch.rand(4, device="cuda").tolist())
    return draws


def test_restored_state_repeats_the_draws():
    state = train.rng_state()
    first = _draw()
    train.restore_rng_state(state)
    assert _draw() == first


def test_restored_state_survives_a_checkpoint_round_trip(tmp_path):
    """Through `torch.save` and the `weights_only=False` load `train.py` uses."""
    path = tmp_path / "checkpoint.pt"
    torch.save({"rng_state": train.rng_state()}, path)
    first = _draw()
    train.restore_rng_state(
        torch.load(path, weights_only=False)["rng_state"]
    )
    assert _draw() == first


def test_a_checkpoint_without_rng_state_restores_nothing():
    """What `train.py` passes for a checkpoint written before these were saved."""
    before = train.rng_state()
    train.restore_rng_state({})
    after = train.rng_state()

    assert torch.equal(before["torch"], after["torch"])
    assert before["python"] == after["python"]
    assert before["numpy"][0] == after["numpy"][0]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert before["numpy"][2:] == after["numpy"][2:]


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_restored_state_repeats_the_draws()
    with tempfile.TemporaryDirectory() as tmp:
        test_restored_state_survives_a_checkpoint_round_trip(Path(tmp))
    test_a_checkpoint_without_rng_state_restores_nothing()
    print("ok")

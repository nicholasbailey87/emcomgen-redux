"""
Tests for `train.per_game_accuracy` and the `shuffled_message_acc` control.

Runnable without pytest:  python tests/test_listener_baseline.py

The listener does not need the message to score above chance. On
`shapeworld-post-silhouette-update.csv` it reached 0.588 train and 0.55
`test_same` on shape concepts while the speaker emitted **one** message for
every game in the split -- so most of what `acc` reported was not communication,
and there was no column that said so until the speaker had already died.

`shuffled_message_acc` is that column: the same candidates scored against
another game's message. In distribution, since it is a real message the speaker
produced, and carrying nothing about this game. The gap between `acc` and it is
what the channel buys; its level is the image-only route the channel has to beat
before it earns any gradient at all.

Two things have to hold or the number is not a control.

**It must be scored by the same code as the live accuracy.** A control computed
a second way would report every divergence between the two implementations as
communication. That is why `per_game_accuracy` was lifted out of the loop, and
it is the only structural change the diagnostic needed.

**The pairing must be a real derangement.** `torch.roll` by one and not
`randperm`: a permutation leaves about one game in `batch_size` holding its own
message, which biases the baseline upwards -- by ~3% at ShapeWorld's 32, which
is the same order as the effects being measured.
"""

import numpy as np
import pytest
import torch

import _bootstrap  # noqa: F401

import train


N_OBJECTS = 20  # ten positive, ten negative, as the listener is handed them
BATCH = 8


def _scores_and_labels(batch=BATCH, seed=0):
    """Labels in the layout `train.py` asserts: positives first."""
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(batch, N_OBJECTS, generator=generator)

    labels = torch.zeros(batch, N_OBJECTS)
    labels[:, : N_OBJECTS // 2] = 1.0
    return scores, labels


# ------------------------------------------------ 1. the shared scorer --

def test_bce_branch_thresholds_at_zero():
    """
    The per-candidate judgement: every score above zero is a "yes", and the
    accuracy is the fraction of candidates called right. The threshold is a
    fixed origin -- `score_bias` is what moves the scores onto it.
    """
    scores, labels = _scores_and_labels()

    expected = (
        ((scores > 0).float() == labels).float().mean(1).numpy()
    )
    result = train.per_game_accuracy(scores, labels, reference_game_xent=False)

    assert np.allclose(result, expected)
    assert result.shape == (BATCH,)


def test_xent_branch_picks_from_the_target_and_the_second_half():
    """
    The pick-one game: candidate 0 against everything past the midpoint, and the
    target is index 0 of that selection by construction.
    """
    scores, labels = _scores_and_labels(seed=1)
    midpoint = N_OBJECTS // 2

    selected = torch.cat((scores[:, :1], scores[:, midpoint:]), 1)
    expected = (selected.argmax(1) == 0).float().numpy()
    result = train.per_game_accuracy(scores, labels, reference_game_xent=True)

    assert np.allclose(result, expected)
    assert set(np.unique(result)) <= {0.0, 1.0}


def test_a_perfect_listener_scores_one_on_both_branches():
    """A shared sanity anchor, so neither branch can be right about nothing."""
    _, labels = _scores_and_labels()

    # Positive candidates high, negatives low: correct under either reading,
    # since candidate 0 is a positive and everything past the midpoint is not.
    perfect = torch.where(labels > 0, 10.0, -10.0)

    for xent in (False, True):
        assert train.per_game_accuracy(perfect, labels, xent).mean() == 1.0


# ------------------------------------------- 2. the ablation is a control --

def test_rolling_pairs_no_game_with_its_own_message():
    """
    The property that makes the baseline honest, and the reason this is not a
    `randperm`. Checked for every batch size a trailing partial batch can take.
    """
    for batch in range(2, 34):
        messages = torch.arange(batch)
        rolled = torch.roll(messages, 1, 0)

        assert not (rolled == messages).any(), batch
        assert sorted(rolled.tolist()) == sorted(messages.tolist())


def test_rolling_a_single_game_would_pair_it_with_itself():
    """
    Which is why `run` guards on `batch_size > 1`: without it a trailing batch
    of one would report the live accuracy as the baseline, and a split whose
    size is one more than a multiple of the batch size would be quietly wrong.
    """
    lone = torch.randn(1, 5)
    assert torch.equal(torch.roll(lone, 1, 0), lone)


def test_the_baseline_falls_when_the_message_carried_the_answer():
    """
    End to end on the arithmetic the column exists to do. A listener whose score
    is driven entirely by its own message reads 1.0 live and chance once the
    messages are rolled; one that ignores the message reads the same either way,
    which is the collapse the column is there to catch.
    """
    batch, labels = 16, torch.zeros(16, N_OBJECTS)
    labels[:, : N_OBJECTS // 2] = 1.0

    # One key per game. The listener answers correctly when the message it is
    # handed is this game's own and inverts otherwise, so it is perfect live and
    # right only where a rolled neighbour happens to share the key.
    generator = torch.Generator().manual_seed(2)
    message = torch.randint(0, 4, (batch, 1), generator=generator).float()
    correct = torch.where(labels > 0, 10.0, -10.0)

    def listen(given):
        matched = (given == message).expand_as(labels)
        return torch.where(matched, correct, -correct)

    live = train.per_game_accuracy(listen(message), labels, False).mean()
    rolled = train.per_game_accuracy(
        listen(torch.roll(message, 1, 0)), labels, False
    ).mean()

    assert live == 1.0
    assert rolled < 0.75

    # And a listener that ignores the message is unmoved by rolling it, which is
    # the reading that says the channel is carrying nothing -- the 2026-08-29
    # run's terminal state, where `acc` sat at 0.52 on a constant message.
    def listen_deaf(bits):
        del bits
        return torch.where(labels > 0, 10.0, -10.0)

    assert train.per_game_accuracy(
        listen_deaf(message), labels, False
    ).mean() == train.per_game_accuracy(
        listen_deaf(torch.roll(message, 1, 0)), labels, False
    ).mean()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

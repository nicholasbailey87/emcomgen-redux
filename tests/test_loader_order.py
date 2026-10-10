"""
Tests for the order `data.loader.load_dataloaders` reads each split in.

Runnable without pytest:  python tests/test_loader_order.py

Topsim samples its fixed `topsim_seed` pairs by position in the pass, so an eval
split read in a different order each epoch measures different game pairs each
epoch. `shuffled_message_acc` pairs each game with its batch neighbour's
message, so an eval split read in file order could pair games of one concept.
The eval splits therefore read one fixed permutation; `train` and the
`train_clean` calibration loader, which is read only up to
`bn_calibration_batches`, shuffle afresh every pass.
"""

import torch
from torch.utils.data import RandomSampler

import _bootstrap  # noqa: F401

import data.loader


SPLITS = ("train", "train_clean", "test", "test_same")
EVAL_SPLITS = ("test", "test_same")
N_ITEMS = 50


def _dataloaders(monkeypatch):
    monkeypatch.setattr(
        data.loader,
        "load",
        lambda config: {split: list(range(N_ITEMS)) for split in SPLITS},
    )
    config = {"data": {"batch_size": 4, "n_workers": 0}}
    return data.loader.load_dataloaders(config)


def _order(dataloader):
    return torch.cat(list(dataloader)).tolist()


def test_train_and_calibration_loaders_shuffle(monkeypatch):
    dataloaders = _dataloaders(monkeypatch)

    assert isinstance(dataloaders["train"].sampler, RandomSampler)
    assert isinstance(dataloaders["train_clean"].sampler, RandomSampler)


def test_eval_order_is_the_same_every_pass(monkeypatch):
    dataloaders = _dataloaders(monkeypatch)

    for split in EVAL_SPLITS:
        assert _order(dataloaders[split]) == _order(dataloaders[split])


def test_eval_order_is_the_same_across_loads_whatever_the_global_rng(monkeypatch):
    torch.manual_seed(1)
    first = _dataloaders(monkeypatch)
    torch.manual_seed(2)
    second = _dataloaders(monkeypatch)

    for split in EVAL_SPLITS:
        assert _order(first[split]) == _order(second[split])


def test_eval_order_is_a_permutation_and_not_file_order(monkeypatch):
    dataloaders = _dataloaders(monkeypatch)

    for split in EVAL_SPLITS:
        order = _order(dataloaders[split])
        assert sorted(order) == list(range(N_ITEMS))
        assert order != list(range(N_ITEMS))


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

from torch.utils.data import DataLoader
import torch

from . import shapeworld
from . import cub


def load(config):
    if "shapeworld" in config['data']['dataset']:
        lf = shapeworld.load
    elif "cub" in config['data']['dataset']:
        lf = cub.load
    else:
        raise ValueError(f"Unknown dataset {config['data']['dataset']}")
    return lf(config)


# The seed of the one fixed order every eval pass reads its split in. A
#     constant rather than `--seed`, so that on ShapeWorld runs differing only
#     in seed are measured on the same topsim game pairs.
EVAL_ORDER_SEED = 0


def load_dataloaders(config):
    datas = load(config)

    def to_dl(split, dset):
        # `train` and `train_clean` shuffle afresh every pass. `train_clean`
        #     has to, because `calibrate_batch_norm` reads only its first
        #     `bn_calibration_batches` batches.
        #
        # The eval splits read one permutation, drawn once from their own
        #     generator, so the order is the same every pass, every epoch and
        #     every run, whatever the global RNG and `--seed`. Fixed, because
        #     topsim's `topsim_seed` pairs index games by position, and so land
        #     on the same ShapeWorld games every epoch only if the order is.
        #     Permuted rather than in file order, because `shuffled_message_acc`
        #     pairs each game with its batch neighbour's message, and file
        #     order may put games of one concept side by side. Birds eval games
        #     are redrawn per item whatever the order.
        if split in ("train", "train_clean"):
            order = dict(shuffle=True)
        else:
            generator = torch.Generator().manual_seed(EVAL_ORDER_SEED)
            order = dict(
                sampler=torch.randperm(len(dset), generator=generator).tolist()
            )

        # No `worker_init_fn`: torch seeds each worker's torch, numpy and
        #     `random` from the main process's torch RNG, so `--seed` fixes
        #     the data stream too. A reseed from OS entropy here would not.
        return DataLoader(
            dset,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['n_workers'],
            pin_memory=True,
            **order,
        )

    return {split: to_dl(split, dset) for split, dset in datas.items()}

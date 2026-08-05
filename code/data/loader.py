import numpy as np
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


def worker_init(worker_id):
    np.random.seed()
    torch.seed()


def load_dataloaders(config):
    datas = load(config)

    def to_dl(dset):
        return DataLoader(
            dset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['n_workers'],
            pin_memory=True,
            worker_init_fn=worker_init,
        )

    return {split: to_dl(dset) for split, dset in datas.items()}

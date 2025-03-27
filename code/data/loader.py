import numpy as np
from torch.utils.data import DataLoader
import torch

from . import shapeworld
from . import cub


def load(data_config):
    if "shapeworld" in data_config['dataset']:
        lf = shapeworld.load
    elif "cub" in data_config['dataset']:
        lf = cub.load
    else:
        raise ValueError(f"Unknown dataset {data_config['dataset']}")
    return lf(data_config)


def worker_init(worker_id):
    np.random.seed()
    torch.seed()


def load_dataloaders(data_config):
    datas = load(data_config)

    def to_dl(dset):
        return DataLoader(
            dset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['batch_size'],
            pin_memory=True,
            worker_init_fn=worker_init,
        )

    dataloaders = {split: to_dl(dset) for split, dset in datas.items()}
    
    # Commented out as never used in practice in EmComGen. Reimplement if needed.
    if data_config['test_dataset'] is not None:
        orig_dataset = data_config['dataset']
        orig_percent_novel = data_config['percent_novel']
        orig_n_examples = data_config['n_examples']

        data_config['dataset'] = data_config['test_dataset']
        data_config['percent_novel'] = data_config['test_percent_novel']
        data_config['n_examples'] = data_config['test_n_examples']

        test_datas = load(data_config)

        data_config['dataset'] = orig_dataset
        data_config['percent_novel'] = orig_percent_novel
        data_config['n_examples'] = orig_n_examples

        dataloaders["test"] = to_dl(test_datas["test"])

    return dataloaders

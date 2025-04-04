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

    dataloaders = {split: to_dl(dset) for split, dset in datas.items()}
    
    # Commented out as never used in practice in EmComGen. Reimplement if needed.
    # if config['data']['test_dataset'] is not None:
    #     orig_dataset = config['data']['dataset']
    #     orig_percent_novel = config['data']['percent_novel']
    #     orig_n_examples = config['data']['n_examples']

    #     config['data']['dataset'] = config['data']['test_dataset']
    #     config['data']['percent_novel'] = config['data']['test_percent_novel']
    #     config['data']['n_examples'] = config['data']['test_n_examples']

    #     test_datas = load(config)

    #     config['data']['dataset'] = orig_dataset
    #     config['data']['percent_novel'] = orig_percent_novel
    #     config['data']['n_examples'] = orig_n_examples

    #     dataloaders["test"] = to_dl(test_datas["test"])

    return dataloaders

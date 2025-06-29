"""
Sample from a pretrained speaker, correcting for train/test imbalance (check #
of concepts)
"""

import os
import sys
from pathlib import Path

import torch
import pandas as pd

import parse_config
import util
# import models
import data
from train import run
import json
from tqdm import tqdm
import emergentlanguageagents.models as elam

def sample(pair, dataloaders, config, exp_dir):
    dname = dataloaders["train"].dataset.name
    if dname == "cub":
        n_train_concepts = 100
        n_test_concepts = 50
    elif dname == "shapeworld":
        # Note - ref game has different numbers, but the same split
        n_train_concepts = 312
        n_test_concepts = 63
    else:
        raise NotImplementedError(dname)

    train_pct = n_train_concepts / (n_train_concepts + n_test_concepts)
    n_train_samples = round(train_pct * config['data']['n_sample'])
    n_test_samples = config['data']['n_sample'] - n_train_samples

    def sample_split(split, n):
        stats = util.Statistics()
        all_lang = pd.DataFrame()
        pbar = tqdm(desc=f"sample {split}", total=n)
        while all_lang.shape[0] < n:
            split_stats, lang = run(
                split, 0, pair, None, dataloaders, None, None, config, force_no_train=True
            )
            if dname == "cub":  # Zero out metadata
                lang["md"] = 0
            all_lang = pd.concat((all_lang, lang))
            pbar.update(lang.shape[0])
            stats.update(**split_stats)
        pbar.close()
        all_lang = all_lang.head(n)
        all_lang["split"] = split
        stats = stats.averages()
        return stats, all_lang

    train_stats, train_lang = sample_split("train", n_train_samples)
    test_stats, test_lang = sample_split("test", n_test_samples)

    if config['force_reference_game']:
        lang_fname = "sampled_lang_force_ref.csv"
    elif config['force_concept_game']:
        lang_fname = "sampled_lang_force_concept.csv"
    elif config['force_setref_game']:
        lang_fname = "sampled_lang_force_setref.csv"
    else:
        lang_fname = "sampled_lang.csv"
    lang_fname = os.path.join(exp_dir, lang_fname)
    all_lang = pd.concat((train_lang, test_lang))
    all_lang.to_csv(lang_fname, index=False)

    # Save statistics
    comb_stats = {}
    util.update_with_prefix(comb_stats, train_stats, "train")
    util.update_with_prefix(comb_stats, test_stats, "test")
    if config['force_reference_game']:
        fname = "sampled_stats_force_ref.json"
    elif config['force_concept_game']:
        fname = "sampled_stats_force_concept.json"
    elif config['force_setref_game']:
        fname = "sampled_stats_force_setref.json"
    else:
        fname = "sampled_stats.json"
    with open(os.path.join(exp_dir, fname), "w") as f:
        json.dump(comb_stats, f)

if __name__ == "__main__":
    
    arguments = sys.argv[1:]
    if not arguments:
        class NoArguments(Exception):
            pass
        raise NoArguments(
            "Intended usage: `python sample.py [experiment directory]`"
        )
    else:
        # TODO: make sure we're actually saving each experiment's config as config.toml in the experiment directory
        # The below will automatically restore defaults if missing
        exp_dir = arguments[0]
        config = parse_config.get_config(Path(exp_dir) / 'config.toml')

    if config['force_reference_game']:
        config['reference_game'] = True
        config['data']['percent_novel'] = 0.0
        if (
            "shapeworld" in config['data']['dataset']
            and "shapeworld_ref" not in config['data']['dataset']
        ):
            # Change SW dataset to ref version (no change for CUB)
            if "shapeworld_all" in config['data']['dataset']:
                config['data']['dataset'] = config['data']['dataset'].replace(
                    "shapeworld_all", "shapeworld_ref"
                )
            else:
                config['data']['dataset'] = config['data']['dataset'].replace(
                    "shapeworld", "shapeworld_ref"
                )
    elif config['force_concept_game']:
        config['reference_game'] = False
        config['data']['percent_novel'] = 1.0
        if "shapeworld_ref" in config['data']['dataset']:
            # Change SW dataset to ref version (no change for CUB)
            config['data']['dataset'] = config['data']['dataset'].replace("shapeworld_ref", "shapeworld")
    elif config['force_setref_game']:
        config['reference_game'] = False
        config['data']['percent_novel'] = 0.0
        if "shapeworld_ref" in config['data']['dataset']:
            # Change SW dataset to ref version (no change for CUB)
            config['data']['dataset'] = config['data']['dataset'].replace("shapeworld_ref", "shapeworld")

    dataloaders = data.loader.load_dataloaders(config)

    sender_class = getattr(elam.senders, config['sender']['class'])
    sender = sender_class(**config['sender']['arguments'])

    receiver_class = getattr(elam.receivers, config['receiver']['class'])
    receiver = receiver_class(**config['receiver']['arguments'])

    pair = util.Pair(sender, receiver)

    # dataloaders = data.loader.load_dataloaders(exp_args)
    # model_config = models.builder.build_models(dataloaders, exp_args)
    # pair = model_config["pair"]

    state_dict = torch.load(os.path.join(exp_dir, "best_model.pt"))
    pair.load_state_dict(state_dict)
    
    if config['cuda']:
        pair.cuda()

    sample(pair, dataloaders, config, exp_dir)

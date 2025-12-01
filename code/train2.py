"""
Train an RNN decoder to make binary predictions;
then train an RNN language model to generate sequences
"""
from time import time
from typing import List, Union

from torch.amp import GradScaler, autocast

import sys
from pathlib import Path

import contextlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import models
import models.backbone

import parse_config
import util
import data
import os
import vis
import emergence

import pandas as pd
# import io_util

# Logging
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# def convert_lang_to_numeric(lang, lang_length, pad_val=-1, skip_sos_eos=True):
#     """
#     Convert lang to numeric, with custom padding, for later language analysis
#     """
#     lang_i = lang.argmax(2)
#     for i, length in enumerate(lang_length):

#         if skip_sos_eos:
#             # zero out EOS
#             lang_i[i, length - 1 :] = pad_val
#         else:
#             lang_i[i, length:] = pad_val

#     # shave off SOS, ending EOS if present
#     if skip_sos_eos:
#         lang_i = lang_i[:, 1:-1]

#     return lang_i


def get_true_lang(
        batch,
        dataset,
        # args isn't used in this function, so we commented it out
        # args,
        join=True):
    spk_inp, spk_y, lis_inp, lis_y, true_lang, md, idx = batch
    true_lang_text = dataset.to_text(true_lang, join=join)
    return true_lang_text


def get_positive_examples(inp, y):
    """
    inp -> batch_size x n_examples x feat_size
    y -> batch_size x y

    output
    """
    where_zero = np.where(y.sum(1) == 0)[0]
    y[where_zero] = 1
    occur_rows, occur_cols = np.where(y)
    row_indices, occur_col_indices = np.unique(occur_rows, return_index=True)
    assert (row_indices == np.arange(len(row_indices))).all()
    assert len(occur_col_indices) == len(y)
    col_indices = occur_cols[occur_col_indices]
    sel = inp[row_indices, col_indices]
    return sel


def subsample(items, idx):
    return [items[i] for i in idx]


def compute_lang_metrics(
    all_lang,
    dataset,
    # args isn't used in this function, so we commented it out
    # args,
    attrs=None,
    reprs=None,
    attrs_numeric=None,
    toks=None,
    max_analysis_length=1000,
):
    lang_metrics = {}
    if all_lang.shape[0] > max_analysis_length:
        idx = np.random.choice(
            all_lang.shape[0], size=max_analysis_length, replace=False
        )

        all_lang = all_lang.iloc[idx].reset_index()

        if attrs is not None:
            attrs = subsample(attrs, idx)
        if toks is not None:
            toks = subsample(toks, idx)
        if reprs is not None:
            reprs = subsample(reprs, idx)

    # topographic similarity between ground truth language and tokens
    # only do it if the ground truth language is meaningful
    if dataset.name == "shapeworld":
        langts = emergence.topsim(
            all_lang["true_lang"], toks, meaning_distance_fn="edit"
        )
        lang_metrics["langts"] = langts

    if dataset.name == "shapeworld":

        def compute_hd(tl1, tl2):
            # Remove SOS, EOS
            tl1 = " ".join(tl1[1:-1])
            tl2 = " ".join(tl2[1:-1])
            return dataset.concept_distance(tl1, tl2)

    elif dataset.name == "cub":

        def compute_hd(tl1, tl2):
            tl1 = int(tl1[1])
            tl2 = int(tl2[1])
            return dataset.concept_distance(tl1, tl2)

    if dataset.concept_distances is not None:
        hd = emergence.topsim(
            all_lang["true_lang"], toks, meaning_distance_fn=compute_hd
        )
        lang_metrics["hausdorff"] = hd

    if attrs is not None:
        # topographic similarity between meanings and tokens
        ts = emergence.topsim(
            attrs, toks, meaning_distance_fn=dataset.meaning_distance_fn
        )
        lang_metrics["ts"] = ts

        # topographic similarity between reprs and attributes
        # For random sets later, worth disentangling prototype repr from
        # individual inputs repr
        reprts = emergence.topsim(
            attrs,
            reprs,
            meaning_distance_fn=dataset.meaning_distance_fn,
            message_distance_fn="euclidean",
        )
        lang_metrics["reprts"] = reprts

    return lang_metrics


def compute_metrics_by_md(all_lang, md_vocab=None):
    metrics_by_md = {}
    per_md_acc = all_lang[["md", "acc"]].groupby("md").mean()
    for i, md_row in per_md_acc.iterrows():
        if md_vocab is None:
            md_name = str(md_row.name)
        else:
            md_name = md_vocab["i2w"][md_row.name]
        md_key = f"acc_md_{md_name}"
        metrics_by_md[md_key] = md_row["acc"]
    return metrics_by_md


def log_epoch_summary(epoch, split, metrics):
    logging.info(
        "Epoch {}\t{} {}".format(
            epoch,
            split.upper(),
            " ".join("{}: {:.4f}".format(m, v) for m, v in metrics.items()),
        )
    )


def log_epoch_progress(epoch, batch_i, batch_size, dataloader, stats):
    meter_str = " ".join(f"{k}: {v.avg:.3f}" for k, v in stats.meters.items())
    data_i = batch_i * batch_size
    data_total = len(dataloader.dataset)
    pct = round(100 * batch_i / len(dataloader))
    logging.info(f"Epoch {epoch} [{data_i}/{data_total} ({pct}%)] {meter_str}")


def init_metrics():
    """
    Initialize the metrics for this training run. This is a defaultdict, so
    metrics not specified here can just be appended to/assigned to during
    training.
    Returns
    -------
    metrics : `collections.defaultdict`
        All training metrics
    """
    metrics = {}
    metrics["best_acc"] = 0.0
    metrics["best_val_acc"] = 0.0
    metrics["best_val_same_acc"] = 0.0
    metrics["best_loss"] = float("inf")
    metrics["best_epoch"] = 0
    return metrics

#TODO: Why is this function so slow?
def run(
    split,
    epoch,
    pair,
    optimizer,
    dataloaders,
    scheduler,
    scaler,
    # args,
    config,
    random_state=None,
    force_no_train=False,
):
    """
    Run the model for a single epoch.

    Parameters
    ----------
    split : ``str``
        The dataloader split to use. Also determines model behavior if e.g.
        ``split == 'train'`` then model will be in train mode/optimizer will be
        run.
    epoch : ``int``
        current epoch
    model : ``torch.nn.Module``
        the model you are training/evaling
    optimizer : ``torch.nn.optim.Optimizer``
        the optimizer
    criterion : ``torch.nn.loss``
        the loss function
    dataloaders : ``dict[str, torch.utils.data.DataLoader]``
        Dictionary of dataloaders whose keys are the names of the ``split``s
        and whose values are the corresponding dataloaders
    args : ``argparse.Namespace``
        Arguments for this experiment run
    random_state : ``np.random.RandomState``
        The numpy random state in case anything stochastic happens during the
        run

    Returns
    -------
    metrics : ``dict[str, float]``
        Metrics from this run; keys are statistics and values are their average
        values across the batches
    """
    epoch_start = time()
    bce_criterion = nn.BCEWithLogitsLoss()
    xent_criterion = nn.CrossEntropyLoss()
    training = (split == "train") and not force_no_train
    dataloader = dataloaders[split]
    torch.set_grad_enabled(training)
    pair.train(mode=training)

    stats = util.Statistics()

    all_lang = []
    all_toks = []  # language, unjoined text form, ragged

    if dataloader.dataset.name == "cub":
        all_attrs = []
        all_reprs = []  # representations
    else:
        all_attrs = None
        all_reprs = None

    if training:
        optimizer.zero_grad()
        # this_epoch_eps = max(0.0, config['sender']['arguments']['eps'] - (epoch * config['sender']['arguments']['eps_anneal']))
        # this_epoch_uniform_weight = max(
        #     0.0, config['sender']['arguments']['uniform_weight'] - (epoch * config['sender']['arguments']['uniform_weight_anneal'])
        # )
        # this_epoch_softmax_temp = max(
        #     1.0, config['sender']['arguments']['temperature'] - (epoch * config['sender']['arguments']['temperature_annealing'])
        # )
    # else:
    #     this_epoch_eps = 0.0
    #     this_epoch_uniform_weight = 0.0
    #     this_epoch_softmax_temp = 1.0
    
    setup_finished_in = time() - epoch_start

    # Performance stats
    def avg(x: List[Union[float, int]]): return sum(x) / len(x)
    batch_preparation = []
    sender_inference = []
    receiver_inference = []
    saving_language = []
    getting_true_lang = []
    pushing_metadata = []
    optimising = []
    updating_stats = []

    for batch_i, batch in enumerate(dataloader):
        batch_start = time()
        spk_inp, spk_y, lis_inp, lis_y, true_lang, md, idx = batch
        batch_size = spk_inp.shape[0]

        # Determine what's input
        if dataloader.dataset.name == "shapeworld":
            spk_inp = spk_inp.float() / 255
            lis_inp = lis_inp.float() / 255
        else:
            spk_inp = spk_inp.float()
            lis_inp = lis_inp.float()

        spk_y = spk_y.float()
        lis_y = lis_y.float()

        if config['cuda']:
            spk_inp = spk_inp.cuda()
            spk_y = spk_y.cuda()
            lis_inp = lis_inp.cuda()
            lis_y = lis_y.cuda()
        
        batch_preparation_end = time()
        batch_preparation.append(batch_preparation_end - batch_start)

        # This is the bit where the models process the inputs
        with autocast(
            device_type='cuda',
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ):
            if config['receiver_only']:
                lis_scores = pair.receiver(lis_inp, None)
            elif config['copy_receiver']:
                sender_emb = pair.sender(spk_inp, spk_y)
                lis_scores = pair.receiver(lis_inp, sender_emb)
            else:
                # (lang, lang_length), states = pair.sender(
                lang, states = pair.sender(
                    spk_inp,
                    spk_y,
                    # max_len=config['sender']['arguments']['message_length'],
                    # eps=this_epoch_eps,
                    # softmax_temp=this_epoch_softmax_temp,
                    # uniform_weight=this_epoch_uniform_weight,
                )

                sender_inference_end = time()
                sender_inference.append(sender_inference_end - batch_preparation_end)

                lis_scores = pair.receiver(lis_inp, lang)#, lang_length)

                receiver_inference_end = time()
                receiver_inference.append(receiver_inference_end - sender_inference_end)

            # Evaluate loss and accuracy
            if config['reference_game_xent']:
                # Take only 0th receiver score + after midpoint. Then do cross
                # entropy
                assert lis_scores.shape[1] % 2 == 0
                midp = lis_scores.shape[1] // 2
                lis_scores_xent = torch.cat((lis_scores[:, :1], lis_scores[:, midp:]), 1)
                zeros = torch.zeros(batch_size, dtype=torch.int64, device=lis_scores.device)
                this_loss = xent_criterion(lis_scores_xent, zeros)
                lis_pred = lis_scores_xent.argmax(1)
                per_game_acc = (lis_pred == 0).float().cpu().numpy()
                this_acc = per_game_acc.mean()
            else:
                this_loss = bce_criterion(lis_scores, lis_y)
                lis_pred = (lis_scores > 0).float()
                per_game_acc = (lis_pred == lis_y).float().mean(1).cpu().numpy()
                this_acc = per_game_acc.mean()

            # Save language
            if config['use_lang']:
                lang_i = lang.argmax(2)
                lang_text_unjoined = util.to_emergent_text(lang_i)
                lang_text = [" ".join(toks) for toks in lang_text_unjoined]
            else:
                lang_text_unjoined = [["N/A"] for _ in range(batch_size)]
                lang_text = ["N/A" for _ in range(batch_size)]

            saving_language_end = time()
            saving_language.append(saving_language_end - receiver_inference_end)

            true_lang_text = get_true_lang(
                batch,
                dataloader.dataset,
                # commented out argument `args`, see definition above
                join=False
            )
            true_lang_text_joined = [" ".join(t) for t in true_lang_text]

            getting_true_lang_end = time()
            getting_true_lang.append(getting_true_lang_end - saving_language_end)

            # Game difficulty/other metadata indicator
            all_lang.extend(zip(lang_text, true_lang_text, per_game_acc, md.numpy()))

            # Get attributes
            all_toks.extend(lang_text_unjoined)
            if dataloader.dataset.name == "cub":
                attrs = md.numpy()[:, 1:]
                all_attrs.extend(attrs)
                all_reprs.extend(states.detach().cpu().float().numpy())

            pushing_metadata_end = time()
            pushing_metadata.append(pushing_metadata_end - getting_true_lang_end)

            if config['joint_training']:
                class DoNotUse(Exception): pass
                raise DoNotUse("This should never happen as joint_training should always be false!")
                # Also train sender on classification task
                spk_scores = pair.sender.classify_from_states(states, lis_inp)
                spk_loss = bce_criterion(spk_scores, lis_y)
                spk_pred = (spk_scores > 0).float()
                spk_per_game_acc = (spk_pred == lis_y).float().mean(1).cpu().numpy()
                spk_acc = spk_per_game_acc.mean()
                stats.update(spk_loss=spk_loss, spk_acc=spk_acc)
                comb_loss = this_loss + config['joint_training_lambda'] * spk_loss
            else:
                comb_loss = this_loss

            if training:
                scaler.scale(comb_loss).backward()

                if (batch_i + 1) % config['optimiser']['accumulator_steps'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(pair.parameters(), config['optimiser']['clip_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    backpropped = True
                else:
                    backpropped = False

            optimisation_step_end = time()
            optimising.append(optimisation_step_end - pushing_metadata_end)

            stats.update(
                loss=this_loss, acc=this_acc, batch_size=batch_size, combined_loss=comb_loss
            )

            updating_stats_end = time()
            updating_stats.append(updating_stats_end - optimisation_step_end)

        if training:
            if batch_i % config['optimiser']['log_interval'] == 0:
                log_epoch_progress(epoch, batch_i, batch_size, dataloader, stats)
                print("SPEED STATS\n======================")
                print(f"Batch preparation:\t\t{avg(batch_preparation)} seconds")
                print(f"Sender inference:\t\t{avg(sender_inference)} seconds")
                print(f"Receiver inference:\t\t{avg(receiver_inference)} seconds")
                print(f"Saving language:\t\t{avg(saving_language)} seconds")
                print(f"Getting true language:\t\t{avg(getting_true_lang)} seconds")
                print(f"Pushing metadata:\t\t{avg(pushing_metadata)} seconds")
                print(f"Optimising:\t\t{avg(optimising)} seconds")
                print(f"Updating stats:\t\t{avg(updating_stats)} seconds")
                print("==========================")
                total = (
                    avg(batch_preparation) +
                    avg(sender_inference) +
                    avg(receiver_inference) +
                    avg(saving_language) +
                    avg(getting_true_lang) +
                    avg(pushing_metadata) +
                    avg(optimising) +
                    avg(updating_stats)
                )
                print(f"Total:\t\t{total} seconds")

        if training and not backpropped:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pair.parameters(), config['optimiser']['clip_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # Compute metrics + collect generation language
    metrics = stats.averages()
    all_lang = pd.DataFrame.from_records(
        all_lang,
        columns=["lang", "true_lang", "acc", "md"],
    )

    if config['use_lang']:
        # Compute emergent communication statistics
        if dataloader.dataset.name == "cub":
            attrs_numeric = dataloader.dataset.attr_to_numeric(all_attrs)
        else:
            attrs_numeric = None

        lang_metrics = compute_lang_metrics(
            all_lang,
            dataloader.dataset,
            # commented out argument `args`, see definition above
            attrs=all_attrs,
            reprs=all_reprs,
            attrs_numeric=attrs_numeric,
            toks=all_toks,
        )
        metrics.update(lang_metrics)

    if dataloader.dataset.name == "shapeworld":
        by_md_metrics = compute_metrics_by_md(
            all_lang, md_vocab=dataloader.dataset.metadata_vocab
        )
        metrics.update(by_md_metrics)

    log_epoch_summary(epoch, split, metrics)

    if config['vis']:
        vis.report(
            spk_inp.cpu(),
            spk_y.cpu(),
            lis_inp.cpu(),
            lis_y.cpu(),
            dataloader.dataset,
            epoch,
            split,
            {"sender": lang_text},
            true_lang_text_joined,
            {"sender": lis_pred},
            exp_dir=config['exp_dir'],
        )

    clean_language(all_lang)
    return metrics, all_lang


def clean_language(all_lang_df):
    def clean_lang(lang):
        # Startswith/endswith
        if lang.startswith("<s>"):
            lang = lang[3:]
        if lang.endswith("</s>"):
            lang = lang[:-4]
        return lang

    def clean_true_lang(true_lang):
        return " ".join(true_lang[1:-1])

    all_lang_df["lang"] = all_lang_df["lang"].apply(clean_lang)
    all_lang_df["true_lang"] = all_lang_df["true_lang"].apply(clean_true_lang)

class NoArguments(Exception):
    pass

if __name__ == "__main__":
    # args = io_util.parse_args()
    # args_dict = vars(args)
    arguments = sys.argv[1:]
    if not arguments:
        raise NoArguments(
            "Intended usage: `python train.py [path to experiment TOML file]`"
        )
    else:
        config = parse_config.get_config(arguments[0])
        # from pprint import pprint
        # pprint(config)
        # import sys
        # sys.exit()

    exp_dir = str(Path(config['experiments_directory']) / Path(config['name']))
    config['exp_dir'] = exp_dir

    print(f"LR: {config['optimiser']['lr']}")
    
    os.makedirs(exp_dir, exist_ok=True)
    util.save_args(config, exp_dir)

    dataloaders = data.loader.load_dataloaders(config)
    model_config = models.builder2.build_models(dataloaders, config)
    this_game_type = data.util.get_game_type(config)

    # sender_class = getattr(models.sender, config['sender']['class'])
    # sender = sender_class(**config['sender']['arguments'])

    # receiver_class = getattr(models.receiver, config['receiver']['class'])
    # receiver = receiver_class(**config['receiver']['arguments'])

    # pair = util.Pair(sender, receiver)

    # if config['cuda']:
    #     pair = pair.cuda()
    

    # optimiser = optim.Adam( # FIXME: build a better optimiser here
    #     pair.parameters(),
    #     lr=config['optimiser']['lr']
    # )
    
    scaler = GradScaler()
    
    # TODO: replace this with PASS scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        model_config['optimiser'],
        factor=0.5,
        patience=10,
    )

    checkpoint_path = os.path.join(exp_dir, "checkpoint_last.pt")
    start_epoch = 0
    
    # Initialize metrics normally first
    metrics = init_metrics()
    all_metrics = [] # Initialize this too

    if config.get('resume', False) and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Load states
        model_config['pair'].load_state_dict(checkpoint['model_state_dict'])
        model_config['optimiser'].load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore metadata
        start_epoch = checkpoint['epoch']
        metrics = checkpoint['metrics'] # Restore best_acc, etc.
        if 'all_metrics' in checkpoint:
            all_metrics = checkpoint['all_metrics']
            
        print(f"Resumed at epoch {start_epoch}")

    run_args = (
        model_config['pair'],
        model_config['optimiser'],
        dataloaders,
        scheduler,
        scaler,
        config
    )

    all_metrics = []
    metrics = init_metrics()
    
    print("Starting to train")

    if config['compile']:
        run = torch.compile(run)

    for epoch in range(start_epoch, config['scheduler']['epochs']):
        if (# No reset on epoch 0, but reset after epoch 2, epoch 4, etc
            config['receiver_reset_interval'] > 0
            and (epoch % config['receiver_reset_interval']) == 0
        ):
            logging.info(f"Resetting receiver at epoch {epoch}")
            model_config['pair'].receiver.reset_parameters()

        metrics["epoch"] = epoch

        # Train
        train_metrics, lang = run("train", epoch, *run_args)
        util.update_with_prefix(metrics, train_metrics, "train")

        # Eval across seen/unseen splits, and all game configurations
        for game_type in ["ref", "setref", "concept"]:
            if config['no_cross_eval'] and game_type != this_game_type:
                continue
            for split in ["val", "test"]:
                split_metrics = defaultdict(list)

                for split_type in ["", "_same"]:
                    sname = f"{split}{split_type}_{game_type}"
                    if sname in dataloaders:
                        eval_metrics, eval_lang = run(sname, epoch, *run_args)
                        util.update_with_prefix(metrics, eval_metrics, sname)
                        if this_game_type == game_type:
                            # Default
                            util.update_with_prefix(
                                metrics, eval_metrics, f"{split}{split_type}"
                            )

                        for metric, value in eval_metrics.items():
                            split_metrics[metric].append(value)

                    if sname == f"test_{this_game_type}":
                        # Store + concatenate test language
                        lang = pd.concat((lang, eval_lang), axis=0)

                # Average across seen and novel
                split_metrics = {k: np.mean(v) for k, v in split_metrics.items()}
                util.update_with_prefix(
                    metrics, split_metrics, f"{split}_avg_{game_type}"
                )
                if this_game_type == game_type:
                    # Default
                    util.update_with_prefix(metrics, split_metrics, f"{split}_avg")

        # Use validation accuracy to choose the best model.
        is_best = metrics["val_avg_acc"] > metrics["best_acc"]
        if is_best:
            metrics["best_acc"] = metrics["val_avg_acc"]
            metrics["best_loss"] = metrics["val_avg_loss"]
            metrics["best_epoch"] = epoch
            if config['use_lang']:
                lang.to_csv(os.path.join(exp_dir, "best_lang.csv"), index=False)
            # Save the model
            model_fname = os.path.join(exp_dir, "best_model.pt")
            torch.save(model_config['pair'].state_dict(), model_fname)

        if epoch % config['save_interval'] == 0:
            model_fname = os.path.join(exp_dir, f"{epoch}_model.pt")
            torch.save(model_config['pair'].state_dict(), model_fname)
            if config['use_lang']:
                lang.to_csv(os.path.join(exp_dir, f"{epoch}_lang.csv"), index=False)

        # Checkpoint every epoch to allow resuming
        checkpoint_state = {
            'epoch': epoch + 1, # Save the NEXT epoch index
            'model_state_dict': model_config['pair'].state_dict(),
            'optimizer_state_dict': model_config['optimiser'].state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'metrics': metrics,
            'all_metrics': all_metrics,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state()
        }
        torch.save(checkpoint_state, checkpoint_path)

        # Additionally track best for splits separately
        metrics["best_val_acc"] = max(metrics["best_val_acc"], metrics["val_acc"])
        if "val_same_acc" in metrics:
            metrics["best_val_same_acc"] = max(
                metrics["best_val_same_acc"], metrics["val_same_acc"]
            )

        all_metrics.append(metrics.copy())

        # if args.wandb:
        #     import wandb

        #     wandb.log(metrics)

        pd.DataFrame(all_metrics).to_csv(
            os.path.join(exp_dir, "metrics.csv"), index=False
        )

"""
Train a sender/receiver pair on a reference or concept game.

Entry point: ``python train.py --config <toml> [--seed N] [--no_resume]``.
See docs/training.md.
"""
from torch.amp import GradScaler, autocast

import os
from pathlib import Path

from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

import models
import models.backbone
import models.builder
import models.sender
import models.receiver

import parse_config
import util
import data
import vis
import emergence

import pandas as pd

# Logging
import logging

import gradboard.cycles
from gradboard.scheduler import PASS

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Opt-in profiling state, set by `--profile-steps` and otherwise left alone.
# `run` checks `_PROFILER is not None` once per batch, which costs an identity
# comparison on a path that already does a `.item()` and a log check, so a
# normal run is unaffected. See `profile_train_epoch`.
_PROFILER = None
_PROFILE_UNTIL = 0


def get_true_lang(batch, dataset, join=True):
    spk_inp, spk_y, lis_inp, lis_y, true_lang, md, idx = batch
    true_lang_text = dataset.to_text(true_lang, join=join)
    return true_lang_text


def concept_keys_from_true_lang(true_lang_text, dataset_name):
    """
    Map each item's ground-truth language to the key identifying its concept: a
    CUB concept is its (integer) class id; a shapeworld concept is the logical-
    form string with SOS/EOS stripped. Items sharing a key are instances of the
    same concept and are pooled into one prototype for the topsim suite.
    """
    if dataset_name == "cub":
        return [int(t[1]) for t in true_lang_text]
    return [" ".join(t[1:-1]) for t in true_lang_text]


def representative_message(messages):
    """The most common token sequence among messages for one concept."""
    counts = Counter(tuple(m) for m in messages)
    return list(counts.most_common(1)[0][0])


def compute_language_metrics(
    messages,
    symbol_embeddings,
    concepts,
    concept_keys,
    max_concepts,
    ground_truth_meaning,
):
    """
    The topsim report over concept prototypes (``emergence.topsim_report``).

    Measurement is per *concept prototype*, not per image; a prototype is the
    modal token sequence its instances emitted, the mean of the contextual
    symbol embeddings of every instance that emitted it, and the mean of the
    instances' concept vectors. See docs/measurement.md.

    ``ground_truth_meaning`` says whether the concept keys are logical forms, in
    which case the ``topsim_gt_`` family is reported against them too.
    """
    concepts = np.asarray(concepts, dtype=np.float64)

    groups = defaultdict(list)
    for i, key in enumerate(concept_keys):
        groups[key].append(i)

    keys = list(groups)
    if len(keys) > max_concepts:
        chosen = np.random.choice(len(keys), size=max_concepts, replace=False)
        keys = [keys[i] for i in sorted(chosen)]

    # Spearman needs at least two pairs, i.e. at least three prototypes.
    if len(keys) < 3:
        return {}

    proto_messages = []
    proto_embeddings = []
    proto_concepts = []
    for key in keys:
        idx = groups[key]
        message = representative_message([messages[i] for i in idx])
        # Every source emitted `message`, so these all stack cleanly. Slice in
        # case an EOS trimmed this message short.
        sources = [i for i in idx if list(messages[i]) == message]
        proto_messages.append(message)
        proto_embeddings.append(
            np.stack([
                np.asarray(symbol_embeddings[i])[: len(message)]
                for i in sources
            ]).mean(0)
        )
        proto_concepts.append(concepts[idx].mean(0))

    return emergence.topsim_report(
        proto_messages,
        proto_embeddings,
        np.stack(proto_concepts),
        formulas=keys if ground_truth_meaning else None,
    )


def build_lr_schedule(config, training_examples, batch_size):
    """
    The learning-rate schedule: `[scheduler] warm_up_epochs` ascending from zero
    to the base rate, then `[scheduler] lr_schedule_shape` over the rest.

    **The warm-up always starts at zero and always ends at the base rate**, and
    the shape that follows always *opens* at the base rate, so the two meet.
    `parse_config.LR_SCHEDULE_SHAPES` is what makes the second half of that true:
    the config names an intention -- `flat` or `cosine` -- rather than a
    `gradboard.cycles.FN_LIBRARY` curve, and the two curves it can name are the
    two that open at their peak.

    **`cool_point_multiplier` governs only what comes after the warm-up.** It is
    how far a descending shape may fall, as a fraction of the base rate, and it
    has nothing to say about how a run opens. `flat` does not descend and so does
    not take one at all; `validate_config` rejects it there rather than leaving
    it unread.

    Neither of those was true before. `PASS.update_learning_rates` computes
    `min_lr + (base_lr - min_lr) * multiplier` with a single
    `min_lr = base_lr * cool_point_multiplier` for the whole run, so handing it
    the configured floor started the warm-up at `floor * base_lr` rather than at
    zero -- and at the `1.0` every rung inherited, pinned every step to `base_lr`
    and flattened the schedule out of existence. `d5c47f5` set that floor
    deliberately alongside `warm_up_epochs = 0`, which was coherent; `b298da5`
    re-enabled the warm-up three weeks later without touching it, and the ramp it
    advertised ran on no rung.

    So the floor is carried in the descending cycle's own `low`, against a `high`
    of 1.0, and `PASS` is built with a floor of 0.0 -- leaving its multiplier to
    act directly on each group's base rate. Every shape in `FN_LIBRARY` returns
    0.0 at its trough and 1.0 at its peak, so `low` and `high` are the fractions
    of base the cycle bottoms and tops out at, uniformly across shapes.

    `PASS` records `cool_point_multiplier` in its `state_dict`, so a run resumed
    from a checkpoint written before this change restores the old floor and goes
    flat again. Start such a run fresh rather than resuming it.

    Args:
        config: the parsed config, for `[scheduler]`
        training_examples: `len(dataset)` for the training split
        batch_size: the *effective* batch, `[data] batch_size` times
            `[optimiser] accumulator_steps`

    Returns:
        A `gradboard.cycles.CycleSequence` over one or two stages.
    """
    epochs = config['scheduler']['epochs']
    warmup_epochs = min(config['scheduler']['warm_up_epochs'], epochs)

    shape, takes_floor = parse_config.LR_SCHEDULE_SHAPES[
        config['scheduler']['lr_schedule_shape']
    ]
    # `validate_config` has already rejected a floor that is absent where it is
    #     needed or present where it is not, so this reads it only where it is
    #     read at all. `flat` is `low = high = 1.0`: constant at base.
    floor = config['scheduler']['cool_point_multiplier'] if takes_floor else 1.0

    lr_stages = []

    if warmup_epochs:
        lr_stages.append(
            gradboard.cycles.Cycle(
                gradboard.cycles.ascent,
                training_examples,
                warmup_epochs,
                batch_size,
                low=0.0,
                high=1.0
            )
        )

    if epochs > warmup_epochs:
        lr_stages.append(
            gradboard.cycles.Cycle(
                shape,
                training_examples,
                epochs - warmup_epochs,
                batch_size,
                low=floor,
                high=1.0
            )
        )

    return gradboard.cycles.CycleSequence(lr_stages)


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


def per_game_accuracy(lis_scores, lis_y, reference_game_xent):
    """
    The listener's per-game accuracy, one number per game in the batch.

    Lifted out of the loop so that `shuffled_message_acc` is scored by exactly
        the code the live accuracy is scored by. A control computed a second way
        is not a control: any divergence between the two would read as
        communication.

    Args:
        lis_scores: (batch, n_objects), the discriminator's output
        lis_y: (batch, n_objects), 1.0 for candidates matching the concept
        reference_game_xent: `[config] reference_game_xent`, which decides
            whether this is a pick-one game or a per-candidate judgement

    Returns:
        A numpy array of shape (batch,), each entry in [0, 1]
    """
    if reference_game_xent:
        # Take only 0th receiver score + after midpoint, and the target is
        # always index 0 by construction.
        assert lis_scores.shape[1] % 2 == 0
        midp = lis_scores.shape[1] // 2
        selected = torch.cat((lis_scores[:, :1], lis_scores[:, midp:]), 1)
        return (selected.argmax(1) == 0).float().cpu().numpy()

    # A fixed threshold at zero, against a score the readout no longer centres.
    #     `score_bias` and the discriminators' own asymmetries are what move the
    #     decision off it. `train_acc` is not comparable across the commit that
    #     removed the centring, in either direction -- see
    #     `receiver.ScoreVolume`.
    return ((lis_scores > 0).float() == lis_y).float().mean(1).cpu().numpy()


def clip_gradients(pair, max_norm):
    """
    Clip each group's gradients to `max_norm` independently. See
        docs/training.md.

    The groups come from `models.builder`, which is also where each one's
        learning rate is read, so a module cannot be rateable without being
        clipped or the other way round. That list used to live here and had
        drifted: it omitted `sender.contrast`, so on every rung with the stage
        on the `other` group *was* the contrast stage under a misleading name.

    Args:
        pair: the sender/receiver `Pair`, with gradients already unscaled
        max_norm: the per-group ceiling, `[optimiser] clip_grad_norm`

    Returns:
        `{group name: that group's gradient norm *before* clipping}`, with a key
            for every entry of `builder.GROUP_NAMES` on every rung and `nan`
            where the group does not exist on this architecture or holds nothing
            with a gradient. The shape is fixed so that the metrics header
            survives a resume against a config that toggles a stage, exactly as
            the contrast columns are NaN-filled rather than absent.
    """
    norms = {}

    for name, params in models.builder.group_parameters(pair):
        with_grad = [p for p in params if p.grad is not None]

        norms[name] = (
            torch.nn.utils.clip_grad_norm_(with_grad, max_norm).item()
            if with_grad
            else float("nan")
        )

    return norms


def run(
    split,
    epoch,
    pair,
    optimizer,
    dataloaders,
    scheduler,
    scaler,
    config,
    random_state=None,
    compute_topsim=False,
):
    """
    Run the model for a single epoch.

    Parameters
    ----------
    split : ``str``
        The dataloader split to use. Also determines model behavior: if
        ``split == 'train'`` the model is in train mode and the optimizer runs.
    epoch : ``int``
        current epoch
    pair : ``models.base.Pair``
        the sender/receiver pair being trained or evaluated
    optimizer : ``torch.nn.optim.Optimizer``
        the optimizer
    dataloaders : ``dict[str, torch.utils.data.DataLoader]``
        Dictionary of dataloaders keyed by split name
    random_state : ``np.random.RandomState``
        The numpy random state in case anything stochastic happens during the
        run
    compute_topsim : ``bool``
        If true, drive the sender through ``speak`` so that message, symbol
        embeddings and concepts all come from one forward pass, collect them,
        and compute the topsim report at the end. Set on the eval passes only.

    Returns
    -------
    metrics : ``dict[str, float]``
        Metrics from this run; keys are statistics and values are their average
        values across the batches
    """
    bce_criterion = nn.BCEWithLogitsLoss()
    xent_criterion = nn.CrossEntropyLoss()
    training = split == "train"
    dataloader = dataloaders[split]
    torch.set_grad_enabled(training)
    pair.train(mode=training)

    stats = util.Statistics()

    all_lang = []

    # Collected only when topsim is computed for this split (the eval passes).
    # Skipped otherwise to keep the train pass light.
    all_messages = []    # emergent messages as ragged content-token-id lists
    all_symbol_embs = []  # (content length, d) contextual embedding per message
    all_concepts = []    # sender concept vectors (positive/negative prototypes)
    all_true_lang = []   # ground-truth language tokens, for concept keys

    # Whether the sender's language model runs at all this pass, and so whether
    # there is a message (and an exploration gain) to report on.
    speaking = (
        config['use_lang']
        and not config['receiver_only']
        and not config['copy_receiver']
    )

    collect = compute_topsim and speaking

    def optimiser_step(loss):
        """
        Called from inside the loop and once more after it, for a trailing
        partial accumulation. Both paths have to agree, which is why they are
        one function. Unscale then clip, per
        https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        """
        scaler.unscale_(optimizer)

        # Recorded per optimiser step rather than per example -- a gradient
        # norm is a property of the step, and the trailing partial
        # accumulation is one step like any other. `clip_gradients` reports
        # every group on every rung, NaN where the group does not exist, so
        # these columns keep their shape across a resume.
        stats.update(**{
            f"clip_{name}": norm for name, norm in
            clip_gradients(pair, config['optimiser']['clip_grad_norm']).items()
        })

        scaler.step(optimizer)
        scaler.update()
        scheduler.step(loss.item())
        optimizer.zero_grad()

    # Bound before the loop so the flush below reads a defined name on an
    # empty split.
    backpropped = True

    if training:
        optimizer.zero_grad()

    for batch_i, batch in enumerate(dataloader):
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
            elif collect:
                # One forward pass, so message, embeddings and concepts
                # correspond. See docs/architecture.md.
                lang, symbol_embs, concepts = pair.sender.speak(spk_inp, spk_y)

                lis_scores = pair.receiver(lis_inp, lang)
            else:
                lang, concepts = pair.sender(
                    spk_inp,
                    spk_y,
                )

                lis_scores = pair.receiver(lis_inp, lang)

            # Evaluate loss and accuracy
            if config['reference_game_xent']:
                # Take only 0th receiver score + after midpoint. Then do cross
                # entropy
                assert lis_scores.shape[1] % 2 == 0
                midp = lis_scores.shape[1] // 2
                lis_scores_xent = torch.cat((lis_scores[:, :1], lis_scores[:, midp:]), 1)
                zeros = torch.zeros(batch_size, dtype=torch.int64, device=lis_scores.device)
                this_loss = xent_criterion(lis_scores_xent, zeros)
            else:
                this_loss = bce_criterion(lis_scores, lis_y)

            per_game_acc = per_game_accuracy(
                lis_scores, lis_y, config['reference_game_xent']
            )
            this_acc = per_game_acc.mean()

            # For `vis.report` at the end of the pass, and nothing else -- the
            # accuracy above is `per_game_accuracy`'s, so this cannot drift from
            # the metric the way it could when the two were computed together.
            lis_pred = (
                lis_scores_xent.argmax(1)
                if config['reference_game_xent']
                else (lis_scores > 0).float()
            )

            # The listener's image-only baseline: the same candidates scored
            # against another game's message. Anything `acc` holds above this is
            # what the channel is actually buying -- and without it a run cannot
            # be read at all, because a listener that has stopped conditioning
            # on the message still scores well above chance from the images
            # alone. On the 2026-08-29 ShapeWorld run it reached 0.588 on shape
            # concepts with the speaker emitting one message for every game.
            #
            # `roll` and not `randperm`: a permutation leaves about one game in
            # `batch_size` holding its own message, which biases the baseline
            # upwards by ~3% at 32. Rolling by one pairs nothing with itself and
            # is reproducible without touching the run's RNG.
            #
            # Eval splits only. `torch.set_grad_enabled(training)` is already
            # False here so this is a forward pass and no graph, but it re-embeds
            # the candidates, and the train pass is where the compute is.
            # `batch_size > 1` because rolling a batch of one pairs it with
            # itself, which would report the live accuracy as the baseline. A
            # trailing partial batch can be that small.
            if not training and speaking and batch_size > 1:
                shuffled_acc = per_game_accuracy(
                    pair.receiver(lis_inp, torch.roll(lang, 1, 0)),
                    lis_y,
                    config['reference_game_xent'],
                ).mean()
                stats.update(
                    shuffled_message_acc=shuffled_acc, batch_size=batch_size
                )

            # Save language
            if config['use_lang']:
                lang_i = lang.argmax(2).detach().cpu()
                lang_text_unjoined = util.to_emergent_text(lang_i)
                lang_text = [" ".join(toks) for toks in lang_text_unjoined]
            else:
                lang_text_unjoined = [["N/A"] for _ in range(batch_size)]
                lang_text = ["N/A" for _ in range(batch_size)]

            true_lang_text = get_true_lang(
                batch, dataloader.dataset, join=False
            )
            true_lang_text_joined = [" ".join(t) for t in true_lang_text]

            # Game difficulty/other metadata indicator
            all_lang.extend(zip(lang_text, true_lang_text, per_game_acc, md.numpy()))

            if collect:
                all_messages.extend(models.sender.trim_messages(lang_i.tolist()))
                all_symbol_embs.extend(symbol_embs.detach().cpu().float().numpy())
                all_concepts.extend(concepts.detach().cpu().float().numpy())
                all_true_lang.extend(true_lang_text)

            if training:
                scaler.scale(this_loss / config['optimiser']['accumulator_steps']).backward()

                backpropped = (
                    (batch_i + 1) % config['optimiser']['accumulator_steps'] == 0
                )
                if backpropped:
                    optimiser_step(this_loss)

            stats.update(
                loss=this_loss.item(), acc=this_acc, batch_size=batch_size,
                combined_loss=this_loss.item()
            )

            # The speaker's channel and prototyper, measured once per batch on
            # the train pass only. See docs/measurement.md for how to read each
            # column and which are NaN on which architecture.
            if training and speaking:
                language_model = pair.sender.language_model
                prototyper = pair.sender.prototyper

                # The contrast stage is optional, so its three columns are NaN
                # rather than absent when it is off -- the header has to be the
                # same shape either way or a run cannot be resumed against a
                # config that toggles it, and a NaN says "not applicable" where
                # a zero would read as a stage that never opened. Guarded on
                # `None` rather than by `hasattr`, matching `Sender`.
                # `contrast_gate` is a parameter and so does not depend on the
                # pass; the other two are per-batch.
                contrast = pair.sender.contrast
                unmeasured = float("nan")

                stats.update(
                    realised_survival=language_model.realised_survival,
                    unmixed_survival=language_model.unmixed_survival,
                    logit_margin=language_model.logit_margin,
                    logit_prior_share=language_model.logit_prior_share,
                    logit_spread=language_model.logit_spread,
                    logit_scale=language_model.logit_scale,
                    sampling_tau=language_model.tau,
                    pool_effective_examples=prototyper.pool_effective_examples,
                    pool_score_norm=prototyper.pool_score_norm,
                    pool_score_sd=prototyper.pool_score_sd,
                    referent_spread=pair.sender.referent_spread,
                    referent_spread_backbone=pair.sender.referent_spread_backbone,
                    polarity_separation=language_model.polarity_separation,
                    contrast_gate=(
                        contrast.contrast_gate.item()
                        if contrast is not None else unmeasured
                    ),
                    contrast_share=(
                        contrast.contrast_share
                        if contrast is not None else unmeasured
                    ),
                    contrast_within_share=(
                        contrast.contrast_within_share
                        if contrast is not None else unmeasured
                    ),
                    batch_size=batch_size,
                )

            # The listener's half. Dispatched on the discriminator's class
            # rather than by `hasattr`, so a rename raises instead of quietly
            # producing a NaN column. See docs/measurement.md.
            #
            # `score_scale` is the volume column on both arms: one scalar per
            # discriminator, in front of a score whose *inputs* are normalised.
            # Read it the way it always was -- a dip and a return is a healthy
            # listener declining to commit while the message is noise, a
            # monotone slide is a collapse. On the bilinear arm it multiplies a
            # score calibrated to open at `1/sqrt(3)`, so the column is
            # comparable across rungs and backbones without further arithmetic.
            #
            # The weight norms stay and are volume again on both arms. Nothing
            # downstream divides a rescaling of `bilinear.weight` back out, so
            # its norm is not the drift it was while the readout standardised;
            # on the attention arm the branches also mix at their own
            # magnitudes, so there the norms set the mix as well.
            #
            # `score_bias` is the offset half of the same readout, and the
            # column to read against `train_acc`. `train.py` decides on
            # `lis_scores > 0`, so this is the parameter that places the scores
            # against that origin -- the only one on the bilinear arm, which had
            # no bias anywhere before this column existed.
            #
            # Expect it near zero. Games are balanced 10 positive / 10 negative,
            # so the loss-optimal global offset is about zero and a value that
            # stays there means the scores were already sitting where the
            # threshold assumes. A sustained drift is the finding: it says they
            # are not, and that the listener is spending a parameter on saying
            # so. It cannot correct a *per-game* offset -- the bilinear score's
            # per-game mean is `mean_j(LN(r_j)) . proj`, which varies by game --
            # so a bias that moves while accuracy does not means the offset was
            # per-game and no scalar reaches it.
            #
            # `mix_alpha` is the mixing *weight* and `mix_share` is the share
            # of the score each path actually contributes. They would agree only
            # if `forward` standardised the branches, which it deliberately does
            # not, so both are reported -- a gap is a loud or quiet branch.
            if training:
                discriminator = pair.receiver.discriminator

                if isinstance(
                    discriminator, models.receiver.BilinearDiscriminator
                ):
                    stats.update(
                        score_scale=discriminator.score_scale.item(),
                        score_bias=discriminator.score_bias.item(),
                        bilinear_weight_norm=(
                            discriminator.bilinear.weight.norm().item()
                        ),
                        batch_size=batch_size,
                    )
                elif isinstance(
                    discriminator, models.receiver.AttentionDiscriminator
                ):
                    stats.update(
                        score_scale=discriminator.score_scale.item(),
                        score_bias=discriminator.score_bias.item(),
                        mix_alpha=discriminator.mix_alpha,
                        mix_share=discriminator.mix_share,
                        bilinear_weight_norm=(
                            discriminator.bilinear.bilinear.weight.norm().item()
                        ),
                        decision_weight_norm=(
                            discriminator.decision.weight.norm().item()
                        ),
                        path_agreement=discriminator.path_agreement,
                        decision_spread=discriminator.decision_spread,
                        decision_kurtosis=discriminator.decision_kurtosis,
                        batch_size=batch_size,
                    )
                else:
                    raise TypeError(
                        "No metrics are defined for discriminator "
                        f"{type(discriminator).__name__}. Add a branch here "
                        "rather than letting a new listener run unmeasured."
                    )

        if training:
            if batch_i % config['optimiser']['log_interval'] == 0:
                log_epoch_progress(epoch, batch_i, batch_size, dataloader, stats)

        # Profiling only, and inert unless `--profile-steps` was passed. The
        # profiler's schedule advances a batch at a time, so it has to be
        # stepped from inside the loop rather than wrapped around it, and the
        # epoch is cut short once the schedule has finished -- a full epoch of
        # trace is both unnecessary and unopenable.
        if _PROFILER is not None:
            _PROFILER.step()
            if batch_i + 1 >= _PROFILE_UNTIL:
                break

    if training and not backpropped:
        optimiser_step(this_loss)

    # Compute metrics + collect generation language
    metrics = stats.averages()
    all_lang = pd.DataFrame.from_records(
        all_lang,
        columns=["lang", "true_lang", "acc", "md"],
    )

    # How much the language compresses. See docs/measurement.md.
    #
    # The count as well as the fraction, because the fraction cannot be read
    # without knowing the split's size and the splits differ: 0.001 on a 1,000
    # game eval split is one message and total collapse, while 0.0551 on 20,000
    # training games is 1,102 messages -- which is *also* total collapse, being
    # what a single message looks like after the Gumbel noise has corrupted it.
    # One of those two numbers says so on its face.
    if speaking and len(all_lang) > 0:
        metrics["unique_message_fraction"] = (
            all_lang["lang"].nunique() / len(all_lang)
        )
        metrics["unique_message_count"] = float(all_lang["lang"].nunique())

    # The column has to exist on every eval pass whether or not any batch was
    # large enough to roll, or a split that produced none would shift every
    # column after it in the appended row. It cannot be NaN-filled per batch the
    # way an absent module's is, because `Statistics` takes a running mean and
    # one NaN would poison the average rather than mark a gap.
    if not training and speaking:
        metrics.setdefault("shuffled_message_acc", float("nan"))

    if collect and all_messages:
        concept_keys = concept_keys_from_true_lang(
            all_true_lang, dataloader.dataset.name
        )
        lang_metrics = compute_language_metrics(
            all_messages,
            all_symbol_embs,
            all_concepts,
            concept_keys,
            max_concepts=config['analysis']['max_concepts'],
            ground_truth_meaning=dataloader.dataset.name != "cub",
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


def profile_train_epoch(run_args, active_steps, trace_path):
    """Profile the opening batches of a training epoch and write a Chrome trace.

    Answers one question -- where the step time on an A100 actually goes -- for
    a rung as configured, rather than for a backbone in isolation the way
    `scripts/vit_throughput.py` and `scripts/vit_geometry_sweep.py` do. Rungs 11
    and 13 put a `ViT2` on both agents, so for those the answer is close to the
    whole picture; a rung with a ResNet on one side splits between the two.

    The schedule discards three batches, warms up on three more, then records
    `active_steps`. Discarding matters under `compile = true`: the first batches
    pay for tracing and autotuning, and a trace of those measures the compiler.

    Two summaries are printed. The op table is the raw ranking by device time.
    The bucket breakdown is the one that answers "are we making good use of the
    A100": it splits device time into matmul/convolution, attention, and the
    fused pointwise and reduction kernels that Inductor emits. Arithmetic is
    what the GPU is fast at, and a large pointwise share means the step is
    moving memory rather than doing work -- which is the expected shape here,
    given broccoli's attention carries a QK-norm and an output norm on top of
    the block's own pre/post norms, and applies axial RoPE per layer.

    Nothing is written except the trace, and no epoch completes, so this cannot
    touch a checkpoint or a `metrics.csv`.
    """
    global _PROFILER, _PROFILE_UNTIL

    wait, warmup = 3, 3
    _PROFILE_UNTIL = wait + warmup + active_steps

    profiler_schedule = torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active_steps, repeat=1
    )
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # Named explicitly because a preemptible partition can hand back a card
    # that is not the one the rungs are timed against, and the numbers below
    # only mean anything against the GPU the experiment actually runs on.
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    logging.info(
        f"Profiling {active_steps} steps after {wait + warmup} discarded, "
        f"on {device_name}; trace -> {trace_path}"
    )

    with torch.profiler.profile(
        activities=activities,
        schedule=profiler_schedule,
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as profiler:
        _PROFILER = profiler
        try:
            run("train", 0, *run_args)
        finally:
            _PROFILER = None

    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))

    averages = profiler.key_averages()

    # torch renamed the CUDA-time attributes to device-agnostic ones; accept
    # either rather than pinning a torch version.
    def device_time(entry):
        for attribute in ("self_device_time_total", "self_cuda_time_total"):
            if hasattr(entry, attribute):
                return getattr(entry, attribute)
        return 0.0

    sort_key = (
        "self_device_time_total"
        if hasattr(averages[0], "self_device_time_total")
        else "self_cuda_time_total"
    )
    print(averages.table(sort_by=sort_key, row_limit=30))

    buckets = {
        "matmul / conv": (
            "gemm", "cutlass", "cublas", "conv", "wgrad", "dgrad", "implicit",
            "sm80", "sm90", "ampere", "addmm", "bmm", "mm_",
        ),
        "attention": ("flash", "attention", "fmha", "mha"),
        "pointwise / reduction": (
            "triton_poi", "triton_red", "triton_per", "elementwise",
            "reduce_kernel", "vectorized",
        ),
        "layout / copy": ("copy", "cat_", "permute", "contiguous", "transpose"),
    }

    totals = dict.fromkeys(buckets, 0.0)
    totals["other"] = 0.0
    grand_total = 0.0
    for entry in averages:
        # Device kernels only. CPU-side operator entries would be counted twice.
        elapsed = device_time(entry)
        if elapsed <= 0:
            continue
        grand_total += elapsed
        name = entry.key.lower()
        for bucket, needles in buckets.items():
            if any(needle in name for needle in needles):
                totals[bucket] += elapsed
                break
        else:
            totals["other"] += elapsed

    print("\nDevice time by kind of work")
    print("-" * 40)
    for bucket, elapsed in sorted(totals.items(), key=lambda kv: -kv[1]):
        share = 100 * elapsed / grand_total if grand_total else 0.0
        print(f"{bucket:>24}  {share:5.1f}%  {elapsed / 1e3:9.1f} ms")

    # `compile = true` is expected to give several graphs rather than one -- the
    # autoregressive decode loop, the Gumbel sampling and the per-batch `.cpu()`
    # calls all break it. How many, and why, bounds what compile can fuse.
    try:
        import torch._dynamo.utils as dynamo_utils

        breaks = dynamo_utils.counters.get("graph_break", {})
        if breaks:
            print("\nDynamo graph breaks")
            print("-" * 40)
            for reason, count in sorted(breaks.items(), key=lambda kv: -kv[1])[:10]:
                print(f"{count:>6}  {reason}")
    except Exception:  # diagnostics only; never fail a profiling run on this
        pass

    print(f"\nTrace written to {trace_path} -- open it at chrome://tracing "
          f"or https://ui.perfetto.dev")


if __name__ == "__main__":
    import argparse
    import random

    import paths

    parser = argparse.ArgumentParser(
        description="Train an emcomgen sender/receiver pair from a TOML config."
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to the experiment TOML config "
             "(experiments/<exp>/configs/<file>.toml)."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed; SLURM array indices map to seeds for repeats."
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Ignore any existing checkpoint and train from scratch."
    )
    parser.add_argument(
        "--profile-steps", type=int, default=0,
        help="Diagnostic. Profile this many training batches and exit without "
             "training: writes a Chrome trace and prints where device time "
             "goes. Six further batches are run and discarded first, so that "
             "the trace is of the steady state rather than of `torch.compile` "
             "warming up. Nothing else is written."
    )
    parser.add_argument(
        "--profile-out", type=str, default=None,
        help="Where the trace from --profile-steps goes. Defaults to "
             "<exp_dir>/profile_trace.json."
    )
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = parse_config.get_config(args.config)
    if args.no_resume:
        config['resume'] = False

    # Must happen *after* get_config(), which branches on the original dataset
    # string to pick the birds defaults. See docs/training.md.
    data_fast_storage = paths.data_fast_storage()
    output_root = paths.output_root()
    emcomgen_data = data_fast_storage / "emcomgen" / "data"
    for key in ('dataset', 'ref_dataset'):
        if key in config['data']:
            basename = Path(config['data'][key]).name
            config['data'][key] = str(emcomgen_data / basename)

    # <output_root>/<experiment>/<config_stem>_seed<seed>/
    experiment = Path(args.config).resolve().parents[1].name
    config_stem = Path(args.config).stem
    exp_dir = str(output_root / experiment / f"{config_stem}_seed{seed}")
    config['exp_dir'] = exp_dir

    print(f"LR: {config['optimiser']['lr']}")

    os.makedirs(exp_dir, exist_ok=True)

    dataloaders = data.loader.load_dataloaders(config)
    model_config = models.builder.build_models(dataloaders, config)

    # After `build_models`, not before: it resolves the per-module muP learning
    #     rates and writes them back into `config` as `resolved_module_lrs`,
    #     which says which groups were built as well as what each ran at, so
    #     running this first would record a config that does not say what was
    #     built. Nothing is lost on the failure path
    #     -- `load_dataloaders` above could already fail after args.json was
    #     written. See docs/training.md.
    util.save_args(config, exp_dir)
    scaler = GradScaler()

    training_examples = len(dataloaders['train'].dataset)
    batch_size = config['data']['batch_size'] * config['optimiser']['accumulator_steps']

    scheduler = PASS(
        build_lr_schedule(config, training_examples, batch_size),
        model_config['pair'],
        model_config['optimiser'],
        scaler=scaler,
        range_test=config['scheduler']['range_test'],
        # Deliberately not `[scheduler] cool_point_multiplier`. `PASS` would
        #     apply that floor to every cycle alike, warm-up included, which is
        #     not what the floor means. `build_lr_schedule` carries it in the
        #     post-warm-up cycle's own `low` instead, so the multiplier reaching
        #     `PASS` is already the fraction of the base rate to run at. See
        #     that function.
        cool_point_multiplier=0.0
    )

    checkpoint_path = os.path.join(exp_dir, "checkpoint_last.pt")
    metrics_path = os.path.join(exp_dir, "metrics.csv")
    start_epoch = 0

    metrics = {}

    if config.get('resume', False) and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # gradboard's PASS restores model + optimiser + scaler + step_count.
        scheduler.load_state_dict(checkpoint["scheduler_state"])

        start_epoch = checkpoint["epoch"]

        print(f"Resumed at epoch {start_epoch}")

    # A fresh start must not append onto a stale metrics.csv.
    if start_epoch == 0 and os.path.exists(metrics_path):
        os.remove(metrics_path)
    elif os.path.exists(metrics_path):
        # A header written by a run with different splits cannot hold this run's
        # rows, and `to_csv(mode="a")` checks nothing. See docs/training.md.
        with open(metrics_path) as f:
            header = f.readline().rstrip("\n").split(",")

        missing = [
            split for split in dataloaders
            if not any(column.startswith(f"{split}_") for column in header)
        ]
        if missing:
            raise RuntimeError(
                f"{metrics_path} was written by a run with different splits: it "
                f"has no columns for {missing}. That header cannot hold this "
                f"run's rows. Start again with --no_resume, or move the existing "
                f"run directory aside."
            )

    # Compile the two feature models, which is where the time goes, rather than
    #     the `Pair`. `Pair` has no `forward` -- it is a container that exists so
    #     the two agents share an optimiser and a checkpoint -- so
    #     `torch.compile(pair)` wrapped a method nothing calls. The training loop
    #     calls `pair.sender(...)` and `pair.receiver(...)`, and
    #     `OptimizedModule.__getattr__` forwards those to `_orig_mod`, handing
    #     back the *uncompiled* submodules. Compile was inert from the day it
    #     went in: a profile of rung 13 shows no `triton_*` kernel anywhere, no
    #     Dynamo output, and `aten::mul` at 30% of device time.
    #
    # It was worth having. On an A100 at 640 images of 64px, fwd+bwd, the ViT
    #     backbone runs 968ms eager against 303ms compiled -- 18.5 TFLOP/s
    #     against 59.2 -- and `ResNet18SmallInput` 170ms against 81ms, so the
    #     baseline rungs gain too and the comparison between them does not
    #     quietly shift.
    #
    # `Module.compile()` rather than rebinding to `torch.compile(module)`,
    #     because it compiles in place and leaves `state_dict` keys alone. The
    #     old call did not: it prefixed every key with `_orig_mod.`, which is
    #     why checkpoints from a compiled rung and an uncompiled one do not
    #     key-match. Checkpoints written from here on carry plain keys, so they
    #     match neither -- this is a re-run, not a resume.
    #
    # The feature models only. `SenderTransformerLM` and `SenderGRULM` decode
    #     autoregressively and would break into graphs at every step, and the
    #     comparers are small; the backbones are ~90% of a ViT rung and have no
    #     data-dependent control flow at all. Compiling the module rather than
    #     its `forward` also means `Sender.speak` -- the eval-pass entry point
    #     -- gets the compiled backbone too, since it goes through the same
    #     submodule.
    if config['compile']:
        print("Compiling the sender and receiver feature models...")
        model_config['pair'].sender.feat_model.compile()
        model_config['pair'].receiver.feature_model.compile()

    run_args = (
        model_config['pair'],
        model_config['optimiser'],
        dataloaders,
        scheduler,
        scaler,
        config
    )

    print("Starting to train")

    # Diagnostic exit. Deliberately after the build, the compile and the
    # checkpoint restore, so what is profiled is the step a resumed rung runs.
    if args.profile_steps:
        profile_train_epoch(
            run_args,
            args.profile_steps,
            Path(args.profile_out or (Path(exp_dir) / "profile_trace.json")),
        )
        raise SystemExit(0)

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

        # Eval on the novel (`test`) and held-out seen (`test_same`) concepts;
        # `test_same` is the paper's Acc (Seen). See docs/data.md.
        split_metrics = defaultdict(list)
        for split in ["test", "test_same"]:
            if split not in dataloaders:
                # `test_same` is optional
                continue

            eval_metrics, eval_lang = run(
                split, epoch, *run_args, compute_topsim=True
            )
            util.update_with_prefix(metrics, eval_metrics, split)

            for metric, value in eval_metrics.items():
                split_metrics[metric].append(value)

            if split == "test":
                # Store + concatenate test language
                lang = pd.concat((lang, eval_lang), axis=0)

        # Average across seen and novel
        util.update_with_prefix(
            metrics,
            {k: np.mean(v) for k, v in split_metrics.items()},
            "test_avg",
        )

        metrics["timestamp"] = datetime.now().isoformat()

        # Appending is what lets earlier rows survive a resume, so the columns
        # are checked against the header on disk first. See docs/training.md.
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                existing_columns = f.readline().rstrip("\n").split(",")

            if existing_columns != list(metrics):
                raise RuntimeError(
                    f"{metrics_path} has a header this run's rows do not match, "
                    f"so appending would misalign the columns. On disk but not "
                    f"measured now: {sorted(set(existing_columns) - set(metrics))}. "
                    f"Measured now but not on disk: "
                    f"{sorted(set(metrics) - set(existing_columns))}. Start again "
                    f"with --no_resume, or move the existing run directory aside."
                )

        pd.DataFrame([metrics]).to_csv(
            metrics_path,
            mode="a",
            header=not os.path.exists(metrics_path),
            index=False,
        )

        # Checkpoint after the row is on disk so the model/optimiser state stays
        # in sync with metrics.csv on resume. A crash between the two writes can
        # re-emit one row, which is benign.
        torch.save(
            {
                "epoch": epoch + 1,  # resume at the NEXT epoch
                "scheduler_state": scheduler.state_dict(),
            },
            checkpoint_path,
        )

    # Fixed endpoint: no best-epoch selection. Skipped when resuming a run that
    # is already done.
    if start_epoch < config['scheduler']['epochs']:
        torch.save(
            model_config['pair'].state_dict(),
            os.path.join(exp_dir, "final_model.pt"),
        )
        if config['use_lang']:
            lang.to_csv(os.path.join(exp_dir, "final_lang.csv"), index=False)

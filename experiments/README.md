# experiments

Each experiment is a folder here holding one or more TOML configs:

```
experiments/<experiment>/configs/<config>.toml
experiments/<experiment>/logs/            # SLURM stdout/stderr, created at launch
```

`scripts/run_experiment.sh <experiment>` submits one SLURM array task per
(config, seed) pair. Results are written *outside* this tree, to
`<output_root>/<experiment>/<config>_seed<N>/` (see `results/README.md` and
`config.json`), and the per-epoch trajectory lands in `metrics.csv` there.

**`configs/` is the queue, not the ladder.** The array is built from
`configs/*.toml`, so the way to run a subset is to move the rest one level up
into `experiments/<experiment>/` and leave only what should be submitted --
which is what `973a68b` did to run rungs 9 and 10 on their own. A rung parked
up there is still a rung; it is just not queued.

Anything that reads a rung must therefore look in both places, and since the
ablation is two folders (below), that is four directories for a rung of the
ladder. `tests/_bootstrap.rung` and `all_rungs` do, and so do the two probes in
`diagnostics/`. The failure mode when something does not is quiet: a
`FileNotFoundError` inside `parse_config.get_config` at collection time reads
as a wall of failing tests rather than as a missing path, and it hid 159 tests
for a day in August 2026. `test_there_are_sixteen_rungs` is the backstop.

The rest of this file is a reference for the columns of that `metrics.csv`.

## The learning-rate sweeps are a serial chain

**`lr_sweep_1_cnn` through `lr_sweep_8_receiver_cross_attention_lm`**, one
folder per rung of the ladder, each measuring the learning rate of the one
component that rung introduces. They must be run **in order**, and DEFAULT.toml
must be updated with each result before the next is launched.

| sweep | rungs | what it tunes | key |
|---|---|---|---|
| `lr_sweep_1_cnn` | 1 / 2 | `ResNet56`, `ResNet18` | `implementation_lr.{sender,receiver}_vision` |
| `lr_sweep_2_sender_vit` | 3 / 4 | the speaker's `ViT2` | `implementation_lr.sender_vision.ViT2` |
| `lr_sweep_3_attention_prototyper` | 5 / 6 | `AttentionPrototyper` | `implementation_lr.sender_prototyper.AttentionPrototyper` |
| `lr_sweep_4_sender_contrast` | 7 / 8 | `ExampleContrast` | `module_lr.sender_contrast` |
| `lr_sweep_5_sender_transformer_lm` | 9 / 10 | `SenderTransformerLM` | `implementation_lr.sender_language_model.SenderTransformerLM` |
| `lr_sweep_6_receiver_vit` | 11 / 12 | the listener's `ViT2` | `implementation_lr.receiver_vision.ViT2` |
| `lr_sweep_7_attention_discriminator` | 13 / 14 | `AttentionDiscriminator` | `implementation_lr.receiver_discriminator.AttentionDiscriminator` |
| `lr_sweep_8_receiver_cross_attention_lm` | 15 / 16 | `ReceiverCrossAttentionLM` | `implementation_lr.receiver_language_model.ReceiverCrossAttentionLM` |

Ten arms each -- 1e-5, 2e-5, 5e-5, 1e-4 and 2e-4 on both datasets, `01`-`05`
birds and `06`-`10` ShapeWorld -- at 60 epochs and one repeat. Sweep 1 is the
exception: nine arms at 100 epochs, and it also settled the gradient estimator.

**Why serial, and what goes wrong if it is not.** Each sweep runs at whatever
rates DEFAULT.toml currently holds, and restates none of them. Sweep 3 tunes
`AttentionPrototyper` in the presence of the ViT backbone sweep 2 measured, so
if sweep 2's result has not been written into DEFAULT.toml first, sweep 3
measures its component against an untuned trunk instead. That does not fail and
it does not warn -- it returns a number that looks like an answer. The protocol
is: run a sweep, read it, write the winning rate into
`[optimiser.implementation_lr]` in DEFAULT.toml with the evidence beside it,
then launch the next.

```
scripts/run_experiment.sh lr_sweep_2_sender_vit
# read the results, update DEFAULT.toml, then:
scripts/run_experiment.sh lr_sweep_3_attention_prototyper
```

**Why the rates are keyed by implementation.** `[optimiser.module_lr]` holds one
rate per module group, and the ladder swaps implementations *within* a group --
rung 1 puts `ResNet56` in `sender_vision`, rung 3 puts `ViT2` there. One number
per group can hold one architecture's rate or the other's, never both, so a
backbone rung would inherit the previous rung's rate and differ from it in two
things at once. `[optimiser.implementation_lr]` is keyed by group and then by
class name and is consulted first; see `models.builder.GROUP_IMPLEMENTATION` for
the six groups whose implementation the config chooses, and DEFAULT.toml beside
the table for the rates themselves. `sender_contrast` is the one component in
this set with no choice of class, which is why sweep 4 moves a `module_lr` key.

**Why 60 epochs.** On sweep 1's own traces the two leading ShapeWorld rates are
*inverted* at epoch 40 and cross at 47, so a 40-epoch sweep returns the wrong
answer; at 60 the winner leads by 0.032 on `train_acc_md_shape`. What 60 does
not buy is the argmax within the winning region -- sweep 1 took 98 epochs to
separate its top two birds arms, and then by 0.001. These sweeps pick a
plateau.

## The ablation ladder is two experiments

The ladder runs as **`ablation_shapeworld` and `ablation_birds`**, one folder
each, where it used to run as a single `ablation`. Submit them separately:

```
scripts/run_experiment.sh ablation_shapeworld
scripts/run_experiment.sh ablation_birds
```

Why. The two arms never shared a job in any meaningful sense -- rung 4 is rung 2
plus one change, not rung 3 plus a dataset -- but they shared a queue, so the
whole sixteen-job array was the unit of submission and of waiting. With
ShapeWorld not learning, the birds arm is the one that says whether an
architecture works at all, and it should be launchable, re-runnable and readable
without ShapeWorld's eight jobs in front of it.

**The rungs kept their numbers.** `ablation_shapeworld` holds the odd rungs 1-15
and `ablation_birds` the even 2-16, so each folder's numbering is gappy and
every existing reference to "rung 10" still names the same file. This is
deliberate: renumbering is what made the previous change to this ladder
expensive to read back (see the table below), and there was no reason to pay it
twice for a change that moves no config content.

What the split does change:

* **Results move.** Output is `<output_root>/<experiment>/...`, so new runs land
  under `ablation_shapeworld/` and `ablation_birds/` rather than `ablation/`.
  Runs recorded before the split are under `ablation/` and are not re-run by
  either experiment -- `run_experiment.sh`'s completed-job check looks only at
  the new paths, so an un-flagged submission will rerun rungs that already have
  results elsewhere.
* **`[slurm]` is now read per arm.** The block is taken from the first config in
  the folder by filename, which is rung 1 for ShapeWorld and rung 2 for birds.
  Both still carry the whole block, as every config in an experiment must.
* The top-level `name` key in each config was updated to match its folder. It is
  documentation -- `train.py` takes the experiment name from the config's path,
  not from that key.

## The ablation ladder was renumbered

The ablation now has **sixteen** rungs rather than fourteen, and the numbers
moved. Anything written before that -- run directories, the
`docs/` prose, `diagnostics/README.md`, commit messages -- names rungs in the old
scheme, so read those numbers against this table rather than against the current
configs:

| old | new | correspondence |
| --- | --- | --- |
| 1, 2 baseline | 1, 2 | exact |
| 3, 4 attention prototyper on a CNN sender | — | gone; the prototyper now sits on top of the ViT |
| 5, 6 sender ViT + prototyper | 5, 6 | exact |
| 7, 8 sender Transformer LM | 9, 10 | plus the contrast stage |
| 9, 10 receiver ViT | 11, 12 | plus the contrast stage |
| 11, 12 cross-attention listener | 15, 16 | plus the contrast stage, and the listener's two halves now enter separately at 13 and 15 |
| 13, 14 parallel speaker arm | — | gone; flip `[sender_language_model] bidirectional` on the top rung to get it back |

Two changes are structural rather than a renaming. The sender's vision swap now
comes **before** the prototyper, so an attention pooler is never measured over
CNN features; and the listener's message encoder and discriminator, which one
`comparer` key used to change together, now enter as two rungs so that "attention
helps" can be attributed to one of them. Rungs 7 and 8 are new entirely.

A consequence worth stating plainly: no rung above 6 is configuration-identical
to any old rung, so old and new trajectories should not be pooled.

## Shape of the file

One row per epoch, appended by `code/train.py` as the epoch finishes (the header
is written only when the file is created). Because rows are appended rather than
rewritten, earlier epochs survive a resume; a crash between writing the row and
writing `checkpoint_last.pt` can re-emit one epoch's row, which is harmless but
worth knowing if you index by `epoch`.

Two columns are unprefixed:

| Column | Meaning |
| --- | --- |
| `epoch` | 0-based epoch index. |
| `timestamp` | Local wall-clock time (ISO-8601) at which the row was written, i.e. when the epoch finished. Diff consecutive values to get epoch duration. |

Every other column is `<split>_<metric>`.

## Splits

| Prefix | Pass |
| --- | --- |
| `train_` | The training pass. |
| `test_` | Held-out **novel** concepts — the generalisation split. |
| `test_same_` | Held-out instances of concepts **seen in training**. For ShapeWorld these are freshly generated worlds; for CUB, photographs held out of the training species (see the birds-splits section of the top-level README). Optional in principle: absent for a dataset that ships no `test_same` split, in which case these columns are missing entirely. |
| `test_avg_` | The unweighted mean of `test_` and `test_same_` for the same metric. Applied to every eval metric, topsim included (so it is a mean of two Spearman rhos, not a rho over the pooled data). With no `test_same` split it equals `test_` — which is what birds runs recorded before `cub.py` grew one look like, so do not pool them with newer birds runs. |

There is no `val` split and no best-epoch selection — training runs to a fixed
endpoint and the trajectory *is* the deliverable — so there are no `best_*`
columns. There is also no cross-game-type eval: a run trains and evaluates a
single game framing.

## Loss and accuracy

Present for all four split prefixes.

| Column | Meaning |
| --- | --- |
| `<split>_loss` | The training objective, averaged over the epoch's batches (batch-size weighted). `BCEWithLogitsLoss` over the listener's per-image scores, or `CrossEntropyLoss` over the candidate set when `reference_game_xent = true`. |
| `<split>_combined_loss` | Duplicate of `<split>_loss`. Vestigial — the two are logged from the same value in `train.py`. Ignore it. |
| `<split>_acc` | Listener accuracy. Under BCE: the fraction of listener images whose sign is predicted correctly, meaned per game then over games. Under `reference_game_xent`: the fraction of games where the target is the argmax. |

## Per-difficulty accuracy (ShapeWorld only)

`<split>_acc_md_<type>` breaks `<split>_acc` down by the **logical type of the
concept** the game is about (`md` = metadata). Derived in
`code/data/shapeworld.py:get_metadata` from the ground-truth logical form; `not`
is ignored, so `not red` counts as `color`. CUB runs have no `acc_md_*` columns.

| Suffix | Concept type | Example logical form |
| --- | --- | --- |
| `md_color` | single colour feature | `red`, `not blue` |
| `md_shape` | single shape feature | `triangle` |
| `md_and_color_color` | conjunction of two colours | `and red blue` |
| `md_and_color_shape` | conjunction of a colour and a shape | `and red triangle` |
| `md_and_shape_shape` | conjunction of two shapes | `and square triangle` |
| `md_or_color_color` | disjunction of two colours | `or red blue` |
| `md_or_color_shape` | disjunction of a colour and a shape | `or red triangle` |
| `md_or_shape_shape` | disjunction of two shapes | `or square triangle` |

Only the operator and the operand *types* are recorded — `and_color_shape`
covers `and red triangle` and `and triangle red` alike. Which of these appear
depends on which concept types the dataset's split actually contains.

## Topographic similarity

Topsim is a property of the language, so it is computed on the eval passes only:
there are no `train_topsim_*` columns.

Every topsim column is a Spearman correlation between pairwise **meaning**
distances and pairwise **signal** (message) distances, measured over concept
*prototypes* — one point per concept, whose message is the modal token sequence
its instances emitted, whose symbol embeddings are the mean over the instances
that emitted that sequence, and whose concept vector is the mean over its
instances. Concepts are capped at `[analysis] max_concepts`.

A value is `NaN` where Spearman is undefined: fewer than two finite pairs, or
either side constant across all pairs. That is the correct reading, not an error.

The column name encodes three things: the meaning space, the signal set, and
whether the embeddings were decontextualised.

### Signal set (`s1`–`s6`)

Six signal distances, differing in what they treat as a difference between two
messages (see `code/emergence.py`):

| Suffix | Signal distance | Insensitive to |
| --- | --- | --- |
| `s1` | soft MoverScore, 1-grams | symbol order and synonymy |
| `s2` | soft MoverScore, 2-grams | blockwise order and synonymy |
| `s3` | soft Levenshtein | synonymy (order matters) |
| `s4` | hard MoverScore, 1-grams | symbol order |
| `s5` | hard MoverScore, 2-grams | blockwise order |
| `s6` | hard Levenshtein | nothing — **this is classic topsim** |

"Soft" = the cost of aligning two symbols is a function of the sender's own
contextual symbol embeddings, so near-synonymous symbols are cheap to align.
"Hard" = the same functions under a fixed 0/1 ground cost, so only symbol
identity matters.

### Meaning space (`topsim_` vs `topsim_gt_`)

| Family | Meaning distance | Question it answers |
| --- | --- | --- |
| `<split>_topsim_s1` … `_s6` | cosine between the sender's concept vectors | Does the language track what the sender *represents*? |
| `<split>_topsim_gt_s1` … `_s6` | word-level edit distance between ground-truth logical forms | Does the language track the *concepts*? |

They come apart when the sender has collapsed onto a subset of the visual
features: the cosine space still scores well, only the ground-truth space
notices. `topsim_gt_s6` is the variant directly comparable to the topographic
rho reported in Mu & Goodman (2021).

`topsim_gt_*` requires concepts with internal structure, so it is emitted for
ShapeWorld but **not** for CUB, whose concept keys are bare class ids.

### `_static` (leakage control, soft variants only)

`<split>_topsim_s1_static`, `_s2_static`, `_s3_static` and their `topsim_gt_`
counterparts repeat the three soft variants with each symbol's contextual
embedding replaced by the corpus mean for that token id. This keeps synonymy
tolerance but removes the channel by which the concept can reach the signal
distance without passing through the symbols actually emitted (the GRU sender
initialises its hidden state from the concept vector, so the embedding behind the
first content symbol is a function of the concept alone).

`raw − static` is the contextuality inflation. The hard variants (`s4`–`s6`) see
nothing but token identities, so they need no control and have no `_static` form.

## Older files: `_adj` instead of `_static`

CSVs written before the `_static` control replaced the pre-training baseline
carry `<split>_topsim_s1_adj` … `_s6_adj` and no `topsim_gt_*` family. `_adj` is
the raw value minus that split's topsim measured on one eval pass *before*
training started. That baseline was dropped because it could not bound the
leakage it was meant to bound — an untrained sender reads ≈0 on every variant —
so `_adj` and `_static` are not interchangeable and should not be pooled.

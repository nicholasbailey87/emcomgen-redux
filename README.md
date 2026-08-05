# Emergent Communication of Generalizations, Redux

This repository represents experiments based on [the work "Emergent Communication of Generalizations" published at NeurIPS 2021](https://github.com/jayelm/emergent-generalization).

## Getting started

Things can get a bit funky if you don't install the required packages in the right order. PyTorch is left out of the requirements as the version of PyTorch you use is highly dependent on the of your environment (e.g. Cuda toolkit version).

* On Windows, first get Cuda 13.0 from here https://developer.nvidia.com/cuda-toolkit-archive
* You will need an environment with Python 3.12
* Then create a virtual env with `python -m venv venv`
* Then `pip install -f requirements.txt`
* Then install pytorch with `pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130`
* Then install mup with `pip install mup-1.0.0` (after PyTorch, so that it doesn't try to pick its own PyTorch version)
* Then install flash attention if using Ampere GPUs or newer, with `pip install flash-attn --no-build-isolation` **NOTE: This must be done in an environment with C++ build tools available, e.g. in "x64 Native Tools Command Prompt" on Windows. This may take some time.**

Other instructions for getting started still apply per the original repository: https://github.com/jayelm/emergent-generalization, see below

# README from the original repository below

## Setup

- Download and process birds (CUB) data [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), then unzip into `/data/cub` directory (i.e. the filepath should be `data/cub/CUB_200_2011/*`), then run `python save_cub_np.py` in the `/data` directory to save cub images to easily accessible npz format.
- Download and process ShapeWorld data: use `data/download_shapeworld.sh`. There are 3 datasets:
    - [shapeworld](http://nlp.stanford.edu/data/muj/emergent-generalization/shapeworld/shapeworld.tar.gz): 20k games over the 312 conjunctive concepts
    - [shapeworld_ref](http://nlp.stanford.edu/data/muj/emergent-generalization/shapeworld/shapeworld_ref.tar.gz): 20k games over the 30 conjunctive concepts possible for
        reference games only
    - [shapeworld_all](http://nlp.stanford.edu/data/muj/emergent-generalization/shapeworld/shapeworld_all.tar.gz): 20k games over the 312 conjunctive concepts, *but no
        compositional split*.

Code used for generating the ShapeWorld data is located [here](https://github.com/jayelm/minishapeworld/tree/neurips2021).

#### Shrink ShapeWorld to 40 images per game (required)

ShapeWorld experiments train on `shapeworld_40`, a 40-image-per-game copy of
`shapeworld`. Build it once after downloading:

```bash
python data/shrink_shapeworld.py --src ../data/shapeworld --dst ../data/shapeworld_40
```

The stored games hold 80-100 images but `split_spk_lis` only ever reads 20 (10
per agent per polarity), so 40 is the smallest row that preserves the concept
game's speaker/listener disjointness. That takes the train store from ~20-25 GB
to ~9.8 GB, which fits in RAM — hence `load_shapeworld_into_memory = true` and
~32 GB less disk I/O per epoch. The script draws its 20 keepers per half
*proportionally within descriptor strata* rather than slicing the first 20,
which would drop an entire stratum of the paper's 1/3-1/3-1/3 hard-target
sampling; run it with `--audit` to see the ordering in your own copy of the
data. Originals are left untouched.

The unmodified `shapeworld` and `shapeworld_ref` datasets are only needed for
reference-game runs and downstream analysis (`acre.py`); concept-game training
does not read them.

## Running experiments

### Quickstart (SLURM array jobs)

Experiments are organised as follows: a top-level `config.json` declares the
storage paths, each experiment is a folder under `experiments/<name>/configs/`
holding one or more TOML configs, and "repeats" are realised as SLURM array
indices mapped to seeds.

```bash
# one-time: stage data from slow storage to fast storage
./scripts/prepare_data.sh
# launch an experiment as a SLURM array (repeats per config -> one array task each)
./scripts/run_experiment.sh transformer            # or: ... 10 gpu-a100 --rerun
# monitor
squeue -u $USER
tail -f experiments/transformer/logs/transformer_*.o
```

Results land under `<output_root>/<experiment>/<config_stem>_seed<seed>/`
(e.g. `~/archive/results/emcomgen/transformer/transformer_seed0/metrics.csv`), with the
topographic-similarity and generalization columns described below.

`./scripts/run_experiment.sh` takes
`<experiment> [max_concurrent] [partition] [--rerun] [--gpu-type T]` (defaults:
`max_concurrent=5`, `partition=gpu-a100`). Without `--rerun` it skips jobs whose
`metrics.csv` already has at least `[scheduler].epochs` rows; `--rerun` resubmits
everything from scratch (passing `--no_resume` to `train.py`).

#### Storage paths (`config.json`)

```json
{
    "data_slow_storage": "~/archive/data",
    "data_fast_storage": "~/sharedscratch/data",
    "output_root": "~/archive/results/emcomgen"
}
```

On Hyperion the CUB training data lives at `~/archive/data/emcomgen/data/cub`
(`data_slow_storage` + `/emcomgen/data/cub`); `prepare_data.sh` stages it to
`~/sharedscratch/data/emcomgen/data/cub` (`data_fast_storage`), which is what jobs
read. `train.py` rewrites each config's `dataset`/`ref_dataset` logical name
(`cub`/`shapeworld_40`/`shapeworld`/`shapeworld_ref`) to its fast-storage
location at runtime.

### 1. Train model (one SLURM array task)

Each array task runs `train.py` once:

```bash
python code/train.py --config experiments/<exp>/configs/<file>.toml --seed <n> [--no_resume]
```

`--seed` seeds `random`/`numpy`/`torch` so repeats differ; the SLURM wrappers map
array index → `(config, seed)` via `scripts/job_utils.py`. The output directory is
derived from the config path: `<output_root>/<experiment>/<config_stem>_seed<seed>/`.

Game type and the other hyperparameters are set **in the TOML config**, not via
CLI flags (the config inherits from the repo-root `DEFAULT.toml`):

- `[data] percent_novel = 1.0`: concept game (fraction of images novel to the student)
- `[data] percent_novel = 0.0`: setref game
- `[data] percent_novel = 0.0` + `reference_game = true`: reference game. **For
    ShapeWorld, use the 30-concept reference dataset `shapeworld_ref`, not the
    standard 312-concept `shapeworld`!**
- `[sender_language_model] message_length` / `[receiver_comparer] message_length`:
    max message length (**includes sos/eos, so true length is this minus 2**; the
    two must match)
- `[sender_language_model] vocabulary`: vocab size of the agents
- `[data] n_examples`: number of examples given to agents
- `[sender_language_model] uniform_weight`: uniform noise on the gumbel-softmax policy
- `wandb = true`: activate wandb logging (run `wandb init` yourself)

Two extra sections drive the launcher (the `[experiment]` and `[slurm]` keys):

```toml
[experiment]
repeats = 5            # array tasks per config

[slurm]
time = "24:00:00"
cpus_per_task = 6
gpus_per_task = 1
mem_gb = 24
```

See `DEFAULT.toml` for the full set of options and their defaults.

#### Topsim & generalization from `train.py` alone

Only two things are measured: **generalization accuracy** and **six variants of
topographic similarity**. Both come out of `train.py`; measurement lives in
`code/emergence.py`.

Classic topsim uses one signal (message) distance — Levenshtein — which is
sensitive to both symbol order and symbol identity, so it scores a language with
free symbol order or with synonyms as non-compositional even when it is not.
Instead we compute one variant per *signal set* S1–S6. They differ **only** in
the signal distance; the meaning distance is cosine between the sender's concept
vectors (`Sender.get_concepts`) throughout.

| metric | set | signal distance | set characterised by |
|---|---|---|---|
| `topsim_s1` | S1 "bag of meanings" | soft MoverScore, 1-grams | free order + synonymy |
| `topsim_s2` | S2 "meaning-block rearrangement" | soft MoverScore, 2-grams | blockwise free order + synonymy |
| `topsim_s3` | S3 "configurational" | soft Levenshtein | strict order + synonymy |
| `topsim_s4` | S4 "bag of symbols" | hard MoverScore, 1-grams | free order, no synonymy |
| `topsim_s5` | S5 "asynonymous blockwise" | hard MoverScore, 2-grams | blockwise free order, no synonymy |
| `topsim_s6` | S6 "asynonymous order-sensitive" | hard Levenshtein | strict order, no synonymy — the classic topsim |

"Soft" means synonymy-tolerant: the alignment cost between two symbols is a
function of the sender's own contextual symbol embeddings. "Hard" is the same
function under a fixed, embedding-free ground cost. The message, its symbol
embeddings and the concept behind it all come from a single `Sender.speak`
forward pass — calling the accessors separately would resample the vision
dropout mask and (for the GRU sender) the message itself, so the three would not
correspond.

Measurement is **per concept prototype**, not per image: one point per concept,
whose message is the modal token sequence its instances emitted, whose symbol
embeddings come from one instance that actually emitted that sequence, and whose
concept vector is the mean over its instances. Pairing individual images would
put many pairs at meaning distance zero (same concept, different image) against
a non-zero signal distance, which only depresses the correlation. Concepts are
capped at `[analysis] max_concepts` (default 500), which bounds measurement at
roughly 30–55 s per eval pass.

**Adjusted topsim.** The soft variants are parameterised by the sender's own
symbol embeddings, which are not independent of its referent embeddings, so some
correlation is available before any language exists. Every run therefore does one
eval pass per split *before training starts* and records that as a baseline;
`topsim_s{1..6}_adj` is the raw value minus that baseline. Baselines are stored
in `checkpoint_last.pt` so a resume does not re-measure them against an
already-trained sender.

#### Metrics

Metrics are logged into `<output_root>/<experiment>/<config_stem>_seed<seed>/metrics.csv`
and to wandb (if `wandb = true`). Each is prefixed with its split — `train`,
`test` (novel concepts), `test_same` (seen concepts), and `test_avg` (the mean of
the latter two):

- `{train,test,test_same,test_avg}_acc` — generalization accuracy. For ShapeWorld
    there are also per-game-difficulty breakdowns, `acc_md_*`.
- `{test,test_same,test_avg}_topsim_s1` … `_topsim_s6` — the six variants above.
    Not computed on the train pass: topsim is a property of the language, so
    there is nothing to gain from measuring it mid-training-pass.
- `{test,test_same,test_avg}_topsim_s1_adj` … `_topsim_s6_adj` — the same six,
    minus the pre-training baseline. `NaN` if no baseline was recorded.
- `{train,test,...}_loss` — the training objective.

A topsim value is `NaN` where Spearman is undefined: fewer than two finite pairs,
or either the meaning or the signal side constant across all pairs. That is the
correct result rather than an error — see the chapter's Case 1 edge case.

#### No val split, no best-epoch selection

Training runs to a fixed endpoint (`[scheduler] epochs`) and the per-epoch
`metrics.csv` trajectory *is* the deliverable — this is open-ended language
evolution, so there is nothing to cherry-pick a best epoch against. Consequently
there is no `val` split, no `best_*` metric columns, and no `best_model.pt` /
`best_lang.csv`. What gets written is the periodic `<epoch>_model.pt` /
`<epoch>_lang.csv` snapshots (`save_interval`) plus a `final_model.pt` /
`final_lang.csv` at the end.

Nor is there cross-game-type eval. A run trains and evaluates a single game
framing; for the concept game (`percent_novel = 1.0`) speaker and listener see
fully disjoint targets *and* distractors, which is itself the control against
context-dependent degenerate codes that cross-eval was there to provide. That
takes the epoch from 13 passes over the data down to 3.

### 2. Sample language from model

The above command produces a `metrics.csv` with most metrics, but I measure
entropy and AMI at the end by sampling a bunch of language from the model and
analyzing that corpus. To do so, run

```
# (no --cuda flag needed; will use whatever flag was set at train time)
python code/sample.py exp/NAME
```

which by default samples 200k messages from a trained model into
`exp/NAME/sampled_lang.csv` and some summary statistics into
`exp/NAME/sampled_stats.json`.

Now, if you just want the information theoretic systematicity metrics, for both
Birds and ShapeWorld run
`python code/acre.py exp/NAME/sampled_lang.csv --dataset DATASET --cuda --stats_only`
which **does not run ACRe**, but rather just dumps some summary statistics:

- `exp/NAME/sampled_lang_overall_stats.json`: this contains entropy,
    unnormalized mutual information, and adjusted mutual information
- `exp/NAME/sampled_lang_stats.csv`: this is a list of utterances generated for
    each concept, with their counts. Also entropy information. This can be used
    to plot the sunburst (i.e. nested pie) plots in the paper. See "4.
    Visualizing Model Outputs"

Again, we haven't actually run ACRe. If you want to run ACRe, read on:

### 3. Train ACRe

If you actually want to train an ACRe model you should train your model with
the `shapeworld_all` dataset, which doesn't involve the compositional split
(though you can still do ACRe analysis on models trained normally).

Run ACRe without the `--stats_only` flag. Rather, run

`python code/acre.py exp/NAME/sampled_lang.csv --dataset DATASET --cuda`

which trains an ACRe model to reconstruct the agent language according to the
concepts of `DATASET`. This prints out some top1 acc/loss metrics and the
following files:

- `exp/NAME/sampled_lang_{train,test}_acre_metrics.csv`: overall loss/top1 acc
    for ACRe reconstruction compared to the ground truth language only (i.e.
    not evaluating a listener model yet), as well as these metrics broken down
    by concept
- `exp/NAME/sampled_lang_{train,test}_sampled_lang.pt`: Contains ground truth
    model language for both train/test ACRe splits, as well as ACRe
    reconstructions. This gets used to evaluate a listener in the next section.
- `exp/NAME/acre_split.json`: The split of train/test concepts used for ACRe.

### 4. Evaluate ACRe on Listener

Run

`python code/eval_zero_shot.py exp/NAME --cuda`.

which evaluates across `--epochs` epochs (default 5), categorizing concepts by
whether they belong to the ACRe train or test split, and evaluates several
types of language on the listener:

- `ground_truth_1`: the model lang located in `exp/NAME/sampled_lang_{train,test}_sampled_lang.pt`.
- `same_concept`: language sampled from other model utterances from the same concept
- `acre`: ACRe reconstructed language.
- `random`: random language uniformly sampled from the possible set of
    utterances (not reported in paper; worse than `any_concept`)
- `any_concept`: random language sampled from utterances from any concept (the random baseline in the paper)
- `closest_concept`: language sampled from utterances for the "closest" concept as measured by edit distance
- `ground_truth_2`: (sanity check) re-sample language from the teacher; should be close to `ground_truth_1` performance.

These results are saved into

- `exp/NAME/zero_shot_{train,test}.json`: BLEU-1 and listener acc aggregated
    across all concepts, and for each concept individually
- `exp/NAME/zero_shot_lang_type_stats.csv`: a lang stats file similar to
    `exp/NAME/sampled_lang_stats.csv` described above, which can be used to
    visualize outputs for the various language distributions as described in
    the next section.

### 5. Visualizing model outputs

This requires `R` and the `sunburstR` package, as well as a generated
`sampled_lang_stats.csv` which is produced by `acre.py` (just the
`--stats_only` flag will do). Then an example usage is located in lines
425--441 of `analysis/analysis.Rmd`.

### 6. Evaluating across different games

Accuracy and topographic similarity metrics are evaluated zero-shot across
different games in the main train script, though entropy/AMI metrics aren't
collected. To obtain those, and to get all the results in one place, sample
language while using a `--force_*` flag to force the game to be ref, setref, or
concept. This adds a `_force_{ref,setref,concept}` prefix to every file
outputted by `sample.py`, e.g. `sampled_lang_force_ref.csv`. For example:

```
python code/sample.py exp/NAME --force_reference_game
python code/acre.py exp/NAME/sampled_lang_force_ref.csv --dataset data/shapeworld_ref
```

which now produces `exp/NAME/sampled_stats_force_ref.json`,
`exp/NAME/sampled_lang_force_ref_overall_stats.json`,
`exp/NAME/sampled_lang_force_ref_stats.csv`, etc.

**If you're analyzing ShapeWorld, remember to specify the right dataset - either
ref, or setref/concept - when printing summary statistics via `acre.py`**.

## Dependencies

This code was tested with python 3.8 and `torch==1.8.1`. A specific
environments file is located in `requirements.txt`, but other common package
versions are likely to be compatible as well.

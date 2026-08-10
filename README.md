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

#### Silhouette augmentation (ShapeWorld only)

The paper reports that ShapeWorld setref and concept games converge to a
colour-only local minimum at ~83% accuracy (appendix A.1). That number is not a
coincidence: `app:hard` samples 1/3 of a conjunction's distractors so they fail
only the shape conjunct, i.e. they are colour-matched to the target, and a
colour-only policy therefore scores 10 targets plus 2/3 of 10 distractors =
16.67/20 = 83.3%. One third hard negatives does not close the shortcut off.

`data.silhouette_p_receiver` (default `0.5`) renders an agent's whole view as
white-on-black silhouettes with that probability, per game, at training time
only. Thresholding is what removes colour — a plain grayscale conversion does
not, since the six colours sit at six distinct luma values (blue 29 through
white 255) that a single conv filter can separate.

The receiver is the side to constrain. Silhouetting the *sender* would teach it
shape-from-silhouette, which is not the shape-from-colour-image competence that
eval requires and that `probe_shape.py` measures, and it would shift the
sender's input distribution out from under that probe. Silhouetting the receiver
leaves the sender's inputs untouched but makes a colour message unrewardable, so
the shape gradient still reaches the sender's vision model through the channel.
It also denies the receiver the option of clustering its own set by colour and
treating the message as a coarse pointer.

`data.silhouette_p_sender` (default `0.0`) is the same knob for the other side,
so the pair selects the regime: `(0, p)` receiver-only, `(p, 0)` sender-only,
`(p, p)` either or both. A colour concept needs *both* agents to see colour, so
it is answerable `(1 - p)` of the time under the first two and `(1 - p)^2` under
the third — the symmetric setting pays a worse ceiling for the same shape
pressure.

The roll is per game rather than per image: with 10 targets in a set, rolling
per image would leave ~`(1-p) x 10` of them coloured and the colour cue still
recoverable from the set.

Eval is never silhouetted, so reported numbers stay comparable to the paper's
and to the `probe_shape.py` sweep. Both rates are `0.0` for CUB — the minimum is
ShapeWorld-specific, species genuinely depend on plumage colour, and
thresholding a photograph would destroy texture and pattern rather than isolate
colour.

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
`experiments/README.md` is the per-column reference for that CSV.

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
- `[sender] prototype_dropout` / `[sender] vision_dropout`: dropout on the pooled
    concept vectors (between prototyper and language model) and on the per-image
    embeddings (between vision model and prototyper) respectively. The former is
    the stronger regulariser and is where jayelm's single speaker-side
    `--dropout` sits; dropping pre-pool features is largely undone by the
    average over n/2 examples.
- `[receiver_comparer] dropout`: the listener's **only** dropout, and the
    counterpart of the sender's `prototype_dropout`. Applied equally to both
    operands of the comparison — the pooled message embedding and the incoming
    referent embeddings — and defaults to `0.5` to match the sender. There is
    deliberately no `[receiver] vision_dropout`: it would mask the same tensor
    as the referent side of this knob, with nothing but a reshape between them,
    so the two would compose into one mask at a rate neither knob names. The
    sender keeps its `vision_dropout` because the prototyper pools between the
    two masks there. It affects **only** those two inputs: module internals are
    fixed constants, so raising it never silently rewires the architecture. The
    listener GRU's inter-layer dropout is pinned to `0.0` (jayelm's listener GRU
    takes no dropout argument), and the cross-attention comparer's attention
    dropout to `MSA_DROPOUT = 0.1`. For reference, jayelm regularised both agents with a
    single `--dropout` at `0.1`, on the listener's vision pathway only, with
    nothing at all on its language pathway.
- `[sender_language_model] token_exploration_rate`: the exploration knob, stated
    as the **expected fraction of symbols the Gumbel noise flips**. `0.1`
    everywhere, ShapeWorld and CUB alike. It is *calibrated*, not assumed:
    `F.gumbel_softmax(..., hard=True)` emits `argmax(logits + g)` with
    `g ~ Gumbel(0,1)`, whose standard deviation is a fixed 1.283, so channel
    fidelity is set entirely by the scale of the speaker's logits — which
    varied by two orders of magnitude across an architecture ladder, giving one
    arm a 0.99-fidelity channel and another 0.24. Each training batch bisects
    for the logit gain that hits the requested rate (`exploration_gain`, an EMA
    buffer) and the result is logged per epoch alongside the `realised_survival`
    it achieved. Applied **on the train pass only**: eval decodes greedily, so
    it measures the learned policy. Note the target is a *mean* over slots and
    confidence is skewed, so 0.1 means a median slot at 0.98 and a p10 tail at
    0.57 — exploration concentrates where the model is unsure.
- `[sender_language_model] uniform_weight`: weight of the uniform component
    mixed into the policy before sampling, train-pass only. Now a *bounds* knob
    rather than the exploration knob: it caps a slot's winner at `1 - w + w/V`
    and floors its losers at `w/V`, so it puts a hard floor of `w * (1 - 1/V)`
    under `token_exploration_rate` — 0.0186 at the default `0.02` and `V = 14`.
    `V` is `vocabulary`, counting the emittable tokens only: the four reserved
    slots are masked before the mixture, which is spread over what is left, so
    these divide by 14 rather than by 18.
    That floor is a permanent per-symbol corruption rate that training cannot
    reduce, which is the point: it keeps late training from committing the
    channel entirely. Requesting a rate below the floor warns at construction.
- `[sender_language_model] layer_norm_logits`: normalise the emittable
    vocabulary logits per example and per position before sampling. `true`.
    This is what fixes the magnitude the gain is calibrated against, so the gain
    cannot drift to compensate for a collapsing logit scale. Replaces a
    `batch_norm_logits` here (the identically-named keys in the two
    `*_feature_model` sections are broccoli's own and are unrelated): LayerNorm
    is ~2.6× tighter on the criterion that matters, is position-invariant for
    both speakers, has no running statistics so train and eval agree, and does
    not couple to `accumulator_steps`.
- `[sender_language_model] tau`: gumbel-softmax tau (jayelm's `--tau`). This
    shapes the straight-through *gradient* only: the hard forward sample is an
    argmax, so it is invariant to tau. Raising it does not buy exploration, it
    just flattens the surrogate gradient. `1.0` everywhere, GRU and Transformer
    alike, so the two speakers differ only in architecture. Leave it there:
    compensating it for the exploration gain only collapses the gradient.
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
the signal distance; within one reading the meaning distance is held constant
across all six (see **Two meaning spaces** below).

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
embeddings are the mean over *every* instance that emitted that sequence
(averaging rather than picking one arbitrary instance is what keeps a single
epoch's reading stable), and whose concept vector is the mean over its
instances. Pairing individual images would put many pairs at meaning distance
zero (same concept, different image) against a non-zero signal distance, which
only depresses the correlation. Concepts are
capped at `[analysis] max_concepts` (default 500), which bounds measurement at
roughly 30–55 s per eval pass.

**Two meaning spaces.** The same six signal distances are read against two
meaning distances, and both families are reported:

| prefix | meaning distance | question it answers |
|---|---|---|
| `topsim_` | cosine between the sender's concept vectors (the third output of `Sender.speak`) | does the language track what the sender *represents*? |
| `topsim_gt_` | word-level edit distance between the ground-truth logical forms | does the language track the *concepts*? |

They come apart when the sender has collapsed onto a subset of the visual
features: the cosine space scores it well for faithfully encoding whatever it
kept, and only the ground-truth space notices the collapse. `topsim_gt_s6` —
hard Levenshtein against the `Edit` concept distance — is the variant directly
comparable to the topographic rho reported by Mu & Goodman (2021). The
`topsim_gt_` family needs concept keys with internal structure, so it is emitted
for ShapeWorld but not for CUB, whose keys are bare class ids.

**`_static`: the leakage control.** The soft variants are parameterised by the
sender's *contextual* symbol embeddings, which differ from a fixed lookup table
in two ways at once. They tolerate synonymy, which is the point. They are also
sensitive to the concept being described, which is not: `SenderGRULM` initialises
its hidden state from the concept vector and its SOS input is constant, so the
embedding behind the *first* content symbol is a function of the concept alone,
with no token in it. A soft variant can therefore read a correlation straight out
of the meaning space without the language taking any part.

So each soft variant is reported twice: raw, and as `_static` — recomputed with
every symbol's embedding replaced by the corpus mean for its token id. That
strips the concept sensitivity while leaving synonymy tolerance intact, and
`raw − static` is the contextuality inflation. The hard variants see nothing but
token identities, so they have no leakage to control and no `_static` form.

This replaces an earlier adjustment against the *untrained* model's topsim
(`topsim_s{1..6}_adj`, still present in CSVs from before the change). That
baseline could not bound the leakage, because training is what creates it: an
untrained `SenderGRULM` has a random `init_h` and a saturating tanh, which
between them read ≈0 on every variant. `_adj` and `_static` are not
interchangeable and should not be pooled.

#### Metrics

Metrics are logged into `<output_root>/<experiment>/<config_stem>_seed<seed>/metrics.csv`,
one row per epoch, appended as the epoch finishes so earlier rows survive a
resume. Each is prefixed with its split — `train`, `test` (novel concepts),
`test_same` (seen concepts), and `test_avg` (the mean of the latter two):

- `{train,test,test_same,test_avg}_acc` — generalization accuracy. For ShapeWorld
    there are also per-concept-type breakdowns, `acc_md_*` (`md_color`,
    `md_and_color_shape`, …), keyed by the logical type of the concept the game
    is about.
- `{test,test_same,test_avg}_topsim_s1` … `_topsim_s6` — the six variants above,
    against the sender's concept vectors. Not computed on the train pass: topsim
    is a property of the language, so there is nothing to gain from measuring it
    mid-training-pass.
- `{test,test_same,test_avg}_topsim_gt_s1` … `_topsim_gt_s6` — the same six
    against the ground-truth logical forms. ShapeWorld only.
- `{test,test_same,test_avg}_topsim{,_gt}_s1_static` … `_s3_static` — the three
    soft variants under the decontextualised embeddings. Soft variants only.
- `{train,test,...}_loss` — the training objective. `_combined_loss` is a
    duplicate of it, kept only so older analysis scripts keep working.
- `train_exploration_gain` — the speaker's calibrated logit gain (the EMA
    buffer). This is the scale diagnostic, and the direct readout of how far
    apart two architectures' channels really are. Train pass only, since that is
    the only pass that calibrates. If it pins at either end of `[1e-2, 1e4]`,
    something upstream of the channel is wrong.
- `train_realised_survival` — the mean winning-token probability at the gain
    actually in use. Confirms the calibration converged; it should sit at
    `1 - token_exploration_rate`.
- `{train,test,test_same,test_avg}_unique_message_fraction` — distinct messages
    over messages emitted. A language that is doing work compresses: healthy
    runs sit around 0.30–0.40, while runs whose channel is too noisy to learn
    through sit at 0.88–1.00, emitting near-noise.

`test_avg` is the unweighted mean of `test` and `test_same` for the same metric,
applied to every eval metric including topsim — so it is a mean of two Spearman
rhos, not a rho over the pooled data. `test_same` is optional; where a dataset
ships no such split those columns are absent and `test_avg` equals `test`.

Two columns are unprefixed: `epoch`, and `timestamp` — the wall-clock time
(ISO-8601, local) at which that epoch's row was written, for reading a run's pace
off the CSV afterwards.

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

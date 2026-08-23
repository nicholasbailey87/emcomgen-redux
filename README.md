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

## When a run sits at chance

See [`diagnostics/`](diagnostics/README.md). A pair stuck at 0.5 accuracy and a loss near ln 2 offers no evidence about which of its parts is at fault, because every one of them fails that way. Those scripts build the real modules from a real rung config and run them on a synthetic task with a known answer, so a failure has one cause rather than five. No dataset, no GPU, about a minute on a laptop.

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

#### Birds splits (CUB only)

A CUB concept is a species: a game picks one class and draws every positive from
it. The splits are built in `code/data/cub.py` at load time — the per-class
`img.npz` archives that `save_cub_np.py` writes hold every image of their
species, and nothing about the split is baked into them.

| Split | Species | Images of those species | Share of the corpus |
| --- | --- | --- | --- |
| `train` | 1–150 | ~80% | ~60% |
| `test_same` | 1–150 | ~20% | ~15% |
| `test` | 151–200 | all | ~25% |

`test` is jayelm's split unchanged, so the generalisation number keeps its
footing. Two things about the other two are ours.

**`test_same` holds out photographs, not species.** The paper reports an
Acc (Seen) column for birds, but the released code never built a seen-concept
split for CUB, so there was no implementation to port. For ShapeWorld the split
comes free — `test_same.npz` holds freshly generated worlds, so its games are
unseen *images* of seen *concepts* — but CUB has a finite pool of photographs per
species, so the same property has to be bought by holding images out of training.
Each training species gives up `max(n_examples, round(0.2 x n))` of its images.
The floor matters: species carry 41–60 images, a game draws `n_examples` = 10
*distinct* positives from one species, and a flat 20% of the smallest species
would hold out 8 and raise. The partition is a blake2b hash of the image name, so
it is identical across seeds, resumes and dataloader workers — an RNG would give
each seed a different split and every seed would train on some other seed's test
set. Distractors are held out along with targets, which falls out of building the
dataset from the held-out image pool rather than filtering at sampling time.

**Species 101–150 have moved into `train`.** They were a val split; there is no
best-epoch selection any more, so they sat unused. Folding them in is what pays
for the holdout — 150 species at 80% of their images is *more* training data than
100 whole species was (~60% of the corpus against ~50%), with half again as much
species diversity. The cost is that our birds train set is no longer the paper's
100 classes. That comparison was already gone: we train the vision backbone from
scratch where the paper fine-tunes a pretrained network, and we play a different
number of games per epoch.

Eval is sized per species (`eval_games_per_species`, 16), so `test` draws 800
games and `test_same` 2,400. Equal coverage per species rather than equal totals,
because topsim builds one prototype per concept from the modal message over that
concept's instances, so it is games-per-species that has to match for the two
splits to be read against each other.

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
- `[sender_language_model] message_length` / `[receiver_language_model] message_length`:
    max message length (**includes sos/eos, so true length is this minus 2**; the
    two must match)
- `[sender_language_model] vocabulary`: vocab size of the agents
- `[data] n_examples`: number of examples given to agents
- `[sender] prototyper`: `AveragePrototyper` pools a concept's examples by
    averaging them; `AttentionPrototyper` pools them with a learned attention
    score per example (broccoli's `SequencePool`, one per polarity). The
    attention version is initialised to *be* the average — its scoring weights
    are zeroed, so the softmax over examples opens exactly uniform — and scores
    from `LayerNorm`ed embeddings while pooling the raw ones. Both are there for
    the same reason `layer_norm_logits` is: without them the softmax over
    examples inherits the backbone's output magnitude, which decides both where
    the pooling opens (near-selection of one example at an unnormalised CNN's scale, near the
    mean at a normalised backbone's) and how fast it leaves the mean. Watch
    `train_pool_effective_examples` to see whether it moved.
- `[sender] prototype_dropout` / `[sender] vision_dropout`: dropout on the pooled
    concept vectors (between prototyper and language model) and on the per-image
    embeddings (between vision model and prototyper) respectively. The former is
    the stronger regulariser and is where jayelm's single speaker-side
    `--dropout` sits; dropping pre-pool features is largely undone by the
    average over n/2 examples.
- `[*] alpha` / `[*] beta`: DeepNorm's residual scaling on every post-norm
    transformer stack — `alpha` on the skip, `beta` on the branch. `"deepnorm"`
    (the default) resolves them at construction from that stack's own `layers`,
    as `(2N)^(1/4)` and `(8N)^(-1/4)`; a number pins them instead, and `1.0` is
    the no-scaling identity every run before this used. The cross-attention
    listener's two slots each resolve their own, from their own table's `layers`
    — and in the decoder form, `(3N)^(1/4)` and `(12N)^(-1/4)`, since those
    blocks carry a cross-attention branch as well.
    Derived rather than configured because two constants restated
    per config is an invitation to leave them at values belonging to a depth the
    stack no longer has.
- `[receiver] language_model` / `[receiver] discriminator`: the listener's two
    slots, and they are chosen independently. `ReceiverGRULM` reads the message
    with a GRU; `ReceiverCrossAttentionLM` reads it with a decoder stack that
    cross-attends into the candidate set, so what it encodes is discriminative
    rather than absolute. `BilinearDiscriminator` scores each candidate by
    `obj·W·m`; `AttentionDiscriminator` scores them with a second decoder stack
    reading the encoded message, interpolated with a bilinear score over that
    same encoding.
    One key used to choose both halves at once, so a rung swapping the GRU
    comparer for the cross-attention one changed the encoder *and* the
    comparison and "attention helps" could not be attributed to either.
    Exactly one message encoder is built whatever the pairing:
    `AttentionDiscriminator`'s bilinear path is a second *comparison*, reading
    whatever the language model produced.
- `[receiver_discriminator] mix_floor` / `mix_logit_init`: the attention
    path's minimum share of the score, and where that share opens.
    `AttentionDiscriminator` returns
    `s · [(1 − a)·bilinear + a·attention] + bias` with both paths standardised
    per game, and `a = mix_floor + (1 − mix_floor)·sigmoid(mix_logit)`.
    The defaults, 0.1 and −4.0, open it at 0.116 — essentially *as* the bilinear
    comparison, which is the configuration measured bootstrapping where the
    attention stacks alone do not. The floor exists so the attention path always
    receives gradient, and it is in the parameterisation and never a `clamp`,
    whose gradient is zero below its bound. Watch `train_mix_alpha` and
    `train_path_agreement` together.
- `[receiver] dropout`: the listener's **only** dropout, and the
    counterpart of the sender's `prototype_dropout`. `Receiver` masks the
    incoming referent embeddings once and hands the same masked tensor to both
    slots, so no pairing can regularise twice; it defaults to `0.1` to match the
    sender. It masks *elements* of `(batch, n_objects, features)`, so it removes
    features within a candidate and never a whole candidate, which would leak
    the label ordering. It used to mask
    the message operand as well; the message arrives through the Gumbel channel,
    whose noise `sampling_tau` and `uniform_weight` already calibrate, so a mask
    on top is a second and uncalibrated perturbation of a signal that has one.
    The referents arrive clean, which is what makes that the side where a mask
    regularises rather than compounds. There is deliberately no
    `[receiver] vision_dropout`: it would mask the same tensor as this knob,
    with nothing but a reshape between them, so the two would compose into one
    mask at a rate neither names. The sender keeps its `vision_dropout` because
    the prototyper pools between the two masks there. It affects **only** that
    one input: module internals are fixed constants, so raising it never
    silently rewires the architecture. `ReceiverGRULM`'s inter-layer dropout is
    pinned to `0.0` (jayelm's listener GRU takes no dropout argument), and the
    attention slots' attention dropout is `cross_attention_dropout` in each of
    `[receiver_language_model]` and `[receiver_discriminator]`, both `0.0`. For
    reference, jayelm regularised both agents with a single `--dropout` at
    `0.1`, on the listener's vision pathway only, with nothing at all on its
    language pathway.
There are exactly **two** exploration controls, and they do different jobs. The
emittable logits are always layer-normalised to unit variance per example and
per position before sampling — there is no knob for that, because both controls
below are expressed against it. (LayerNorm replaced a `batch_norm_logits` here.
It is ~2.6× tighter on the criterion that matters, is
position-invariant for both speakers, has no running statistics so train and
eval agree, and does not couple to `accumulator_steps`. Without an affine it is
argmax-preserving, so it changes no eval-time message.)

- `[sender_language_model] init_energy`: the **starting point**, and the range
    the speaker can move through. `0.9` everywhere. It is the fraction of maximum
    entropy a *freshly initialised* speaker's per-position distribution retains —
    `H(p) / log2(V)`, so `1.0` is a speaker that emits uniformly at random and
    `0.0` one that emits a single token with certainty. A fraction, not a
    percentage.

    It is not the scale itself. `logit_scale` in `models/sender.py` solves once
    at construction, by bisection against a fixed sample, for the multiplier that
    delivers this entropy at a given `vocabulary` and `uniform_weight`: `0.802` at
    ShapeWorld's `V = 14`, `0.839` at CUB's `V = 20`. Asking for entropy rather
    than a scale is what makes the two datasets comparable — an earlier
    `c * ln(V)` form over-corrected for vocabulary by about four times.

    Why entropy rather than a symbol error rate or a channel capacity: at
    initialisation there is no correct symbol to have an error rate *against*
    (argmax is an accident of the init, not an intended message), and capacity
    runs backwards, since a high-capacity channel is a sharp one with less room
    to explore. The reason to start high is bootstrapping — a fresh speaker's
    argmax barely varies with its input, so a low-entropy start means it emits
    near enough one message for everything, confidently, from the first batch,
    and the listener co-adapts to that before the speaker's embeddings are worth
    grounding on.

    **This is a starting point, not a target**: what a speaker actually does is
    reported as `realised_survival` and `logit_scale`, and is expected to move
    over a run — including *downwards* in fidelity early on, which is the
    speaker annealing itself rather than a fault, and the birds baseline makes
    that descent every run: 0.81x of its opening by epoch 5, recovered by epoch
    8, and it is bootstrapping from a realised survival of 0.12-0.14 while it is
    down there. Flooring the scale at this value was tried and reverted — it
    cost that arm fifteen epochs against a same-seed control. `logit_scale`'s
    docstring carries the derivation and the reference points to rederive `0.9`
    from.
- `[sender_language_model] uniform_weight`: the **ceiling** on fidelity, and so
    the floor under exploration. Weight of the uniform component mixed into the
    policy before sampling, train-pass only. It caps a slot's winner at
    `1 - w + w/V` and floors its losers at `w/V`, so at least `w * (1 - 1/V)` of
    symbols are flipped however sharp the logits get — 0.0186 at the default
    `0.02` and `V = 14`. `V` is `vocabulary`, counting the emittable tokens only:
    the four reserved slots are masked before the mixture, which is spread over
    what is left, so these divide by 14 rather than by 18. That floor is a
    permanent per-symbol corruption rate that training cannot reduce, which is
    the point: it keeps late training from committing the channel entirely. This
    is jayelm's `--uniform_weight` doing the job it does there.
- `[sender_language_model] tau`: gumbel-softmax tau (jayelm's `--tau`). This
    shapes the straight-through *gradient* only: the hard forward sample is an
    argmax, so it is invariant to tau. Raising it does not buy exploration, it
    just flattens the surrogate gradient. `1.0` everywhere, GRU and Transformer
    alike, so the two speakers differ only in architecture. Because the scale is
    now a constant, the ratio between the two is fixed for a whole run and the
    estimator sits at one operating point throughout — that was worth arranging
    deliberately when the scale moved batch to batch.
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
- `train_realised_survival` — the mean winning-token probability, i.e. the
    fraction of symbols surviving the noise. Train pass only, since that is the
    only pass that samples. This is the channel diagnostic. It is a finding, not
    a target — expect it to start near 0.5 and climb as the speaker sharpens,
    bounded above by the `uniform_weight` ceiling of `1 - w + w/V`. Flat at ~0.5
    for many epochs means the channel is not opening; pinned at the ceiling
    early means it has nothing left to explore with. Since `87c1027` the
    speaker owns its sharpness, so this is no longer a pure readout of logit
    *shape* — read it with `train_logit_scale`, which separates the two.
- `train_logit_scale` — the multiplier applied to the normalised logits before
    sampling, `exp(log_logit_scale)`. This is the channel's fidelity: the Gumbel
    noise floor is a fixed sd of 1.283 and `layer_norm_logits` pins the logits
    to unit variance, so the scale alone says how much of the speaker's
    distribution survives. It opens where `init_energy` puts it (0.802 for
    ShapeWorld, 0.839 for birds) and a usable channel is somewhere around 4 to
    6. It is the disambiguator for a falling survival: a flatter policy at a
    steady scale is the speaker's own doing, a falling scale is the channel
    closing. Both happen, and the distinction that matters is depth rather than
    direction: a healthy arm dips about 0.2 log-units below its opening and
    climbs back through it within a few epochs, while the preliminary ViT-sender
    arm fell 0.94 log-units over a hundred and never returned.
    Its travel is bounded by `lr * steps`, because AdamW normalises by the
    gradient's second moment and this is a lone scalar — so it moves at about
    `logit_scale_lr` per step whatever the gradient's size, and comparing its
    observed climb to that ceiling reads off how sign-consistent the gradient
    is. Expect a flat start (no gradient reaches it while the listener cannot
    use the message), then a climb, then a plateau once the `uniform_weight`
    cap saturates fidelity and the gradient dies with it. A scale climbing
    while accuracy stays at chance is co-adaptation to a premature code.
- `train_score_scale` — the listener's counterpart, `exp(log_score_scale)`, and
    live on every rung. Both comparers reach it by the same route with different
    parts: on `BilinearGRUComparer` both operands of the dot product are
    layer-normalised without an affine and the product is divided by
    `sqrt(referent_embedding_size)`; on `TransformerCrossAttentionComparer` the
    readout direction is normalised to unit length and its input is
    layer-normalised without an affine. Either way the architecture is left
    setting the *direction* of the comparison and this scalar setting its
    volume, and it is the only thing that can. It cannot move the decision:
    `scores > 0` and the reference-game argmax are both invariant to it. What it
    moves is BCE, and through BCE every gradient in the pair.
    It was NaN on the cross-attention rungs until the readout was separated
    that way, which is how rung 12 ran a whole 30-epoch smoke test with its
    scores collapsed 25x — from sd 0.42 to sd 0.016 inside the first epoch —
    while every column that would have said so was either blank or, in
    `train_loss`, parked at `ln 2` to four decimal places.
    It opens at exactly 1.0, which the `sqrt(d)` division makes the calibrated
    value rather than an arbitrary one, so unlike `train_logit_scale` it has no
    traverse to cover and a flat start is not expected. Read it the same way
    otherwise, and against the same separation: a dip while the message is still
    noise is the listener correctly refusing to commit, and `29b18ea`'s numbers
    are the yardstick for when a dip has become a collapse. Its travel is bounded
    by `score_scale_lr * steps` for the same AdamW reason as the speaker's — at
    `2e-3` and 194 steps that is 0.388 log-units an epoch, against an observed
    healthy dip of about 0.2 over five. There is no floor, deliberately;
    `e3fcabd` fitted one to the speaker's scale and `29b18ea` removed it after it
    cost fifteen epochs.
- `train_sampling_tau` — the temperature actually handed to `gumbel_softmax`,
    as against the configured `tau`. A function of `train_logit_scale` and the
    epoch counter alone, so it carries no independent information, but it is
    what sets how much straight-through bias the run is paying. It equals `tau`
    at initialisation and whenever the scale sits below its opening value,
    rises with the scale so that losing tokens keep receiving gradient, and
    returns to `tau` by the last epoch as the coupling retires.
- `train_logit_spread` — the standard deviation of the emittable logits
    *before* normalisation, so it reports the size of the logit *shape* rather
    than of the channel: `layer_norm_logits` divides this magnitude back out,
    and since `87c1027` the channel's sharpness lives in `train_logit_scale`
    instead. What it is still good for is the normaliser's floor. Below a spread
    of ~1e-6 LayerNorm can no longer rescue it (see `LAYER_NORM_EPS`); anywhere
    above that the spread is absorbed and only shape reaches the channel.
- `train_pool_effective_examples` — `1 / sum(p^2)` over the prototyper's
    attention weights, in examples: how many of a concept's images the prototype
    is actually built from. It opens at the number of positive examples, where
    the pooling is uniform and the prototype is exactly the mean, and falls
    towards 1 as the pooler commits to particular images. `AveragePrototyper`
    reports the example count by construction, so the two arms of the ladder
    stay readable side by side.
- `train_pool_score_norm` — the norm of the attention prototyper's scoring
    vector, which is what carries that departure. It opens at exactly zero, so
    this and the column above are what separate "the pooler learned something"
    from "the pooler stayed at the average" — otherwise indistinguishable
    outside a checkpoint. NaN for `AveragePrototyper`, which has no scoring
    vector, rather than a zero that would read as a pooler which had not moved.
- `train_polarity_separation` — the distance between the two rows of the
    Transformer speaker's `polarity_embedding`, the learned tag that marks which
    of its prototypes is the positive concept. Only the *difference* between the
    rows can do anything: a constant added to both shifts every key and value
    alike and cannot separate them. It opens at `2 * sqrt(d_model)` — 35.8 on a
    320-wide speaker — the tag being an antipodal draw at the scale of the
    layer-normed prototype it is added to, where under the old zero init it
    opened at 0 and the question was whether the tag was ever used at all. Read
    it against rung 10, the one rung that both builds this speaker and learns:
    from a zero init that column went 0.098 → 13.19 over thirty epochs, with
    `train_acc` leaving chance in the same epoch the tag crossed 1.0. A learning
    run wants a tag of order 10, so the new opening is a factor of 2.7 high, not
    the order of magnitude that rungs 11 and 12 would suggest — those never
    learned, so their 0.16 to 0.79 is where a *dead* run leaves it.
    NaN for `SenderGRULM`, which is handed the distinction by `init_h` — it
    reads `torch.cat(prototypes, 1)`, so each polarity gets its own weight
    columns — and so has nothing to learn and nothing to report. Without the tag
    the Transformer speaker cannot make the distinction at all: its
    cross-attention carries no positional or type encoding on the key side, so
    its output is bit-identical under swapping the two prototypes.
- `{train,test,test_same,test_avg}_unique_message_fraction` — distinct messages
    over messages emitted. A language that is doing work compresses: healthy
    runs sit around 0.30–0.40, while runs whose channel is too noisy to learn
    through sit at 0.88–1.00, emitting near-noise.

`test_avg` is the unweighted mean of `test` and `test_same` for the same metric,
applied to every eval metric including topsim — so it is a mean of two Spearman
rhos, not a rho over the pooled data. Both datasets ship a `test_same` split, so
`test_avg` is a genuine mean in every current run; the split stays optional in
principle, and where one is absent those columns are missing entirely and
`test_avg` equals `test`. Note that birds runs recorded before `cub.py` grew a
`test_same` split have `test_avg_*` columns that are copies of `test_*`, and are
not poolable with newer ones.

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
`best_lang.csv`. What gets written is a `final_model.pt` / `final_lang.csv` at
the end, alongside the rolling `checkpoint_last.pt` that exists only so an
interrupted run can resume. There are no periodic `<epoch>_model.pt` /
`<epoch>_lang.csv` snapshots: they were never read, and on a cluster they cost
far more disk than the mid-training language was worth. The consequence to be
aware of is that the *evolution* of the language is not recoverable after the
fact — `metrics.csv` records topsim and accuracy per epoch, but the messages
themselves survive only for the final epoch.

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

# Training

`code/train.py` is the entry point: `python train.py --config <toml> [--seed N]
[--no_resume]`. SLURM array indices map to seeds for repeats.

## Config resolution (`code/parse_config.py`)

`get_config` combines a repo-root `DEFAULT.toml` with the experiment's TOML:

1. Take the plain defaults and the dataset-specific defaults out of
   `DEFAULT.toml` (the `[shapeworld]` and `[birds]` tables are the latter).
2. Build a *provisional* config by overwriting the generic defaults with the
   custom TOML. This is done first so the dataset can be read from it — reading
   the custom TOML directly would fail if it does not specify one.
3. Overwrite the generic defaults with the dataset-specific ones.
4. Overwrite the result with the custom TOML again, giving the final config.

**Which family of defaults applies is decided by the dataset's *name*, not its
location**, because `train.py` later rewrites the path to fast storage.
ShapeWorld has several variants (`shapeworld_40`, `shapeworld_ref`, …) which all
share the same defaults.

The result is wrapped in a `SafeDict`, which warns and returns `None` on a
missing key rather than raising.

### Validation

`validate_config` rejects contradictory configs up front:

- `use_lang` must be false if `copy_receiver` or `receiver_only` is true.
- `copy_receiver` and `receiver_only` are mutually exclusive.
- `reference_game_xent` requires `reference_game`.
- `joint_training` must be false — there is no joint-training objective in this
  codebase. This used to be checked once per batch inside the training loop;
  rejecting the config up front fails in the same cases, but before any work is
  done.
- Speaker and receiver `message_length` must agree.
- `sender_language_model.init_energy` must be present and in (0, 1] — a *fraction
  of maximum entropy, not a percentage*. Checked here rather than left to the
  speaker's constructor, because `SafeDict` only warns on a missing key and hands
  back `None`, which would fail confusingly deep inside the decode instead of at
  parse time.
- `silhouette_p_sender` / `silhouette_p_receiver` in [0, 1].
- `data.dataset` present.

`builder.build_models` additionally asserts `receiver_only`, `copy_receiver`,
`share_language_model` and `share_feat_model` are all off, and that the features
are not 1-dimensional. None of that applies in these experiments, and the asserts
say so.

## Storage paths (`code/paths.py`)

A single JSON file at the repo root declares `data_slow_storage`,
`data_fast_storage` and `output_root`. SLURM jobs read `data_fast_storage`
(staged data) and `output_root` (where results are written, arranged by
experiment).

`train.py` resolves dataset logical names (`cub`, `shapeworld`,
`shapeworld_ref`) to their location on fast storage. **This must happen after
`get_config()`**, which branches on the original path string to pick the birds
defaults.

Results land at `<output_root>/<experiment>/<config_stem>_seed<seed>/`, where
`<experiment>` is the `experiments/<exp>/configs/<file>.toml` parent directory.

## Optimiser groups (`models/builder.py`)

`gradboard.get_optimiser` builds the parameter groups, keyed on
`(lr, weight_decay)`. `split_out_parameter` then moves every parameter whose name
ends in a given suffix into a group of its own at a given learning rate.

**Why after the fact rather than asking `get_optimiser` for it:** that function
keys its groups on `(lr, weight_decay)` and takes a single `lr`. The parameters
this is used for currently share a group with every other undecayed parameter —
`log_logit_scale` because it is 0-dimensional, and `polarity_embedding` because
`gradboard`'s `EXCLUDE_FROM_WEIGHT_DECAY` matches "embedding" — so both fall to
the `weight_decay = 0.0` branch, and retagging that group would drag the biases
and norms along with it.

**It must run before `PASS` is constructed.** The scheduler deep-copies the
groups once at construction and thereafter scales each group from its *own*
recorded base lr, so the override rides the schedule shape correctly and is not
flattened by it — but a group added afterwards would not appear in
`original_param_groups` and would break the `strict=True` zip.

New groups are appended, so group 0 remains the main one that `PASS.lr` reports.
Calling it more than once is fine for the same reason.

The new group takes `weight_decay = 0.0` to match what `get_optimiser` gave both
of these, and for the same reason in each case. The scale is a log, so decay
would pull `exp` towards 1, and a scale of 1 is not a meaningful anchor —
`init_energy` solves to 0.839 for birds and 0.802 for ShapeWorld, so landing near
1 would be an accident of vocabulary. The polarity tag opens at the scale of the
layer-normed prototype it is added to, so decay would be a force on it that
answers to neither the loss nor that scale.

If the suffix matches nothing, `split_out_parameter` raises rather than silently
doing nothing — the error names the config key so a rename says which knob went
quiet.

### The three overrides, and why each is gated the way it is

**`logit_scale_lr`** — ungated. `log_logit_scale` exists on both speakers.

**`polarity_embedding_lr`** — gated on `isinstance(language_model,
SenderTransformerLM)`. Deliberately *not* tied to `logit_scale_lr`, though
`DEFAULT.toml` opens them at the same value: `log_logit_scale` exists on *both*
speakers, so raising it to help the polarity tag would also retune the GRU
baseline's channel and shift the comparison the ablation is there to make. Two
keys, one number, and either can move alone.

Gated on the speaker class rather than on finding the parameter, because the two
failures need different answers. A GRU speaker has no polarity tag by
construction — it reads `torch.cat(prototypes, 1)` and is told which is which —
so the key is simply inapplicable, exactly as `heads` and `ff_ratio` are, and
skipping is right. A Transformer speaker *missing* the parameter is a rename, and
`split_out_parameter` raises.

**`score_scale_lr`** — gated on `isinstance(comparer, BilinearGRUComparer)`, for
the same reason. `TransformerCrossAttentionComparer` has no learnable scale: its
readout is a plain `nn.Linear(d_model, 1)` whose weight carries the volume, so
there is no lone scalar for a rate to apply to.

Do not read that gate as a verdict on the scale. That comparer lost its scale to
a rewrite that standardised the readout at a fixed gain, which closed the
collapse it was aimed at and stopped four rungs learning at all (see
[anecdotes.md](anecdotes.md)). What survived the revert was the absence of the
parameter, not an argument against it. The knob is gated rather than deleted
because `BilinearGRUComparer` keeps its scale and is the ablation's baseline
listener — and gating on the class that *has* the parameter leaves
`split_out_parameter`'s error on duty where it can still fire.

## Gradient clipping

Clipping is **per submodule**, not one norm across the whole pair.

`torch.nn.utils.clip_grad_norm_` scales every gradient by the single factor
`max_norm / total_norm`, so a global call hands every module a coefficient set by
whichever module dominates the norm — and the listener's comparer dominates it
heavily (~90% of the squared norm at init, against ~0% for the speaker's vision
model, whose gradient reaches it through the whole language model and the
straight-through Gumbel sample). The coefficient then carries the comparer's
batch-to-batch fluctuation into every other module, which is noise added to
gradients that are already the weakest in the pair.

Clipping per module makes each coefficient depend only on that module's own
gradient. It does not change the *ratio* between modules — a uniform rescale
never did, and AdamW normalises per coordinate anyway — what it removes is the
cross-module noise coupling.

`CLIP_GROUPS` partitions the pair into six named modules; an `other` group
catches anything a future architecture adds, so no parameter can silently go
unclipped. Every group's pre-clip norm is returned for logging, including ones
that did not reach the ceiling.

## AMP and accumulation

The forward pass runs under `autocast` in bfloat16 where supported and float16
otherwise. Gradients are scaled, accumulated over `accumulator_steps` batches,
then **unscaled, clipped, stepped** — that order per the PyTorch AMP gradient
clipping recipe. A trailing partial accumulation is flushed after the loop.

The scheduler is `gradboard.scheduler.PASS`, driven by
`gradboard.cycles.CycleSequence` over up to two stages: an `ascent` warm-up of
`warm_up_epochs`, then `lr_schedule_shape` for the remainder.

`torch.compile` is applied to the pair when `config['compile']` is set. Note that
the `.item()` calls in the diagnostic paths cost a sync and a graph break under
it; that cost is paid deliberately (see [measurement.md](measurement.md)).

## Receiver resets

`receiver_reset_interval > 0` re-initialises the whole receiver every N epochs,
including at epoch 0. Every `reset_parameters` in the codebase is written to
reproduce construction *exactly*, including buffers: BatchNorm running statistics
are reset, because leaving them would carry the pre-reset feature distribution
across the reset, which is not what the interval is asking for.

## `metrics.csv` discipline

Rows are **appended** per epoch rather than rewritten from an in-memory list,
which is what lets earlier rows survive a resume. The metrics dict itself is
rebuilt from scratch each epoch, so nothing in it needs to survive.

`to_csv(mode="a")` writes values in this row's key order under whatever header is
already on disk, and checks nothing against it. That makes two failure modes
possible, and both are refused rather than appended to:

1. **Different splits.** A run started before this run's splits existed — a birds
   run from before `cub.py` built `test_same`, say — has no `test_same_*`
   columns, so resuming would write wider rows under a narrower header. Checked
   once at startup.
2. **Different metrics.** Any new column does the same damage for the same
   reason: `pool_effective_examples` arriving mid-flight silently misaligns every
   column from that row on. Checked before every append, comparing the full
   header to the row's key order.

The failure is invisible until someone reads the CSV months later, so both raise
with instructions to use `--no_resume` or move the run directory aside.

A fresh start (`--no_resume`, or no checkpoint to resume from) deletes any stale
`metrics.csv` first.

## Checkpoints

The checkpoint holds the epoch to resume *at* (`epoch + 1`) and
`scheduler.state_dict()` — gradboard's `PASS` restores model, optimiser, scaler
and step count together.

It is written **after** the metrics row is on disk, so the model state stays in
sync with `metrics.csv` on resume. As in vit, a crash between these two writes
can re-emit one row, which is benign.

At the end of a run — a **fixed endpoint**, since there is no best-epoch
selection — the deliverables are `final_model.pt`, `final_lang.csv`, and the
per-epoch trajectory already in `metrics.csv`. Skipped when resuming a run that
is already done.

## Reproducibility metadata (`code/util.py`)

`save_args` writes both `args.json` and `config.toml` (the former for the
original tooling, the latter for the new), stamped with the git hash and with
`dependency_versions`.

`current_git_hash` warns if the working tree has unstaged changes, and covers
*this repository only* — which is not enough. The model is mostly broccoli's, and
broccoli's version has moved underneath this repository before without any commit
here to show for it. `PINNED_DEPENDENCIES` is therefore recorded alongside every
run: `broccoli-ml`, `gradboard`, `torch`, `torchvision`.

For anything installed from git, pip records the resolved commit in the
distribution's `direct_url.json`, so the exact source is recoverable even if the
requirement was written as a branch or a tag. `read_text` returns `None` for a
missing file but *raises* for a distribution installed in a layout without a
metadata directory, hence the `OSError` guard. Values are always strings, since
this ends up in `config.toml` and TOML cannot represent `None`.

## Visualisation

`code/vis.py` renders a Jinja2 report of one batch's games — the speaker's view,
each listener's view, the emitted message, the true logical form, and per-object
correct/incorrect marks — to `<exp_dir>/html/<epoch>_<split>.html`, with images
written alongside. Enabled by `config['vis']`. `dataset.vis_input` both returns
the HTML fragment and saves the image as a side effect, and is told to overwrite
only on the first language type so the same image is not rewritten once per
variant.

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
`(lr, weight_decay)`. Two functions then move parameters out of them.

`split_out_parameter` takes a **name suffix** and moves every parameter of the
pair whose name ends in it. That is the right selector for the lone scalars of
`SPLIT_LEARNING_RATES`, each of which is one distinctively named tensor.

`split_out_module` takes a **module object** and moves every non-exempt
parameter in it. "Every tensor in `sender.language_model`" is not a suffix, and
a prefix would be no better — the widths that decide these rates live on the
constructed module, not in its name. Selecting on the object means there is no
name-matching ambiguity at all, and nothing a rename can quietly break. It is a
sibling rather than an argument to the suffix version because the two differ in
how they *fail*: an unmatched suffix is a rename and raises, whereas a module
whose parameters are all exempt is a legitimate answer and returns unchanged.

Both share `_regroup`, which is the part that actually moves anything: filter the
selected ids out of every existing group, then `add_param_group`.

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
quiet. `split_out_module` has no equivalent, because it cannot fail that way.

### The muP rule

Every module with a width takes

```
lr(module) = [optimiser] lr × [optimiser] mup_reference_width / d_model(module)
```

with the reference width at **1024**: jayelm's `--speaker_hidden_size` and
`--listener_hidden_size`, and therefore the width at which `lr = 1e-4` was tuned.
Under muP, Adam's stable rate for a matrix mapping one width to another goes as
1/fan_in, so a module three times narrower than the one the rate was tuned on is
being trained three times too slowly. `resolve_mup_learning_rates` applies it and
`split_out_module` does the regrouping.

The rule is stated once rather than as six literal rates, because the rates are a
consequence of widths that move. `mup_width` reads `d_model` off the
**constructed module** rather than out of the config: several of these widths are
derived — `SenderTransformerLM` and `AttentionPrototyper` take theirs from the
vision model's `final_feat_dim` — so a config key would be a second statement of
the same number, able to disagree with it.

**One rate per module, keyed on `d_model`.** Exact for the attention projections,
which are square in it and are most of the parameters; approximate for the
feedforward inner layers and for adapters that read a foreign width. This is a
heuristic applied at module granularity, and saying so is cheaper than a
per-tensor scheme nobody can check by eye.

**Two exemptions, both by rule rather than by list** (`is_mup_exempt`), so a
module added later is covered without anyone having to remember a list.

- `p.dim() < 2` — biases, norm gains, and every learned scalar. No fan-in to
  scale by.
- name containing `"embedding"` or `"query"` — `polarity_embedding`,
  `label_embedding`, `token_embedding` and `query`. muP gives input
  embeddings a Θ(1) rate because their fan-in is a one-hot index rather than a
  width, and every one of these is Θ(1)-*initialised* as well, so scaling their
  rate by width would be scaling against an init that never shrank. Matched as a
  substring exactly as `gradboard`'s `EXCLUDE_FROM_WEIGHT_DECAY` matches
  `"embedding"`, and with the same caveat: a rename would move a tensor silently.

The dimension exemption is load-bearing twice over. It is also what makes the muP
groups **disjoint from `SPLIT_LEARNING_RATES` by construction** — every scalar in
that table is 0-d, and `polarity_embedding` is 2-d but matches `"embedding"`. No
ordering trick is needed. muP runs first anyway, so that if the exemption is ever
loosened the elevated scalar rates are the ones that survive.

**Whole modules out of scope**, all by the same test — no `d_model` attribute.
The convolutional backbones (`ResNet18.final_feat_dim` is a hardcoded 512, muP's
rules are stated for transformers, and a ResNet at 1e-4 is what jayelm tuned);
`AveragePrototyper` and `nn.Embedding`, which have no matrices between widths;
and `BilinearDiscriminator`, whose single tensor's fan-in is the language model's
`output_size` — 1024 on the restored listener GRU, i.e. the reference width, so
its factor would be 1.0.

**What the rule does not claim.** muP transfers a tuned learning rate across a
change of *width* in one architecture. Two of the changes here are not that: the
speaker's language model goes GRU → Transformer as well as 1024 → 320, and `ViT2`
replaces a ResNet with no 1024 anywhere in it. Principled heuristic, not transfer
guarantee.

Only the learning-rate half of muP is adopted. Broccoli's `nn.Linear` init is
already 1/√fan_in, `msa_scaling = "d"` is already muP's attention scaling, and
`layer_norm_logits` already makes the speaker's readout scale width-invariant.
The `mup` package in `requirements.txt` stays unimported.

**Do not read the `CLIP_GROUPS` norms as evidence about these rates.** Under Adam
a uniform rescale of a module's gradients cancels in `m̂/√v̂`, so gradient
magnitude says nothing about whether a learning rate suits the module.

**Do not reach for the LR range test either.** `PASS._apply_range_test_result`
calls `set_all_lr` and then overwrites `original_param_groups`, which would
flatten every split rate — muP's and the scalars' alike — permanently.

### The overrides, and why each is gated the way it is

Note this subsection still names `BilinearGRUComparer` and
`TransformerCrossAttentionComparer` below, which the listener split replaced with
`BilinearDiscriminator` and `AttentionDiscriminator`, and it predates
`mix_logit_lr` and `mix_scale_lr`. The reasoning holds; the names want a pass
with the rung overhaul.

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

**`contrast_gate_lr`** — gated on `pair.sender.contrast is not None`, i.e. on the
stage existing at all, since `[sender] contrast` is a boolean and "off" is the
absence of the module rather than a different one.

This is the override the arm depends on rather than merely benefits from.
`contrast_gate` opens at exactly zero — that is what makes the contrast arm an
ablation of one thing — and a lone scalar cannot travel further than `lr × steps`.
At the base 1e-4 and birds' 62 steps an epoch it would take sixteen epochs of
perfectly sign-consistent gradient to reach 0.1, so a run at the base rate would
report "contrast does nothing" and be measuring the learning rate. At 2e-3 it is
fifty steps. See [architecture.md](architecture.md) for why the gate is a scalar
rather than a zero-initialised projection, which has the same problem and no way
out of it.

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

**`cool_point_multiplier = 1.0` makes `lr_schedule_shape` a complete no-op**, and
the configs in the August 2026 ablation all set it there. `PASS.update_learning_
rates` computes

```python
min_lr = base_lr * self.cool_point_multiplier
current_lr = min_lr + (base_lr - min_lr) * self._schedule_multiplier
```

so at a multiplier of 1 the floor equals the ceiling and every group sits at its
own `base_lr` for the whole run, whatever shape is named in the config. The knob
reads as live and is not. Check it before attributing anything — a flat opening,
a late takeoff — to the schedule.

**One optimiser step per `accumulator_steps` batches, and the scheduler steps
with it.** `scheduler.step` is called from inside `optimiser_step`, so the
schedule length and the traverse budgets below are both counted in optimiser
steps, not batches.

### Effective batch size is a cost when takeoff is gated on a scalar

Adam's update is bounded at roughly ±lr per step regardless of gradient
magnitude, so a lone scalar cannot travel further than `lr × steps` (the ceiling
quoted for `logit_scale` in [measurement.md](measurement.md), and the reason
`contrast_gate_lr` exists at all). Averaging `accumulator_steps` microbatches
into one update buys a better gradient *estimate*, which a single scalar does not
need, and costs it the moves it would otherwise have made.

So when a run's takeoff waits on one of these scalars — `log_logit_scale`,
`contrast_gate`, `log_mix_scale` — raising the effective batch delays it in
direct proportion, at identical compute. That is a real trade against whatever
the larger batch was for, and on ShapeWorld the reference setup's batch of 128
(32 × `accumulator_steps` 4) is four times the traverse cost of the same compute
at an accumulator of 1.

## Reading a run: deadlock, or undertrained?

Both look like a flat loss near `ln 2`. They want opposite responses, and the
diagnostic columns separate them cheaply.

**The deadlock signature is joint and it is exact.** Every learned quantity on the
speaker side stationary to four decimal places across tens of epochs:
`realised_survival` flat near chance (`1 / vocabulary`), `logit_scale` at its
`init_energy` solve, `contrast_gate` unmoved, `pool_effective_examples` pinned at
the positive-example count, `polarity_separation` at its `2·sqrt(d_model)` init.
The channel is carrying noise, the listener has nothing to learn from, and the
speaker has no gradient telling it to sharpen.

`score_scale` sliding downwards through this is a *symptom*, not the fault — it is
the listener correctly declining to be confident about noise, and it is not by
itself fatal (see [anecdotes.md](anecdotes.md), where a run does the collapse and
learns anyway).

**Then read the direction of `logit_scale` to tell the two apart.** Rising slowly
means undertrained and more epochs help. Falling monotonically means the run is
drifting *away* from escape and more epochs will not reach it. In the August 2026
ablation the dead ShapeWorld rungs fell from 0.8700 to 0.8668 and from 0.8812 to
0.8805 over thirty epochs apiece, while the rungs that escaped did so as a sharp
transition — `realised_survival` 0.203 → 0.260 at epoch 5 → 0.697 at epoch 10.
Escape is visible as an event, so its absence is informative.

**Depth is the safe axis, including on a marginal bootstrap.** The instinct that
a deeper stack is riskier does not survive contact with what these stacks
actually do at init, and three mechanisms are behind that. See
[broccoli.md](broccoli.md) for the DeepNorm constants themselves.

1. **DeepNorm's `beta` shrinks with depth**, so a deeper stack opens *closer* to
   the identity, not further from it. On the decoder form `beta = (12N)^(-1/4)`,
   so going from 3 blocks to 6 takes the branch multiplier down by a further
   factor of `2^(1/4)`. Each block contributes less of itself at step zero, and
   the stack as a whole opens nearer a pass-through than the shallower one did.
   `alpha = (3N)^(1/4)` grows over the same range, which is what bounds the
   update rather than the signal.

2. **The opening is a floor, not a ceiling.** broccoli applies `beta` as a
   forward multiplier rather than an initialisation scaling, and the post-norm
   `RMSNorm` after it carries a learnable gain, so a branch that earns its way
   out of the opening ratio can take it. Nothing is lost to the scaling; it is
   only deferred.

3. **Stochastic depth does not accumulate with depth.** The rate is a linear
   ramp across the blocks — `step_size = stochastic_depth / (n_layers - 1)` —
   so the *mean* drop rate is about half the configured value whatever the block
   count. At `stochastic_depth = 0.1` that is ~0.05 at three blocks and ~0.05 at
   six. And a dropped block reverts to a clean identity rather than a rescaled
   one, since the mask is applied as `alphas = (alpha - 1) * mask + 1`.

So adding blocks costs compute and parameters, and very little else. When a
bootstrap is marginal, the thing not to spend budget on is anything that changes
the *channel* — the vocabulary, the message length, the opening `logit_scale`, or
the effective batch that the scalar traverses on. Depth is not in that set.

Note this reverses an earlier version of this section, which claimed depth was
the axis most likely to reproduce a marginal bootstrap and cited stochastic depth
as adding noise on top of a noisy channel. The ramp above is normalised by the
block count, so that reasoning was wrong.

**As of 2026-08-25 `stochastic_depth` is 0.0 in all five stacks**, so the rates
quoted above are the historical ones. It was turned off while diagnosing rung
09's failure to ignite; the argument is written out at `[sender_language_model]`
in DEFAULT.toml. That argument is *not* the one retracted in the paragraph above
— it is not about the rate growing with depth, but about `_residual` redrawing
its mask on every `forward` while `decode_autoregressively` calls the decoder
once per symbol, so the speaker's concept-to-symbol map jitters within a single
message. At a ramped mean rate of ~0.05 it is a small effect and may well be
irrelevant; it is off so that it is not a variable, not because it has been
shown to matter.

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

**It runs after `build_models`, not before.** `build_models` resolves the muP
learning rates from widths that are derived rather than declared and writes them
back as `[optimiser] resolved_mup_lrs`, so saving first would record a config
that does not say what was built. Nothing is lost on the failure path — the
dataloaders are constructed before either, so a run could already fail after
`args.json` was written.

**Diff the flattened `args.json` when comparing runs, never the `config.toml`
text.** The two agree, but the toml carries per-dataset blocks of differing
length, so a line-by-line `diff` between a ShapeWorld run and a birds run pairs
the wrong sections and invents differences that are not there. This produced a
phantom "rungs 13 and 14 are differently sized" reading in August 2026 that the
checkpoints then disproved.

**Read parameter counts off `final_model.pt`, not off either file.** Several
config blocks are inert for a given class combination — `[sender_feature_model]`
under a ResNet backbone, `[sender_contrast]` when `contrast = false` — and
several widths are derived rather than declared: `SenderTransformerLM` takes its
`d_model` from the vision model, and the discriminator is sized from
`receiver_language_model.output_size` (see
[architecture.md](architecture.md)). A config key is not evidence that the
module was built that way.

`[optimiser] resolved_mup_lrs` is the one derived quantity that *is* recorded, and
deliberately: it is a per-module `name → lr` mapping computed from those same
derived widths, so it is a statement about the pair that was constructed rather
than about the pair the config asked for. Read it as the answer to "which module
was trained at what rate", and note that it describes each module's matrices —
its biases, norms, scalars and embedding tables stayed at the base rate, and the
scalars named in `SPLIT_LEARNING_RATES` then moved again.

**Measure against the pinned dependency, not the installed one.** The version
string in site-packages can match `dependency_versions` while the commit does
not. `git archive <sha> | tar -x` into a scratch directory and put it first on
`PYTHONPATH`; a reading taken against a broccoli that has moved underneath the
run is not a reading of that run.

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

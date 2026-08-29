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

`split_out_module` takes a **module object** and moves every parameter in it
that no other group has claimed. "Every tensor in `sender.language_model`" is not
a suffix, and a prefix would be no better. Selecting on the object means there is
no name-matching ambiguity at all, and nothing a rename can quietly break. It is
a sibling rather than an argument to the suffix version because the two differ in
how they *fail*: an unmatched suffix is a rename and raises, whereas a module
whose parameters are all claimed elsewhere or frozen is a legitimate answer and
returns unchanged.

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

### The group table

One table in `models/builder.py` decides both what is clipped together and what
is trained at what rate.

`MODULE_GROUPS` names eight modules, each picked off the constructed pair by
attribute:

`sender_vision`, `sender_prototyper`, `sender_contrast`,
`sender_language_model`, `receiver_vision`, `receiver_token_embedding`,
`receiver_language_model`, `receiver_discriminator`

`SCALAR_GROUPS` names the four scaling scalars — `log_logit_scale`,
`log_score_scale`, `mix_logit`, `contrast_gate` — each of which is a group of
one tensor. `claimed_separately` holds them out of their module's group on both
sides, so a scalar is never clipped twice, never inflates its module's norm, and
never inherits its module's rate.

**Why one table.** There used to be three lists and none was derivable from
another. `train.py` had its own `CLIP_GROUPS`, which omitted `sender.contrast`,
so on every rung with the stage on the `other` catch-all *was* the contrast
stage under a name that said nothing about it. `MUP_MODULES` was a second list,
which included the contrast stage and omitted the listener's embedding table.
`SPLIT_LEARNING_RATES` was the third. Adding a module to one and not the others
was a silent partial change. Now a module added to `MODULE_GROUPS` is clipped
and rateable at once, and `group_parameters` is the single walk both mechanisms
take.

**Why the scalars are alone.** `clip_grad_norm_` takes one norm across a group
and scales every member by one factor, so a scalar sharing a group with a
thousand matrices is renormalised by *their* norm. At recorded speaker norms of
~10 against `clip_grad_norm = 1.0` that is a tenfold attenuation, applied on
every step that binds, to a parameter whose whole travel is already bounded by
`lr × steps` — and these are the parameters ignition waits on.
`scripts/ignition_audit.py` finds `logit_scale`, `contrast_gate` and
`pool_score_norm` leaving the plateau in the same epoch, so what constrains them
constrains the run.

**Why `score_bias` and `polarity_embedding` are not.** An offset is not a scale
and a 2-d tag is not a scalar; both belong to the norm of the module producing
the output they modify. Both still take a rate of their own through
`SPLIT_LEARNING_RATES`, which is why the mapping from clip group to config key
is not 1:1 — a clip group and a learning rate are separate questions.

### Per-module learning rates

`[optimiser.module_lr]` gives one rate per module group, keyed by the group's
name. An absent key means the base `lr`; a key naming no group raises in
`parse_config`, which is the same guard `split_out_parameter` gives the scalars,
moved to parse time because a module group selects an attribute and so cannot
fail by rename. `resolve_module_learning_rates` reads the table and
`split_out_module` does the regrouping.

DEFAULT.toml states all eight and no rung overrides any of them, so this is what
every rung runs:

| group | rate |
|---|---|
| `sender_vision`, `sender_prototyper`, `sender_contrast`, `sender_language_model` | 2e-4 |
| `receiver_vision`, `receiver_token_embedding`, `receiver_language_model`, `receiver_discriminator` | 1e-4 |

One factor of two — the whole listener at half the whole speaker — pinned at
`1e-4` on `receiver_language_model`, the module jayelm tuned `lr` on. Nothing is
split *within* an agent. The vision backbones took half their agent's rate until
2026-08-29, on an ordering borrowed from the VLM fine-tuning literature; that
argument is about preserving a pretrained representation, and nothing here is
pretrained. See DEFAULT.toml's `[optimiser.module_lr]` block.

**These are literals now, and that is the change.** Until August 2026 they came
out of a rule: `lr × reference_width / d_model / layers`, with the reference
width at 1024 — jayelm's `--speaker_hidden_size`, and therefore the width
`lr = 1e-4` was tuned at. The width half was muP, whose claim is that Adam's
stable rate for a matrix mapping one width to another goes as 1/fan_in. The
depth half was not muP and had nothing behind it.

The rule was tried twice and neither half survived on its own terms. `89ab6fc`
reinstated the width half alone and rungs 9 and 10 both failed to converge —
which is evidence against the reading that motivated it, namely that those rungs
were starved of rate. `afcefd0` added the depth half to pull the other way, and
the four rates above are what it produced.

Two things are worth keeping from that. First, the exponent: the literature's
figure for a residual stack is 1/√L rather than 1/L, and broccoli's DeepNorm
already puts this architecture in that regime — its α = (2L)^(1/4) and
β = (8L)^(−1/4) give an effective branch scale of exactly `0.5·L^(−1/2)`, so
dividing the rate by L on top of that over-corrects by √L. Second, and more
usefully: what the rule bought was never a principled exponent, it was the
per-module *structure*. A width rule alone cannot express any of it — at 320 wide
it cannot tell a ten-block ViT from a one-layer scoring projection. Every rate
above that differs from another at the same width does so because of the half
with nothing behind it.

So the structure is kept and the derivation is not. A number in that table is a
number somebody chose, and it can be argued with on the evidence of a run rather
than defended as arithmetic. **Do not report any of this as muP.**

Note what the record here is, because no grid on this ladder is a neutral
baseline. Under the derived rates above — the two single-layer sender stages
highest in the pair and the ViT lowest by an order of magnitude — rung 10 took
off in epoch 0, having previously spent whole runs above `ln 2` at a flat 1e-4;
rung 9 carried the same four and did not ignite. The prototyper's elevation was
the deliberate part: `pool_score_norm` opens at exactly zero and its travel is
bounded by `lr × steps`. Flat 1e-4 has its own record too, and so does every
grid since.

**Do not read the clip norms as evidence about these rates.** Under Adam a
uniform rescale of a module's gradients cancels in `m̂/√v̂`, so gradient
magnitude says nothing about whether a learning rate suits the module.

**Do not reach for the LR range test either.** `PASS._apply_range_test_result`
calls `set_all_lr` and then overwrites `original_param_groups`, which would
flatten every split rate — the module groups' and the scalars' alike —
permanently.

### The overrides, and why each is gated the way it is

Note this subsection still names `BilinearGRUComparer` and
`TransformerCrossAttentionComparer` below, which the listener split replaced with
`BilinearDiscriminator` and `AttentionDiscriminator`, and it predates
`mix_logit_lr`. The reasoning holds; the names want a pass with the rung
overhaul.

**`logit_scale_lr`** — ungated. `log_logit_scale` exists on both speakers.

**`polarity_embedding_lr`** — gated on `isinstance(language_model,
SenderTransformerLM)`. Deliberately *not* tied to `logit_scale_lr`:
`log_logit_scale` exists on *both* speakers, so raising it to help the polarity
tag would also retune the GRU baseline's channel and shift the comparison the
ablation is there to make. They opened at one number while the tag had a
traverse to cover; since `2026-08-29` the tag takes the speaker's module rate,
`2e-4`, and the channel scalar is at `6e-3`. Separable was always the point, and
now the two values say so.

Gated on the speaker class rather than on finding the parameter, because the two
failures need different answers. A GRU speaker has no polarity tag by
construction — it reads `torch.cat(prototypes, 1)` and is told which is which —
so the key is simply inapplicable, exactly as `heads` and `ff_ratio` are, and
skipping is right. A Transformer speaker *missing* the parameter is a rename, and
`split_out_parameter` raises.

**`score_scale_lr`** — ungated, since `7b10d47`. `ScoreVolume` puts one
`log_score_scale` on every discriminator, so one key and one suffix reach both,
and the `mix_scale_lr` that once moved `AttentionDiscriminator`'s own scalar has
no successor. It was briefly gated on the bilinear class, when the other arm's
readout was a plain `nn.Linear(d_model, 1)` whose weight carried the volume and
there was no lone scalar for a rate to apply to.

**`score_bias_lr`** — ungated for the same reason, and added by the commit that
gave `ScoreVolume` a `score_bias` beside its volume. One offset per
discriminator, so one key and one suffix reach both, and
`AttentionDiscriminator.mix_bias` — which had no key at all — has no successor
either. Elevated to 2e-3 like every other lone scalar here: a 0-d parameter moves
about `lr` per step whatever its gradient, so its whole travel is bounded by
`lr × steps`, and at the base 1e-4 `mix_bias` could cover 0.58 in thirty birds
epochs against a score opening at 0.577 spread.

Elevated, and that was once the accusation: at 2e-3 the listener could squash its
own logits fast, which multiplied down the gradient reaching the speaker. The
accusation was wrong rather than answered — AdamW updates by `m / √v`, so a
uniform factor on a parameter's gradient cancels, and `clip_gradients`
renormalises per submodule whatever survives that. `7b10d47` answered it by
hiding the scale from the backward pass; `485b38e` removed that helper, since it
made the forward and backward disagree about what the module was. So a fast
calibration is now just a fast
calibration.

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

The groups are `models.builder`'s — eight modules and four lone scalars, see
[the group table](#the-group-table) — and an `other` group catches anything a
future architecture adds, so no parameter can silently go unclipped. `other`
being non-empty is the alarm and not the fix: it held the whole of
`sender.contrast` until August 2026, because `train.py` kept a list of its own
that had never been given the stage.

Every group's pre-clip norm is recorded, including ones that did not reach the
ceiling, as `train_clip_<group>` in `metrics.csv`. They are averaged **per
optimiser step** rather than per example — a gradient norm is a property of the
step — and NaN-filled where the group does not exist on that architecture, so
the header keeps its shape across a resume against a config that toggles a
stage. These columns are new in August 2026: `clip_gradients` had built the
norms and documented them "for logging" since it was written, and the call site
discarded the return, so no run before then recorded one.

## AMP and accumulation

The forward pass runs under `autocast` in bfloat16 where supported and float16
otherwise. Gradients are scaled, accumulated over `accumulator_steps` batches,
then **unscaled, clipped, stepped** — that order per the PyTorch AMP gradient
clipping recipe. A trailing partial accumulation is flushed after the loop.

The scheduler is `gradboard.scheduler.PASS`, driven by
`gradboard.cycles.CycleSequence` over up to two stages: an `ascent` warm-up of
`warm_up_epochs`, then `lr_schedule_shape` for the remainder. `train.build_lr_
schedule` builds it, and its docstring is the fuller account.

**Two invariants**, and the second is what makes the first safe:

* the warm-up ascends from **zero** to the configured rates, whatever else is
  set;
* the shape after it **opens at** the configured rates, so the handover is
  continuous.

`lr_schedule_shape` takes an intention — `flat` or `cosine` — rather than a
`gradboard.cycles.FN_LIBRARY` curve name. That restriction is the second
invariant: two `FN_LIBRARY` curves, `ascent` and `triangle`, open at their
*trough*, so a warm-up in front of one is discarded at the handover and
re-climbed over the remaining epochs. Naming intentions makes that unreachable
rather than merely unused. `parse_config.LR_SCHEDULE_SHAPES` holds the mapping;
`cosine` is `half_cosine`, the falling half, because `FN_LIBRARY`'s own `cosine`
is a full period that ends where it began.

`cool_point_multiplier` is the fraction of the base rate a descending shape falls
to, and it governs **only** the shape after the warm-up. `flat` does not descend
and takes no floor at all; `validate_config` rejects one set beside it rather
than leaving it unread.

### What this replaced, and what it means for old traces

Until 2026-08-28 the configured floor was passed straight to `PASS`, which
applies one floor across every cycle alike:

```python
min_lr = base_lr * self.cool_point_multiplier
current_lr = min_lr + (base_lr - min_lr) * self._schedule_multiplier
```

At `cool_point_multiplier = 1.0` the floor equals the ceiling, so every group sat
at its own `base_lr` for the whole run whatever shape the config named. `d5c47f5`
set that on 2026-08-09, deliberately and coherently, alongside
`warm_up_epochs = 0` and a flat shape. `b298da5` then set `warm_up_epochs = 10`
without touching the floor, and **the ten-epoch warm-up it advertised ran on no
rung**. Traces written between those dates ran flat whatever their `[scheduler]`
said, so nothing measured in that window is contaminated — but nothing in it
tests a warm-up either.

The floor now lives in the descending cycle's own `low` and `PASS` is built with
a floor of `0.0`, leaving its multiplier to act directly on each group's base
rate. `PASS` records `cool_point_multiplier` in its `state_dict`, so **a run
resumed from a checkpoint written before this change restores the old floor and
goes flat again**. Start such a run fresh rather than resuming it.

`tests/test_lr_schedule.py` asserts both invariants against a real `PASS` at real
step counts, and against every rung.

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
`log_score_scale`, `contrast_gate` — raising the effective batch delays it in
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
the positive-example count, `polarity_separation` at its opening — `sqrt(2·d_model)`
= 25.3 at 320 wide since `843dc81` drew the two polarity tags independently,
where the antipodal pair it replaced opened at `2·sqrt(d_model)` = 35.8.
The channel is carrying noise, the listener has nothing to learn from, and the
speaker has no gradient telling it to sharpen.

`score_scale` sliding downwards through this is a *symptom*, not the fault — it is
the listener correctly declining to be confident about noise, and it is not by
itself fatal (see [anecdotes.md](anecdotes.md), where a run does the collapse and
learns anyway). It is also not a mechanism, though not for the reason `7b10d47`
gave: the scale is back in the backward pass since `485b38e`, and what makes it
harmless is that AdamW divides a uniform factor out and `clip_gradients`
renormalises whatever survives that.

`score_bias` beside it reads differently, and is not part of the deadlock
signature: it should sit near zero throughout on balanced games, so it is a
finding rather than a symptom when it does not. See
[measurement.md](measurement.md).

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

**It runs after `build_models`, not before.** `build_models` writes the resolved
per-module learning rates back as `[optimiser] resolved_module_lrs`, which says
which groups were actually *built* as well as what each ran at, so saving first
would record a config that does not say what was built. Nothing is lost on the failure path — the
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

`[optimiser] resolved_module_lrs` is the one derived quantity that *is* recorded,
and deliberately: it is a `group → lr` mapping over the groups that were
constructed, so `sender_contrast` is absent when the stage is off and its
presence is itself a fact about the run. Read it as the answer to "which module
was trained at what rate", and note what it does not cover — the four scaling
scalars keep their own rate whatever their module's is, and `score_bias` and
`polarity_embedding` are moved again afterwards by `SPLIT_LEARNING_RATES`.

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

# Dubious claims

Statements carried over from the code that may be stale, unverified, or
self-contradictory. Nothing here has been checked against a run; this file exists
so that the claims are quarantined rather than silently trusted.

Note also that this whole documentation set was extracted from comments and
docstrings without running the code or the tests. Any measurement quoted anywhere
in `docs/` is somebody's earlier reading, not a fresh one.

## Claims that contradict the code as it now stands

### `Receiver`'s docstring said the comparer masks both operands

The removed docstring read: *"The listener's regularisation lives entirely in the
comparer, which masks the referent and message embeddings **equally** off a
single `dropout`."*

Both comparers now mask **the referents only** — `BilinearGRUComparer.__init__`
and `TransformerCrossAttentionComparer.__init__` both say so explicitly and give
the reason (the message already arrives through a calibrated noise process). The
`Receiver` docstring was not updated with them. The *conclusion* it draws — that
a dropout on `Receiver` would compose with the comparer's mask — still holds, but
the premise as written did not.

### "Because the scale is a constant, the ratio between the two is fixed"

`SenderGRULM.decode` carried a comment saying `tau` and `logit_scale` sit at one
fixed ratio for the whole run "because the scale is a constant". **The scale is
not a constant any more** — it is `log_logit_scale`, a learned parameter — and
`sampling_tau` is explicitly a cosine schedule over a ratio that moves with it.
The comment predates that change and was straightforwardly wrong when removed.

### `train.py`'s module docstring

*"Train an RNN decoder to make binary predictions; then train an RNN language
model to generate sequences."* There is no two-phase training in this file, and
the models are not necessarily RNNs. Inherited from an ancestor of the script.

### `run()`'s parameter list

The docstring documented `model`, `criterion` and `args`. The signature has
`pair`, `optimizer`, `dataloaders`, `scheduler`, `scaler`, `config`,
`random_state`, `compute_topsim`. `criterion` is constructed inside the function
and `args` was replaced by `config`; commented-out `args` parameters lingered in
the signature and in `get_true_lang` and have been removed.

### `builder.py` on `score_scale_lr`

The comment on `score_scale_lr` has now argued three positions in turn: that both
comparers carried `log_score_scale` and so the key was ungated; that only the
bilinear one did and so it was gated; and — since `a9a6a9c` deleted both scalars
and `7b10d47` restored one shared `ScoreVolume` — that the key is ungated again,
for a different reason than the first time. Each text was right about the code it
described. The passage is a record of how often this particular parameter has
moved.

### `layer_norm_logits` on its own headroom

Also self-correcting: *"the headroom is much smaller than the raw logit scales
quoted in `1510a55` suggest ... a margin of roughly 24×, not the three orders of
magnitude previously claimed here."* Both numbers appeared in the same comment at
different times.

### "156 steps an epoch" and "62 steps an epoch" are stale, not misattributed

`measurement.md` derives `logit_scale`'s per-epoch ceiling as *"at 2e-3 over 156
steps an epoch"* for birds, and `builder.py`'s `contrast_gate_lr` comment gives
"62 steps an epoch on birds" for the same quantity. Neither matches the current
config, which has `games_per_epoch = 3100` at `batch_size = 16` and
`accumulator_steps = 1` — ~194 optimiser steps an epoch.

This entry previously guessed that 156 might be ShapeWorld's number attached to
the wrong dataset. It is not, and the reason the guess was tempting is the
interesting part. `DEFAULT.toml`'s `logit_scale_lr` comment shows its working:
156 is **birds at `games_per_epoch = 2500`** (2500 / 16 = 156.25), and the
comment then supersedes it with 194 at the current 3100. It coincides with
ShapeWorld's 156 *by construction* — the note beside `games_per_epoch` says 2,500
was picked partly to equalise the two datasets' step counts. So the figure is
right for birds and identical to ShapeWorld's for a reason, which is exactly the
shape of coincidence that invites a misattribution reading. `builder.py`'s 62 is
the same quantity at an older `games_per_epoch` again — 62 × 16 ≈ jayelm's 1,000.

Both figures are therefore the right dataset at a superseded config: a duller
failure than misattribution, and a likelier one to recur, since the arithmetic in
these comments is pinned to a key that has now moved twice. The *form* of the
argument — a scalar cannot travel further than `lr × steps` — is unaffected. The
entry stays because the stale numbers are still in the source.

Note also that the 20,000 figure is itself read from a profiling log for a
different run (`13_shapeworld_sender_transformer_lm_latent`), not from the loader.

### `record_polarity_separation`'s docstring says the tag opens at zero

The docstring reads *"the only part of `polarity_embedding` the cross-attention
can act on, opening at exactly zero"*. `reset_parameters` initialises the tag as
an antipodal pair of `randn(d_model)` draws, so the separation opens at
`2·sqrt(d_model)` ≈ 35.8 on a 320-wide speaker, which is what the runs show and
what `measurement.md` documents. The docstring predates the antipodal init and
describes the zero init it replaced.

## Claims that depend on an external version

These were true of some version of a dependency and have no guard in the code.

- **`transformer_initial_ff_residual_path=True` is safe because "broccoli 30.1.0
  carries the residual with `ResizeAndPadPatches`".** Version-specific by its own
  wording. The claim that it "only raises below `d_model = 3`" is likewise a
  reading of that version.
- **`gradboard`'s `EXCLUDE_FROM_WEIGHT_DECAY` matches `"embedding"` as a
  substring**, which is what keeps `polarity_embedding` out of weight decay. The
  code's own comment says renaming the parameter would reintroduce decay
  silently; the same is true if gradboard changes its matching rule.
- **`get_optimiser` keys its groups on `(lr, weight_decay)`**, and **`PASS`
  deep-copies `original_param_groups` at construction and zips them
  `strict=True`.** `split_out_parameter`'s ordering requirement rests entirely on
  both.
- **broccoli's docstrings say `positional_heads` is applied with `floor` while
  the code is `math.ceil`.** Recorded as a reason not to use a fraction; not
  re-checked.
- **broccoli's `ViT` resolves `ff_ratio` / `ff_inner_size` in the opposite order
  to `FeedforwardBlock`.** The `ViT2` call site is written on the assumption that
  a non-`None` ratio silently discards the explicit inner size.
- **`MHAttention` asserts `query_tokens == seq_len` whenever causal.** The whole
  re-read-the-prefix design of `decode_autoregressively` and `TransformerDecoder`
  rests on this.
- **`MHAttention` has no residual** (returns `out_norm(out_proj(attention))`),
  which is the premise of the `latent_length` bandwidth argument.
- **`TransformerEncoder.preprocess` adds its position embedding in place
  (`x += ...`) on the caller's tensor.** The comparer's stage 2 is documented as
  relying on nothing reading `messages` afterwards.

## Live assumptions with no enforcement

### `BilinearGRUComparer` reads timestep `-1`

Correct only because messages are never padded. The code's own note calls the
assumption *"dormant, not satisfied by design elsewhere"*, and names two things
that would break it:

- EOS dropped from the reserved mask, to let the sender choose message length
  (the more faithful reproduction of the original paper) — at which point this
  reads post-EOS junk;
- anything feeding padded language to the receiver, e.g. an ACRe-style eval
  replaying sampled messages.

Nothing does either today; every call site takes `lang` straight from the sender.
The bidirectional branch has the mirror exposure: it reads the backward pass at
position 0, which under end-padding is the state after that pass has run
*through* the padding.

### `AveragePrototyper.forward(labels=...)`

Documented as existing "for backwards compatibility" and not read, on the grounds
that the first half of the examples is always positive.
`Sender.get_prototypes` does assert exactly that before calling, so the claim
holds today — but the argument is still a silent no-op.

## Dead or near-dead code

- **`cub.LOAD_INTO_MEMORY = True`** is never read; the only reference is a
  commented-out `if`. The behaviour it named is now unconditional.
- **`shapeworld.load(config, fast=False)`** — no caller passes `fast`;
  `data.loader.load` calls `lf(config)`. The parameter reaches
  `load_split(fast=...)`, where it short-circuits shape parsing, and is
  effectively subsumed by `need_shapes`.
- **`CUBDataset.vis_input`** has an unreachable bare `return` immediately after
  `return img_html`.
- **`vision.py`** still carries a large commented-out block under the heading
  `# MODELS BELOW HERE SHOULD NOT BE USED`: `ConvNetNopool`, `ConvNetS`,
  `ConvNetSNopool`, `Conv6`, `Conv4NP`, `Conv6NP`, `Conv4S`, `Conv4SNP`,
  `ResNet10`, a module-level `reset_parameters`, `PretrainedResNet18`, `ResNet34`,
  `ResNet50`, `ResNet101`. It was left in place: commenting-out is somebody's
  deliberate archive rather than an expository comment, so removing it is a
  separate decision. `distLinear`, `Linear_fw`, `Conv2d_fw`,
  `BatchNorm2d_fw`, `BottleneckBlock` and the `maml` class flags are live code but
  unreferenced by anything this repository builds.
- **`ResNet(flatten=False)`** sets `final_feat_dim = [indim, 7, 7]`, which
  hardcodes the 224px assumption the adaptive pool was introduced to remove. No
  caller uses `flatten=False`.
- **`data.loader.worker_init`** calls `torch.seed()` and discards the result;
  only the `np.random.seed()` call has an effect on the worker's stream.
- **`shapeworld.get_metadata`** raises `ValueError(f"Unknown feature type
  {this_md}")` in a branch where `this_md` may be unbound.
- **`SenderTransformerLM.__init__`** carried a docstring consisting of `...` and
  a bare arXiv link (<https://arxiv.org/abs/2502.20604>). The link is preserved
  here because it is presumably the intended reference for the architecture; what
  it is a reference *for* was never written down.

## References that were not followed while extracting

Files and commits named in the comments, quoted throughout `docs/` on trust:

- Commits: `87c1027`, `17ae9f9`, `3b3b857`, `29b18ea`, `e3fcabd`, `fccba0f`,
  `1510a55`.
- Files outside `code/`: `diagnostics/README.md`, `diagnostics/bootstrap_probe.py`,
  `issue.csv`, `receiver-cross-attention-birds.csv`,
  `receiver-cross-attention-shapeworld.csv`, `probe_shape.py`, `DEFAULT.toml`,
  `data/save_cub_np.py`, `scripts/prepare_data.sh`.
- Tests asserted to pin behaviour: `tests/test_exploration.py`,
  `tests/test_emergence.py`, `tests/test_backbones.py`,
  `test_the_referent_norm_is_not_a_global_rescale`,
  `test_the_cross_attention_norms_that_must_be_affine_free_are`,
  `test_unique_message_fraction`.
- "Rung" numbers (1, 3, 10, 11, 12, 13, 14) refer to the ablation ladder's config
  files, which live outside `code/`.

# Working with broccoli

`broccoli` (`broccoli-ml`) supplies the transformer primitives, the ViT, the
activations and `SequencePool`. This file collects the conventions this
repository applies when calling into it, and the DeepNorm residual scaling that
`models/model_util.py` resolves.

## Every argument is passed explicitly

**broccoli's defaults are not a stable interface.** Between 27.1.1 and 30.1.0
`TransformerEncoder` flipped from `pre_norm=True, post_norm=False` to the
reverse, which would have silently inverted the architecture of every model here
on a `pip install` — with no error and no diff in this repo.

So every broccoli module in `receiver.py`, `sender.py`, `vision.py` and
`transformer_decoder.py` is constructed with its full argument list, **including
arguments that are inert under the current settings**, so that a future default
change cannot quietly make them live.

`util.dependency_versions` records the resolved broccoli commit alongside every
run for the same reason.

## Arguments pinned rather than promoted to config

### `absolute_position_embedding=False`

Pinned everywhere, and no longer a config option. Every stack here runs rotary,
which encodes position where it is used — as a rotation of the query and key
subspace — rather than as a vector added to the residual stream once at the
input. A rotation of part of the vector is sufficient for relative position, so
an absolute embedding on top is not covering anything RoPE leaves out; it is a
second, differently-shaped answer to the same question, learned from scratch, and
one that has to be re-learned for every sequence length.

broccoli agrees: its own `ViT` defaults to exactly this pair — `False` absolute,
`True` relative.

It cost ~190k parameters a rung, most of it two 289-position tables in the `ViT2`
backbones.

### `positional_heads=1.0`

Pinned at 1.0 everywhere, so every head receives axial RoPE, and no longer
configurable. At any fraction below 1 broccoli splits the head axis —
`math.ceil(fraction × n_heads)` heads take RoPE and the rest are carried through
a second value projection and concatenated back — so the size of the partition
moved whenever `heads` moved, which made it a hidden confound in a study that
varies width. At the full head count every one of those splits collapses to the
identity path.

It is also pinned at sites where it is *inert* (wherever `rotary_embedding` is
`None`), so that turning rotary on there could not quietly introduce a head
partition.

Note broccoli's docstrings say the fraction is applied with `floor`; the code is
`math.ceil`. Another reason not to sit on a fraction. And broccoli's own defaults
differ by class — 0.25 on `MHAttention`, 0.5 on `TransformerEncoder` — so neither
is a value to inherit.

### `ff_ratio=None`, `ff_inner_size=<configured>`

`ff_inner_size` is the live knob and `ff_ratio` must be `None`, but **for
opposite reasons in the two classes**, so this is easy to get wrong:

- `FeedforwardBlock` takes `int(ratio × width) if inner_size is None else
  inner_size` — the explicit size wins.
- broccoli's `ViT` takes `if transformer_ff_ratio is not None:` and derives the
  inner size from the ratio, **discarding** whatever explicit size was passed.

So on the `ViT2` call site, leaving `ff_ratio` as a number would make the
explicit `ff_inner_size` silently dead.

### `ff_dropout=0.0`

Not configurable, and pinned rather than promoted. broccoli's `FeedforwardBlock`
uses this only as a fallback — `inner_dropout if inner_dropout is not None else
dropout` — and `TransformerEncoder` always forwards `ff_inner_dropout` and
`ff_outer_dropout`, which default to `0.0` rather than `None`. So the argument
can never take effect, and TOML has no way to write the `None` that would let it.
Use the inner/outer knobs instead.

### GRU `dropout=0.0` on the listener

Fixed at 0.0 to match jayelm, whose listener GRU takes no dropout argument at all
(`rnn.py:21`). Inert either way while `layers = 1` — PyTorch only applies this
*between* layers — but wiring the knob in meant it would switch on unannounced
the moment anyone raised `layers`. Zero also silences PyTorch's warning about
that combination.

### `ViT2`-specific pins

**`cnn=False` and the whole `cnn_*` group.** Inert while `cnn` is False: broccoli
swaps in an Identity and the image goes straight to pooling. Pinned so that
flipping `cnn` on is a deliberate act with visible settings, rather than silently
adopting defaults.

**`transformer_initial_ff_residual_path=True`.** On, because broccoli 30.1.0
carries the residual with `ResizeAndPadPatches`: it bilinearly downscales each
patch to the largest volume that fits `d_model` and zero-pads the rest, so the
skip connection no longer ties `d_model` to the patch size. It was off here on
the older assumption that it did. The resizer has no parameters, so this costs
nothing but the interpolation, and it only raises below `d_model = 3`
(`in_channels`), which no config approaches.

**`transformer_initial_ff_*_dropout=None`.** `None` means "fall back to the
corresponding `transformer_ff_*` value", which is 0.0 in each case — not "no
dropout argument".

**`batch_norm_logits=False`.** Pinned, and deliberately not a config key.
`image_classes` is `d_model`, so what broccoli calls the logits is this
backbone's output *embedding*, and this flag would put an
`nn.BatchNorm1d(d_model, affine=False)` on it as the last operation before the
prototyper or the comparer ever sees it.

Both objections are ones this repository already accepted one level down, when
`layer_norm_logits` replaced an `nn.BatchNorm1d` over the vocabulary logits:
BatchNorm keeps running statistics, so train and eval normalise by different
numbers and every `test_` column is read through an estimate the training pass
never used; and it couples to the batch, which means it also couples to
`accumulator_steps`.

The batch coupling is *worse* here than it was there. The listener forwards
`batch × n_objects` images in one flat call, so this normalised targets and
distractors from *different games* against each other — a referent's embedding
depended on which other candidates happened to share its batch. The speaker has
the same shape one step earlier, where the pooled positives and negatives of
unrelated games meet in the same statistic.

Turning it off leaves this backbone's output unnormalised: `SequencePool` into a
plain `Linear`. That is the intended state. Whichever consumer needs the referent
at a controlled magnitude should normalise it where the score is formed —
`SenderTransformerLM.referent_layer_norm` and
`TransformerCrossAttentionComparer.referent_layer_norm` both already do — rather
than have one flag inside the vision model decide it for every consumer at once,
per batch, differently at eval. Note `BilinearGRUComparer` has no such norm, so
its score inherits whatever magnitude the backbone emits.

**Pooling geometry is derived from the image size, not configured**: these size
the patch grid, and so the transformer's `source_size`, from the data.

### `causal` on the latent arm's stack

Never causal, and no longer configurable. It used to read `not bidirectional`,
which masked the *latent* array left to right — and the latent array is not a
sequence in time. It is read back out in one shot by `output_query`, so latent
index 0 is no earlier than latent index 9 in any sense the task can see. All that
mask did was hide most of the image from the low-index latents, which is exactly
what the encoder cross-attention declares it does not want ("whole image informs
whole latent array"). Ordering the message is the decoder arm's job, and it does
it by conditioning rather than by masking this.

### Attention dropout is its own key

Attention internals take `cross_attention_dropout` / `self_attention_dropout`,
never the agent's `dropout`, which regularises the inputs to the comparison. The
speaker's cross-attention used to read `self.dropout` while the listener's
matching cross-attention took a separate constant, so raising the speaker's input
regularisation silently rewired its attention and the two agents were regularised
on different terms. In the decoder arm, the per-block cross-attention takes the
speaker's `cross_attention_dropout` too, so every cross-attention in the speaker
is on one knob.

broccoli gates attention dropout on `self.training` (the `dropout_p` argument in
`MHAttention.forward`), so it does not leak into eval the way a bare
`F.scaled_dot_product_attention(dropout_p=...)` would.

## Activations (`model_util.ACTIVATIONS`)

Name → broccoli activation, so that `activation` can be set from a TOML config.
This mirrors the map broccoli's own `ViT.__init__` applies to string arguments,
and deliberately offers the same four options: `ReLU`, `GELU`, `SquaredReLU`,
`SwiGLU`. `TransformerEncoder` has no such map and takes the class (or, for the
GLU variants, the factory) directly, so the lookup has to happen on this side.

`get_activation` raises `ValueError` naming the valid options, rather than
letting a config typo surface as a bare `KeyError` from deep inside model
construction.

## DeepNorm residual scaling

From Wang et al. (2022), *"DeepNet: Scaling Transformers to 1,000 Layers"*
(<https://arxiv.org/abs/2203.00555>), for a post-norm stack `layers` deep:

```
encoder:  alpha = (2N)^(1/4)    beta = (8N)^(-1/4)
decoder:  alpha = (3N)^(1/4)    beta = (12N)^(-1/4)
```

`alpha` and `beta` in the config are each either a number, which is passed
through untouched, or the string `"deepnorm"`, which is replaced by the derived
constant. **Mixing the two is allowed**: pinning one while deriving the other is
a coherent thing to want, and refusing it would only push the arithmetic back
into the config. A string that is neither raises, rather than reaching broccoli
and failing inside a forward pass.

The sentinel exists so that a config changing `layers` does not have to restate
two constants.

### Which form is right is decided by the block, not by the name

DeepNorm's decoder constants assume a cross-attention sublayer inside *every*
block, so a block with three residual branches takes the decoder form and a block
with two takes the encoder form.

**Most stacks here are the encoder case, and not by accident.**
`SenderTransformerLM`'s latent arm runs one cross-attention over the prototypes
to build the sequence its encoder then reads, and
`TransformerCrossAttentionComparer` runs one between two encoder stacks. In both,
the cross-attention sits outside the residual path whose depth this is correcting
for, so it is not a sublayer and the encoder constants are the right ones.

**The exception is `transformer_decoder.DecoderBlock`**, which
`SenderTransformerLM` builds for its autoregressive arm. That one does
cross-attend into the latent memory inside every block, which is the
configuration DeepNorm's decoder form was derived for, so it asks for
`decoder=True`.

### Depth is per sub-stack

`layers` is the depth of the residual path being scaled. For a stack split into
sub-stacks, each resolves its own. `TransformerCrossAttentionComparer` resolves
two pairs, one per decoder stack: `message_layers` for the stack that encodes the
message and `referent_layers` for the stack that scores the candidates. Both ask
for `decoder=True`, because a `DecoderBlock` carries three residual branches
rather than two — self-attention, cross-attention and feedforward — so the
sublayer count is `3N` and the constants are `(3N)^(1/4)` and `(12N)^(-1/4)`.

The depth passed is the block count, not a multiple of it: `deepnorm_constants`
does the three-per-block accounting itself. An earlier version of this module had
three hand-written residuals resolved at depth 1, on the argument that bare
attention sublayers with no feedforward are one layer's worth of residual path;
those residuals are gone, and with them the special case.

`ViT2` derives from `layers` alone, which counts the transformer blocks; the
initial feedforward residual path it also scales with `alpha` and `beta` is one
branch outside that count, and a quarter root is forgiving enough that it is not
worth a second constant.

### The constants set an opening, not a ceiling

broccoli applies `beta` as a forward multiplier on the branch rather than as an
initialisation scaling on its projections, and the post-norm `RMSNorm` that
follows carries a learnable gain — so a branch that earns it can learn its way
back out of the opening ratio.

`deepnorm_constants` raises below one layer: a stack with no blocks has no
residual path to scale, so pin `alpha` and `beta` to 1.0 instead.

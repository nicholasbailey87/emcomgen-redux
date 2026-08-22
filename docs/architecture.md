# Architecture

The task is a reference/concept game. A **sender** sees a set of positive and
negative examples of a concept and emits a fixed-length message; a **receiver**
sees a different set of candidates and scores each one. `models/base.Pair`
holds the two so they train as a unit.

## Sender

`Sender` is three modules in sequence:

1. `feat_model` — a vision backbone, embedding every image independently.
2. `prototyper` — pools the positive examples into one vector and the negative
   examples into another.
3. `language_model` — turns the two prototypes into a message.

### Two dropouts, not one

`vision_dropout` sits between the backbone and the prototyper, i.e. on
per-image embeddings, before pooling. `prototype_dropout` sits between the
prototyper and the language model, on the pooled concept vectors. jayelm's
single `--dropout` is the latter. Dropping features *before* the pool is much
weaker, because the average over n/2 examples largely restores them — which is
exactly why the two are not redundant.

The listener has no counterpart to `vision_dropout`. Its regularisation lives
entirely in the comparer, and a dropout on `Receiver` would land on the same
tensor the comparer masks with nothing but a reshape between them, so the pair
would silently compose into one mask at a rate neither knob names.

### `speak` versus `forward`

`Sender.speak` returns the message, the symbol embeddings behind it, **and**
the concepts, all from one pass. Compositionality analysis needs all three
paired: the soft signal distances compare embeddings of symbols that were
actually emitted, and the semantic distance is measured between concepts.
Fetching them through separate calls would resample the `vision_dropout` mask
and — for an autoregressive speaker — the message itself.

`Sender.forward` is the training path and returns message plus concepts only.
It prototypes once and reuses, for the same reason: a second `get_prototypes`
call would re-run the vision model under a fresh dropout mask.

### `reset_parameters` has no `hasattr` guard

It used to. The guard existed for `ViT2`, which had no `reset_parameters`; it
now has one, and every other feature model already did. A guard turns a missing
method into a silently skipped backbone rather than an error — which is how the
speaker's ViT went unreset. See [anecdotes.md](anecdotes.md).

## Prototypers

### `AveragePrototyper`

The mean of the positives and the mean of the negatives. The `labels` argument
exists for signature compatibility and is not read: the first half of the
examples is always positive by construction (`data.util.split_spk_lis`).

### `AttentionPrototyper`

Pools each polarity with `SequencePool`'s learned attention — one scoring
direction per polarity, softmaxed over the examples — rather than averaging.
Two departures from a bare `SequencePool`, both there to stop the *softmax over
examples* inheriting the problem the softmax over tokens had before
`layer_norm_logits`: a pre-softmax input whose magnitude is set by the backbone
rather than by anything learned.

**Zero-initialised scoring weights.** Scores are then equal across examples, the
softmax is exactly uniform, and the prototype is exactly the mean — so this rung
opens at `AveragePrototyper`'s behaviour and can only depart from it where the
loss pays for the departure. That is what an ablation rung should isolate. Left
at broccoli's default init, the opening pooling is an arbitrary weighting, and
how arbitrary depends on the feature scale: with a random scoring direction the
softmax's sharpness goes as the between-example standard deviation of the
embeddings, so at Conv4's scale a fresh pooler is within a whisker of selecting
one example, while at a normalised backbone's it is within a few percent of the
mean. Two arms of the ladder would then differ in their pooling as well as in
the thing being ablated.

Zeroing a weight matrix is safe here in a way it is not for a hidden layer:
there is one output unit, so no symmetry between units to break, and
`dL/dW = sum_i (dL/ds_i) x_i` depends on the examples rather than on `W`, so it
is non-zero as soon as the examples differ and the loss cares which of them
carries weight. The softmax Jacobian at uniform weights, `(1/n)(δ_ij − 1/n)`, is
full rank on the zero-sum subspace and transmits it. Only the bias is inert — a
constant added to every score cancels in the softmax, so it has exactly zero
gradient — and it is zeroed too, to say so.

**A parameter-free `LayerNorm` on the scoring path only.** The pooled *values*
are the raw embeddings, so prototype magnitude is unchanged and whatever the
backbone emits still reaches the language model intact; only the scores are
computed from normalised examples. This is what makes the rate of departure from
the mean comparable across arms. Growth in the scoring vector's norm buys score
spread in units of the embeddings' scale, so without it an arm on a
large-magnitude backbone leaves the mean tens of times faster than one on a
normalised backbone — the same architecture-dependence, relocated from where the
pooling starts to how fast it moves. `elementwise_affine=False` on purpose: a
learnable gain here would be one more route to score magnitude that differs by
arm, which is the thing being removed.

`reset_parameters` runs broccoli's own reset *first* and only then overrides the
scoring projection, so that any parameter `SequencePool` grows in future is
still initialised the way broccoli intends. The override lives here rather than
in broccoli because `SequencePoolClassificationHead` uses the same module and
wants the usual init.

## Speaker language models

### `SenderGRULM`

The speaker of *Emergent Communication of Generalizations*
(<https://arxiv.org/abs/2106.02668>). `init_h` reads `torch.cat(prototypes, 1)`,
so each polarity lands in its own slice of the input and gets its own weight
columns — which is why this speaker needs no polarity tag and reports
`polarity_separation` as NaN.

`decode` runs the sampling loop and returns both the message and the symbol
embeddings that produced it. Because the speaker is autoregressive, each
embedding depends on the symbols sampled before it, so embeddings and message
correspond only when they come from the *same* call. `forward` discards the
embeddings, so any analysis needing the two paired must call `decode` directly,
as `Sender.speak` does.

There is no greedy or epsilon-greedy generation option. The former is only used
in the parts of the original code relating to ACRe; the latter is off by default
and is not discussed in the original paper.

### `SenderTransformerLM` — two architectures behind one flag

The flag is `bidirectional`, exactly as it is on `SenderGRULM`.

**`false` (the default) — an autoregressive Transformer decoder.** A learned
query cross-attends the two prototypes into a latent memory, and `layers`
decoder blocks generate the message one symbol at a time, each conditioned on
the symbols before it and cross-attending back into that memory. Causal in the
sense the GRU is causal, and comparable to it on generation regime as well as on
parameters.

**`true` — Perceiver IO.** The same cross-attention into the same latent array,
then `layers` blocks of self-attention over the latents, then a second learned
query reading them back out as every symbol at once. Nothing is conditioned on
anything; the message has an order only because its slots are numbered.

Both live in one class because they share everything up to the latent array —
the polarity tag, the encoder cross-attention, the logit scale and its
diagnostics — and because the ablation configs select between them by setting
one key.

The two arms are **not the same size at the same `layers`**: a decoder block
carries a cross-attention the encoder block does not, so it costs about
`4·d_model²` more — 1,354,951 against 944,711 at 320 wide with
`ff_inner_size = 554`. The ablation rungs therefore run the decoder arm at 4
layers and the latent arm at 5, which puts both within 2% of the GRU baseline.

#### The latent array and `latent_message_multiplier`

`latent_length = round(content_length × latent_message_multiplier)` is the
length of the array the self-attention stack runs over, as distinct from the
message it eventually produces. This is the Perceiver IO shape: a learned query
array cross-attends into the byte array (here, the two prototypes), a
self-attention stack processes the result, and a *second* learned query reads
that latent array back out at whatever length the task wants.

Perceiver's own reason for the split does not apply — its latent array is
*smaller* than its input so the quadratic attention stays affordable, whereas a
byte array of two prototypes is smaller than anything. What earns the split its
place here is **bandwidth**. `MHAttention` has no residual (it returns
`out_norm(out_proj(attention))`) and there are exactly two keys, so each query
position's entire dependence on the referents is one softmax weight per head.
The referents therefore reach the language model through `heads × latent_length`
scalars and nothing else — 20 of them under the pre-`latent_message_multiplier`
configuration. Lengthening the query array is the only knob that widens that
without touching `message_length`.

Rounded rather than floored, so the knob is symmetric about the integers; at the
configured 2.0 it is exact for every message length anyway.

The output query is built at multiplier 1.0 as well as above it, deliberately.
The knob's job is to vary the latent width and nothing else; if 1.0 also removed
a module then a sweep over it would confound two changes at once, and the
`state_dict` shape would move with the knob, so checkpoints could not be
compared across sweep points. At 1.0 it is a learned re-read of an array of its
own length — redundant, but honestly so.

#### The polarity embedding

A learned tag marking which row of the prototype sequence is the positive
concept and which is the negative one. Row 0 is positive, row 1 negative,
matching the order `Sender.speak` and `Sender.forward` hand over and
`Sender.get_prototypes` asserts.

Without it this speaker cannot read that order at all. The encoder
cross-attention carries no positional or rotary embedding on its key side —
correctly, since two prototypes have no sequence to encode — so its output is a
weighted *sum* over the keys and is bit-identical under swapping them. The
ordering is there in the tensor; there was no parameter that could condition on
it.

What survived the symmetry was a content cue — positives are a tight cluster and
negatives a diverse one, so the negative prototype sits nearer the global mean
with a smaller norm — and `referent_layer_norm` normalises each prototype
independently over its feature axis, which divides that norm difference out
before the attention ever sees it. Only direction was left, and at initialisation
not even that: an untrained backbone makes both prototypes the mean of noise. So
the cost was heaviest exactly during bootstrapping, where this speaker started
with zero polarity information while the GRU had it for free.

Design details, each load-bearing:

- **Added after the norm, not before.** This is the opposite of where a ViT puts
  its position embedding, for a reason that does not apply to a ViT: there the
  embedding rides a residual stream re-read by every pre-norm block, whereas here
  the prototypes are normalised once, consumed by one cross-attention and
  discarded. Inside a single LayerNorm the tag and the content compete for one
  unit budget, so growing the tag enough to be read reliably suppresses the
  prototype it is tagging. After the norm the two scales are independent.
- **Zero-initialised**, so the rung opens at the previous behaviour exactly, in
  the same spirit as `AttentionPrototyper`'s scoring weights: departure has to be
  paid for by the loss. The gradients still separate the two rows — `dL/de_i` is
  the gradient at row `i` of the sequence, and the rows differ in content — so
  there is no symmetry to break.
- **Not normalised or otherwise scale-pinned.** A unit-norm tag would be
  `sqrt(d_model)` and so exactly as loud as a whole normed prototype, which is an
  arbitrary number rather than a principled one, and nothing here needs the
  cross-arm scale comparability that earned `AttentionPrototyper` its
  parameter-free norm.
- **The name matters.** `gradboard`'s `EXCLUDE_FROM_WEIGHT_DECAY` matches
  `"embedding"` as a substring, so this lands at `weight_decay = 0.0`. A 2-D
  parameter would otherwise be decayed — and decayed *up*, by
  `sqrt(in_features)/sqrt(d_base)` — which on a zero-init tag is a force pulling
  it back to the zero it is trying to leave. Renaming it to anything without
  "embedding" in it reintroduces that silently. `polarity_embedding_lr` in
  `[optimiser]` is the other half of the same concern.

#### The decoder arm's sampling loop

`decode_autoregressively` is a step-for-step mirror of `SenderGRULM.decode` from
the sampling onwards — same normalisation order, same mask-then-explore order,
same scale-the-unmasked-logits-then-remask discipline, same greedy eval branch,
same per-step accumulation of diagnostics pooled once at the end. All the
reasoning behind those is in [channel.md](channel.md).

What differs is what carries the state. The GRU threads a hidden state through
the loop; this threads the symbols themselves, re-reading the whole prefix
through the stack at every step.

**Why re-read rather than extend:** broccoli's `MHAttention` asserts
`query_tokens == seq_len` whenever it is causal, so a growing prefix is not
something the module will accept. The loop therefore runs the stack over a
full-length sequence every step, with the positions after the cursor held at
zero. The causal mask makes those positions unable to reach the ones before
them, so they are inert and the result is exactly what a growing prefix would
have given. The cost is `content_length` passes over a `content_length`
sequence — five of them at ShapeWorld's message length, over a stack a few
million parameters wide, which is not worth a KV cache.

The input sequence is built fresh at each step rather than written into in
place. In-place would be the obvious way and does not work: the previous step's
forward pass has already saved that tensor for backward, so mutating it makes
autograd refuse.

The last symbol is never fed back — there is nothing left to condition — so the
sequence stays `content_length` long and position 0 stays the SOS.

Both speakers feed the *soft* one-hot through the token embedding
(`onehot @ weight` rather than an index lookup) so that the straight-through
gradient reaches the step that produced the symbol. The decoder arm's
`token_embedding` is sized at `d_model`, not `token_embedding_size`: the two are
required to be equal for this class anyway, and writing the width the stack
actually consumes keeps the dependency visible.

`embeddings()` is the latent arm's whole forward pass and is deliberately not
given a decoder-arm branch — the decoder arm's embeddings are not a function of
the prototypes, so there is no honest signature it could have there.

`encode()` returns the latent array **unnormalised**, because the two arms want
it normalised at different points: the latent arm feeds the raw array to its
self-attention stack and norms afterwards, the decoder arm norms once and hands
the result to every block as a fixed memory. Normalising per block would repeat
identical work `layers × content_length` times per batch.

## `models/transformer_decoder.py`

broccoli has no decoder. `TransformerEncoder` and its `EncoderBlock` are
self-attention and feedforward only, and every cross-attention in this repository
otherwise is a single `MHAttention` sitting *between* stacks — the speaker's
Perceiver IO encode/decode pair, and `TransformerCrossAttentionComparer`'s bridge
from the message to the referents.

A speaker that generates left to right needs the other arrangement: a causal
self-attention over the symbols emitted so far and a cross-attention into a fixed
memory, both inside every block, so that each layer can revisit the memory in the
light of what the layer below it made of the prefix.

`DecoderBlock` mirrors `broccoli.transformer.EncoderBlock` deliberately closely —
same residual scheme, same `alpha`/`beta` placement, same stochastic-depth draw,
same pre/post-norm branches — with one extra sublayer between the self-attention
and the feedforward. Everything inside is a broccoli module; what this file adds
is the wiring, so a change to broccoli's attention or feedforward reaches this
stack too.

Three residual branches rather than two is exactly the configuration DeepNorm
derived its *decoder* constants for; a stack built from these blocks must ask for
`decoder=True` (see [broccoli.md](broccoli.md)).

**Stochastic depth is drawn inside `forward`**, as `EncoderBlock` draws it, and
the same mask is shared by all three branches of a block so that a dropped block
is the identity rather than a partial one. Note what that means for a speaker:
`SenderTransformerLM` runs this stack once per symbol, so a message's five
positions are generated by five independent draws — five sub-networks, not one.
That is a deliberate choice. Hoisting the draw to once per message would make the
regulariser mean "drop this block for this utterance" instead of "drop it for
this symbol". Both are defensible; this one keeps the block a faithful mirror of
broccoli's, with no bespoke mask plumbing to drift out of sync when broccoli's
changes.

The cross-attention carries **no positional information on either side**. The
queries carry the decoder's own order, through the absolute position embedding
and the rotary self-attention; the memory's order is already baked into it by
whatever produced it.

`TransformerDecoder` mirrors `TransformerEncoder`: it owns the absolute position
embedding and the learned `bos_tokens` prefix, applies them in `preprocess`,
checkpoints each block, and strips the prefix on the way out unless asked to keep
it. Its stochastic-depth schedule is identical to `TransformerEncoder`'s,
including its treatment of the single-layer case, so the two stacks are
regularised on the same terms at the same setting.

The sequence length is fixed and the mask square over it because broccoli's
`MHAttention` asserts `query_tokens == seq_len` when causal — see the re-read
argument above.

## Receiver

`Receiver` embeds every candidate through the vision backbone, embeds the message
through a shared token embedding (`messages @ token_embedding.weight`, keeping
the straight-through gradient), and hands both to a comparer.

### `BilinearGRUComparer`

A GRU reads the message; a bias-free `nn.Linear` projects the final state into
referent space; the score is the dot product with each referent.

**Why no bias in the projection.** With a bias the score expands to
`obj·W·m + obj·b`. That second term is a message-independent *prior* that would
make the model prefer certain objects regardless of what the message said. The
bilinear form is deliberately pure.

**The `-1` timestep.** Taking timestep `-1` gives the state after the GRU has
consumed the last *slot*, not the last real token. That is correct here only
because messages are never padded: `mask_reserved_tokens` puts PAD/SOS/EOS/UNK at
`-inf` so the sender cannot emit EOS mid-message, and the decode loops always
build `SOS + (message_length − 2) content symbols + EOS`. Every message is
therefore exactly `message_length` long and position `-1` is always the real EOS.
This diverges from jayelm, whose speaker *does* sample EOS early and tracks a
per-example `lang_length`, and whose listener therefore has to
`pack_padded_sequence`. The assumption is dormant, not satisfied by design
elsewhere — see [dubious-claims.md](dubious-claims.md).

**Both operands are normalised**, per example over the feature axis, so the
score's magnitude stops being inherited from whichever vision model the rung
mounts. Both are in referent space: the message operand is `bilinear`'s *output*,
not the GRU state, because a norm upstream of a free `Linear` constrains nothing
downstream of it.

No affine on either norm. The score is `r·p`, so a per-dimension gain is
absorbable into `bilinear` and could only add a second, unbounded route to score
magnitude — the one these exist to close. It also keeps `sum(LN(r)) = 0`, which
is what annihilates the message operand's mean-subtraction; with a `beta` that
term would start shifting scores between objects.

**`log_score_scale`** is the listener's one degree of freedom over its own
confidence, and the counterpart of the speaker's `log_logit_scale`. Normalising
both operands closes every other route to score magnitude, and BCE is not
scale-invariant, so without this the listener could only ever sharpen by aligning
the two — never by committing harder to an alignment it already has.

One scalar, not one per operand: `c·LN(p)·LN(r)` and `LN(p)·c·LN(r)` are the same
function. It multiplies the message operand, which is shared across the objects
of a game, so it cannot change which object wins — only how loudly the listener
says so.

It opens at 1.0, which `forward`'s `1/sqrt(d)` makes the calibrated value: both
operands leave LayerNorm at norm `sqrt(d)` and start mutually random, so the
division puts the untrained score at unit standard deviation and BCE within a
hair of `ln 2` on both arms. Nothing here has a traverse to cover — unlike
`log_logit_scale`, which opens at 0.839 against a usable channel of 4 to 6.

It is stored as a log anyway, for different reasons: zero is where every gradient
in the pair is gated, since `s` multiplies the only path from the message to the
loss, and `exp` puts it out of reach; halving and doubling a gain should cost the
same step; and it gives `train_score_scale` a known ceiling of
`score_scale_lr × steps` log-units per epoch, which is what makes the column
readable rather than merely present.

The `1/sqrt(referent_embedding_size)` division is load-bearing rather than
cosmetic now that both operands are normalised. It is what makes
`score_scale = 1.0` the calibrated opening instead of a number whose meaning
moves with the embedding size — 512 on ResNet18 against 320 on ViT2, which would
otherwise open the two arms 1.26× apart and both far too loud.

**Dropout masks the referents only.** It used to mask the message operand too,
on the argument that a dot product lets the listener lean on whichever side is
left intact. True, but it assumed the two sides arrive on equal terms and they do
not: the message comes through the Gumbel channel, whose noise is already
calibrated by `sampling_tau` and `uniform_weight`, so a mask on top is a second
perturbation of a signal that has one — and the listener cannot tell which of the
two it is being asked to be robust to. The referents arrive clean.

### `TransformerCrossAttentionComparer`

Four stages:

1. **The message queries the candidate set** (`message_cross_attention`).
2. **`encoding`** refines that reading in context (a broccoli
   `TransformerEncoder`).
3. **Each candidate queries the refined message** (`referent_cross_attention`).
4. **The candidates query each other** (`referent_self_attention`).

Then a normalised linear readout scores each one.

**Why the message reads the referents before it is encoded.** Without that first
pass the encoder sees the message alone, so the best it can build is an
*absolute* meaning — "a red square" — when the task is discriminative and what
distinguishes the target from this particular set of distractors may be something
else entirely. The candidate set is not privileged information: the listener is
holding it. Letting the message query it is the difference between encoding what
the message says and encoding what the message says *about these objects*.

This costs the first cross-attention its position information, because `encoding`
is where position is embedded and it now runs second. So two identical symbols in
different slots query the candidate set identically, and `encoding` has to tell
them apart from context afterwards. Cheap at `message_length` 7 to 10, and the
alternative — lifting absolute position out of broccoli's encoder into this class
— buys little for the wiring it costs.

**Why every residual is post-normed rather than a bare add.** `MHAttention`
already RMS-normalises its output, so `x + attn(...)` adds two tensors of norm
`sqrt(d)` and the residual stream grows by `sqrt(2)` per stage — 2.8× across the
three here. Each add is therefore `RMSNorm(α·x + β·attended)`, which is what
broccoli's `EncoderBlock` does internally and what DeepNorm's constants are
derived for.

**Stage 4 is the only stage at which a score can depend on the rest of the set.**
Redundant for a criterion like "bigger than average", which the message could
carry on its own; load-bearing for one like "the odd one out", which no per-object
reading can express. Neither is in the task as it stands, and it is the stage this
class had all along — `fusion`, minus the feedforward.

**Stage 3's residual carries referent identity to the readout linearly.** Without
it a candidate reaches the score only through near-uniform attention weights, and
this stage halved the between-object share of the variance (0.415 going in, 0.221
coming out) at init.

**`layers` is the message encoder's depth and nothing else's.** It used to be a
total split between a reading stack and a fusion stack, which meant `layers = 5`
bought a 3-layer encoder and asking for one more block moved two.

**Stochastic depth is suppressed below two layers.**
`depthwise_linear_stochastic_depth` spreads the rate linearly across layers, so a
one-layer stack would get a single rate of 0.0 regardless. The three hand-written
residuals get none — they are not `EncoderBlock`s and have no branch to drop.

**Attention is never causal, and that is not negotiable** for the two stages
whose sequence axis is the referent set. In this codebase referent *order is the
label vector*: `data.util.split_spk_lis` writes positives into the first half of
each agent's view and negatives into the second, and the augmentation permutes
only *within* each half. Anything that could index its own sequence axis could
learn "the first half are targets" and score perfectly while ignoring the
message. With no position embedding and no mask, all three attentions are
permutation-equivariant and cannot read the ordering at all.
`BilinearGRUComparer` is immune for a different reason: it scores each referent
in isolation and never sees the set.

**`referent_adapter` has `bias=False`, and that is load-bearing rather than
tidy.** `referent_layer_norm` is what makes the score independent of the size the
vision model happens to emit, and it can only do that exactly if what reaches it
is homogeneous in the input: `W(cx) = cW(x)` gives `LN(W(cx)) = LN(W(x))`, where
`W(cx) + b` does not. With a bias, a backbone emitting features a hundred times
smaller gets a score shaped partly by this layer's bias and one emitting large
features does not — a weaker form of exactly the defect being removed, and one
that would leave the invariance test asserting an approximation. The following
norm subtracts the mean anyway, so most of a bias here would be annihilated a
line later.

**`referent_layer_norm` is parameter-free, and not for the reason originally
given.** broccoli's `project_qkv` RMS-normalises Q and K per head, so the
attention *logits* are already free of the vision model's scale, and
`MHAttention.out_norm` handles a uniformly louder backbone (measured: the whole
set at 10× moves the output by 0.0%). What neither handles is *per-object*
magnitude. At `message_cross_attention` the referents are the values; V is not
normed anywhere, so the attention output is a magnitude-weighted mixture: one
candidate 50× larger than its neighbours moves that output by 116% without this
norm and by 0.0% with it, and no downstream norm can undo it because the
averaging has already happened. That is an object winning for being large rather
than for matching. An affine here would also be a route to *global* score
magnitude.

**`input_dropout` is placed after `referent_layer_norm`,** not before the
adapter, which is where it used to be. A mask upstream of a learned projection is
a mask the projection can average away, and a mask upstream of a LayerNorm has
its `1/(1−p)` rescale thrown away and its survivors renormalised *up*, so the
perturbation is neither the size nor the shape the knob names. As on
`BilinearGRUComparer`, only the referents are masked; attention dropout is a
separate setting (`receiver_comparer.cross_attention_dropout`).

**`decision_layer_norm` is parameter-free and not optional, on two counts.**
broccoli's post-norm is `nn.RMSNorm(d_model)` with `elementwise_affine=True` by
default, so `referent_self_attention_norm` carries a learnable gain, and this
norm is the only thing standing between it and global score volume. It also
equalises the candidates' lengths, which nothing downstream ever could. Without
it, `scores` is `|refined_j| · cos(θ_j)` and an object can be read loudly for
being large rather than for matching — the same defect as the referent-norm case,
one stage later. With it, every candidate is read at norm `sqrt(d)` and only the
angle can separate them.

**The readout is a plain `nn.Linear(d_model, 1)` with a bias and a free weight
magnitude.** That makes one vector both the *direction* the head reads out and
the *volume* it reads out at, and it deliberately leaves the volume-collapse
route open. This is the current design after two attempts to close it; the full
history and the numbers are in [anecdotes.md](anecdotes.md). `decision_spread`
and `decision_kurtosis` are the columns that watch it — see
[measurement.md](measurement.md).

**Note stage 2 mutates its input.** broccoli's
`TransformerEncoder.preprocess` adds its position embedding with
`x += position_embedding`, in place, on the tensor handed to it. Harmless as
written because nothing reads `messages` again — but a second residual taken from
the pre-encoding message would silently be reading a positional embedding as
well, so take a copy first if one is ever added.

## Vision backbones (`models/backbone/vision.py`)

Adapted from
<https://github.com/facebookresearch/low-shot-shrink-hallucinate>. Every backbone
factory swallows its arguments, because `builder.py` splats the entire
`[*_feature_model]` config section into them and most of it applies to only one
backbone.

### `ViT2`

A thin wrapper over broccoli's `ViT`. The patch-grid geometry is derived from the
image size rather than configured:

```
pooling_kernel_size   = largest even number ≤ (max_side / 32) × 4
pooling_kernel_stride = kernel_size / 2
pooling_padding       = stride
```

`image_classes` is `d_model`, so what broccoli calls the logits is this
backbone's output *embedding*. It is left unnormalised — `SequencePool` into a
plain `Linear` — which is the intended state, not an oversight. Whichever
consumer needs the referent at a controlled magnitude normalises it where the
score is formed. See [broccoli.md](broccoli.md) for `batch_norm_logits=False` and
the rest of the pinned arguments.

### `ResNet` and the small-input stem

`ResNet18SmallInput` replaces the ImageNet stem — 7×7 stride 2 followed by a 3×3
stride-2 maxpool — with a 3×3 stride-1 convolution and no pooling, as SimCLR does
for CIFAR-10 (Chen et al. 2020, arXiv:2002.05709, CIFAR-10 appendix: "we replace
the first 7×7 Conv of stride 2 with 3×3 Conv of stride 1, and also remove the
first max pooling operation"). He et al. 2015 §4.2 is the underlying precedent,
though their CIFAR network is a separate architecture rather than a modified
ResNet.

The stock stem discards 4× resolution before any residual block runs. On
ImageNet's 224px that is proportionate; on ShapeWorld's 64px it leaves 16×16 into
stage 1 and a 2×2 map at the end, which the adaptive pool then averages to a
single position. What survives that is colour, and what does not is shape — which
is precisely the wrong bias for a study whose known failure mode is the speaker
learning to name colours. With the small stem the final map is 8×8 at 64px, so
the pool has 64 positions to average rather than 4.

Cheaper too, though only just: `3·3·3·64 = 1,728` parameters against
`7·7·3·64 = 9,408`, and the maxpool has none. The point is the resolution, not
the 7,680 parameters.

It is a separate factory rather than a flag on `ResNet18` because `ResNet18` is
pinned tensor-for-tensor against `torchvision.models.resnet18` by
`tests/test_backbones.py`, and because the backbone is selected by name from the
config — so a name is the whole of the registration.

**The final pool is adaptive** rather than `AvgPool2d(7)`, which hardcodes a
224px input. Below that, the pooling window is larger than the feature map and
the forward pass errors; above it, a single 7×7 window silently *crops* the map
rather than pooling it (at 320px the map is 10×10 and three rows and columns are
discarded), which also leaves `final_feat_dim` wrong. Numerically identical at
224, where the map is exactly 7×7. This matches torchvision's `resnet18`, which
is otherwise this network exactly: same layout, same stride placement, same
fan-out init.

`ResNet.reset_parameters` and `ConvBlock.reset_parameters` both reproduce
construction exactly rather than calling PyTorch's own `reset_parameters` — see
[anecdotes.md](anecdotes.md) for what they used to do instead.

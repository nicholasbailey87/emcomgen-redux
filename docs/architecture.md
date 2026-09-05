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

The listener has no counterpart to `vision_dropout`. Its one rate is
`[receiver] dropout`, applied by `Receiver` at the end of each referent
interface; a second between the backbone and those would land on the same tensor
with nothing but a reshape between them, so the pair would silently compose into
one mask at a rate neither knob names.

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

## `ExampleContrast` — the optional contrast stage

Both prototypers pool **within** a polarity: `AveragePrototyper` means each half
and `AttentionPrototyper` scores each half with its own `SequencePool`. So
nothing in the speaker compares a positive example against a negative one. The
two halves first meet in the language model's cross-attention, by which point
each is already a single vector, and whatever distinguished a positive example
from the distractors it was shown beside has been averaged away.

`[sender] contrast = true` inserts one self-attention over all `2n` referents
between the vision model and the prototyper, and adds its output back as a
residual:

```
x    = vision(samples)                        (batch, 2n, feat), post vision_dropout
h    = LayerNorm(adapter(x)) + label_embedding[labels]
out  = x + contrast_gate * out_projection(MHSA(h, h, h))
```

The prototyper downstream is unchanged and still receives the backbone's own
width, which is what lets either of them compose with this.

A boolean rather than a class name, because there is one of these or there is
nothing: it is a residual on the referents, so "off" is the absence of a module
rather than a different one. `Sender` holds it as `None` and guards on that,
never `hasattr` — see `reset_parameters has no hasattr guard` above.

### It opens at the identity, and the gate is what makes that survivable

`contrast_gate` is a scalar opening at exactly zero, so a speaker with the stage
on is bit-identical to one without it at step 0 and the arm is an ablation of one
thing. That is the same recipe as `AttentionPrototyper`'s zero-initialised
scoring weights and `AttentionDiscriminator`'s mix floor: open at the simple
behaviour, depart only if it pays.

A zero-initialised `out_projection` would open at the identity too, and it would
not move. AdamW steps a parameter by about `lr` per step whatever the gradient's
size, so the matrix would have to climb from 0 to its own init scale
`1/sqrt(d_model)` = 0.056 one `lr`-step at a time: 560 steps of perfectly
sign-consistent gradient at `lr` 1e-4, which on birds' 62 optimiser steps an
epoch is nine epochs of flat, optimistically. That is the arithmetic that made
the logit scale's traverse the bottleneck for those runs, and it is why
`contrast_gate_lr` exists at 2e-3 — fifty steps to 0.1 instead. A scalar is also
better shaped than a matrix here: `out_projection` starts at a properly scaled
random direction, so the branch contributes at a sensible magnitude the moment
the gate opens rather than having to build one first.

The gate is **not** log-parameterised, unlike `log_score_scale`. That is a
volume that must stay strictly positive and open at 1.0; this one must be able to
be exactly zero, which `exp` cannot reach. Its sign is free because the branch's own direction is arbitrary —
a negative gate is the same branch pointing the other way. And zero is a
starting point rather than a floor: `dL/dgate = <branch, dL/dout>` is non-zero
there, which is the same distinction that keeps the discriminator's mix floor in
the parameterisation and out of a `clamp`.

### Polarity arrives through the tag and nowhere else

`rotary_embedding=None`, so this attention is permutation-equivariant and cannot
read the first-half-positive ordering that the rest of the speaker relies on.
`label_embedding` — row 0 positive, row 1 negative, indexed from the labels
rather than from the halving index — is the only route by which polarity reaches
the stage, and it is initialised antipodally at unit per-element variance for the
reasons set out under [the polarity embedding](#the-polarity-embedding): it is
added after a parameter-free norm, so that is the scale of what it marks.

The name is deliberate twice over. It keeps `"embedding"` in it, which is what
holds it out of `gradboard`'s weight decay; and it is *not* `polarity_embedding`,
because `SPLIT_LEARNING_RATES` selects by name suffix and anything ending that
way — `contrast_polarity_embedding` included — would silently join the speaker
tag's parameter group.

The adapter carries `bias=False` for the same reason
every other `model_util.LinearInterface` does: the norm can only divide the
backbone's scale out exactly if what reaches it is homogeneous in the input,
which is what makes the *rate* of departure comparable across arms. The residual
is over the raw `x`, so what the prototyper pools is still at the backbone's own
scale — the same "score from normalised selves, weight over raw selves" split
`AttentionPrototyper` makes.

### What it costs

The message becomes a function of the sampled negatives rather than of the
concept alone: the same concept with different distractors gets a different
message. `topsim` measures exactly that correspondence, so this stage can raise
accuracy and lower compositionality at once, and that outcome is a finding rather
than a bug. `contrast_share` and `contrast_within_share` are what make it
reportable — see [measurement.md](measurement.md). Note also that
`AveragePrototyper` stops being a parameter-free control in this arm: its pooling
is still a mean, but what it means is no longer the backbone's output.

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

### `SenderTransformerLM` — one architecture, and `bidirectional` is a mask

A learned query cross-attends the two prototypes into a latent array, `layers`
blocks run over that array, and **the message is its tail** — the last
`content_length` slots, one per content symbol. That is the whole speaker on
both arms. `bidirectional` chooses whether the blocks are masked.

**`false` (the default) — causal.** The blocks are masked left to right, and the
tail slots are read in order: run the stack, read slot `first_message_slot + i`,
sample, and overwrite that slot with the symbol's embedding so the next slot is
conditioned on it. Causal in the sense the GRU is causal, and comparable to it on
generation regime as well as on parameters.

**`true` — parallel.** The same blocks unmasked, and the whole tail read in one
shot. Nothing is conditioned on anything; the message has an order only because
its slots are numbered.

The arms are the same size at the same `layers` to within `token_embedding` —
5,760 parameters, which only the causal arm needs, because only it reads a symbol
back. So a difference between them covers the generation regime and nothing else.

#### What this replaced, and why

The arms used to be two genuinely different architectures. `true` was Perceiver
IO: a latent array, a self-attention stack, and a *second* learned query
(`output_query`) reading the latents back out at the message's length. `false`
was an encoder–decoder: the latents became a cross-attention **memory**, and
`layers` `TransformerDecoder` blocks generated a message-length sequence while
cross-attending into it. They were not the same size at the same depth — a
three-branch decoder block cost about `4·d_model²` more than a two-branch encoder
block — so the ablation ran them at 4 and 5 layers respectively.

That cost the causal arm its first symbol. Its sequence began at SOS, one learned
vector shared by every example, so symbol 0's residual stream carried nothing
about the concept and the referents reached it only through cross-attention
branches scaled by DeepNorm's `beta / alpha` ≈ 0.20. Measured at initialisation
over five seeds, the probability that two different concepts chose *different*
first symbols came out 0.740, 0.323, 0.770, 0.098 and 0.493 — one seed in five
emitting the same first symbol for every concept, against ~0.87 uniformly for
`SenderGRULM`, whose `init_h` builds its hidden state from the prototypes.

Reading the message off the latent array puts the concept in the residual stream
on the Transformer speaker too, where `alpha` amplifies it rather than
attenuating it. See [anecdotes.md](anecdotes.md) for the measurement and what it
did and did not settle.

#### The latent array and `latent_message_multiplier`

`latent_length = round(content_length × latent_message_multiplier)` is the
length of the array the blocks run over. The message is its last
`content_length` slots, so what the multiplier buys is **free slots ahead of the
message**: `first_message_slot = latent_length - content_length`. At the
configured 2.0 that is five free slots and five message slots on ShapeWorld,
eight and eight on CUB.

1.0 is a hard floor and `SenderTransformerLM` raises below it — an array shorter
than the message has nowhere to put the message.

The free slots are load-bearing rather than spare capacity. Nothing ever
overwrites them, so under the causal mask every message slot reads all of them at
every step. That is what cross-attending into a memory used to do, folded into
the self-attention branch — and it is why removing them costs something real: at
1.0 the measured initialisation-time code strength falls from 0.2236 to 0.2059.
Adding more buys almost nothing (0.2254 at 3.0), so this is a floor effect rather
than a capacity one.

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

There used to be an `output_query` here — a second learned query reading the
latent array back out at the message's length — and it was built at multiplier
1.0 as well as above it, so that a sweep varied one thing rather than two and
`state_dict` shapes stayed comparable across sweep points. Taking the tail says
the same thing with one module fewer, and says it identically on both arms. The
cost is that `query` is now the one parameter whose shape moves with the knob:
it is one learned row per latent slot, so a sweep over the multiplier can no
longer share checkpoints.

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
- **An antipodal draw**, one `randn_like` vector for row 0 and its negation for
  row 1. Only `e_pos - e_neg` reaches the attention, so an antipodal pair buys
  twice the readable separation per unit of tag magnitude that two independent
  draws would; near-orthogonality, which is what independent draws in `d_model`
  dimensions actually give you, is a property nothing here wants. The pair is a
  starting point and not a constraint: `dL/de_i` is the gradient at row `i` of
  the sequence and the rows differ in content, so nothing holds `e_neg` at
  `-e_pos` once training starts.

  This replaces a zero init, which opened the rung at the untagged speaker's
  behaviour exactly, in the spirit of `AttentionPrototyper`'s scoring weights.
  What that cost was a traverse: the tag had to climb out of zero at a rate
  bounded by `lr * steps` before the cross-attention could read it at all, and
  the climb sat in exactly the bootstrapping window where the GRU speaker has
  its polarities free — the same shape of bottleneck the logit scale turned out
  to have.

  Rung 10 is what makes that concrete, being the one rung on this ladder that
  both builds this speaker and learns. Its tag went 0.098 → 13.19 from a zero
  init, and `train_acc` moved with it rather than after it: the tag crosses 1.0
  at epoch 4 and accuracy leaves chance in the same epoch, both plateau together
  around epoch 15. Seven epochs of a thirty-epoch run were spent climbing.

  Read the opening against that 13.19 and not against zero. At rung 8's 320-wide
  speaker the antipodal draw opens at `2 * sqrt(320)` = 35.8, so it starts a
  factor of 2.7 above where a learning run settled — an overshoot, but a mild
  one. The 0.16 to 0.79 that rungs 11 and 12 reached is not the comparison: those
  runs never learned, so it is where a dead run leaves the tag rather than where
  the loss puts it.
- **Scaled by what it is added to, not pinned to a number.** `randn_like` is at
  per-element unit variance, which is exactly what `referent_layer_norm` emits
  when it is reset, so the tag opens at the scale of the prototype it marks with
  no constant to choose and none to keep in step with `d_model`. That puts the
  opening separation at `2 * sqrt(d_model)` — about 64 at the configured 1024,
  roughly twice a normed prototype's norm. Loud, but along a *single* random
  direction out of `d_model` rather than broadband, so the cross-attention can
  attenuate that one direction if the loss wants the content back.
- **The name matters.** `gradboard`'s `EXCLUDE_FROM_WEIGHT_DECAY` matches
  `"embedding"` as a substring, so this lands at `weight_decay = 0.0`. A 2-D
  parameter would otherwise be decayed — and decayed *up*, by
  `sqrt(in_features)/sqrt(d_base)` — which is a force on the tag that answers to
  neither the loss nor the scale of what it is added to. Renaming it to anything
  without
  "embedding" in it reintroduces that silently. `polarity_embedding_lr` in
  `[optimiser]` is the other half of the same concern.

#### The causal arm's sampling loop

`decode_autoregressively` is a step-for-step mirror of `SenderGRULM.decode` from
the sampling onwards — same normalisation order, same mask-then-explore order,
same scale-the-unmasked-logits-then-remask discipline, same greedy eval branch,
same per-step accumulation of diagnostics pooled once at the end. All the
reasoning behind those is in [channel.md](channel.md).

What differs is what carries the state. The GRU threads a hidden state through
the loop; this decodes the latent array in place, re-reading the whole array
through the stack at every step.

**Why re-read rather than extend:** broccoli's `MHAttention` asserts
`query_tokens == seq_len` whenever it is causal, so a growing prefix is not
something the module will accept. The loop therefore runs the stack over the
full `latent_length` array every step, reading slot `first_message_slot + i` and
overwriting it with that symbol's embedding before the next pass. The cost is
`content_length` passes over a `latent_length` sequence — five over ten at
ShapeWorld's message length, over a stack a few million parameters wide, which is
not worth a KV cache.

**The slots after the cursor keep their latent vectors** rather than being
blanked. The causal mask makes them unreachable from the slot being read, so what
they hold cannot matter; leaving them alone is simply cheaper than zeroing them.
`test_latent_phase.py` scribbles noise into them at every step and asserts the
embedding does not move, which is the test for the mask as much as for the slots.

**Why each message slot opens concept-derived.** Slot `first_message_slot + i`
holds what `encode` put there until the moment it is read, so the residual stream
that produces symbol `i` *starts* as the referent and DeepNorm's `alpha`
amplifies it. This is structurally what `SenderGRULM.init_h` does. The sequence
used to open at SOS instead — see *What this replaced, and why* above.

The sequence is rebuilt with `torch.stack` at each step rather than written into
in place. In-place would be the obvious way and does not work: the previous
step's forward pass has already saved that tensor for backward, so mutating it
makes autograd refuse. Overwriting an entry of a python list of row tensors,
which is what the loop does, is not the same thing and is fine.

Both speakers feed the *soft* one-hot through the token embedding
(`onehot @ weight` rather than an index lookup) so that the straight-through
gradient reaches the step that produced the symbol. The causal arm's
`token_embedding` is sized at `d_model`, not `token_embedding_size`: the two are
required to be equal for this class anyway, and writing the width the stack
actually consumes keeps the dependency visible.

`embeddings()` is the parallel arm's whole forward pass and is deliberately not
given a causal-arm branch — the causal arm's embeddings are not a function of
the prototypes alone, so there is no honest signature it could have there.

`encode()` returns the latent array **unnormalised** and both arms normalise it
once, immediately, before the stack. The parallel arm used to feed the raw array
straight in and norm afterwards; it cannot now, because the causal arm shares the
sequence between latent vectors and token embeddings, which arrive at their own
scale, and the two arms must stay identical up to the mask.

## `models/transformer_decoder.py`

broccoli has no decoder. `TransformerEncoder` and its `EncoderBlock` are
self-attention and feedforward only, and every cross-attention in this repository
otherwise is a single `MHAttention` sitting *between* stacks — the speaker's
Perceiver IO encode/decode pair, and the bridge the listener used to run from
the message to the referents.

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
the straight-through gradient), masks the referents once, and hands both to two
slots:

```
language_model(messages, referents) -> (batch, slots, output_size)
discriminator(referents, message_repr) -> (batch, n_objects)
```

Four combinations are legal and all four are configurable:

| | `BilinearDiscriminator` | `AttentionDiscriminator` |
|---|---|---|
| **`ReceiverGRULM`** | the historical baseline | new |
| **`ReceiverCrossAttentionLM`** | new | the attention arm |

**Why the split.** One `comparer` key used to choose both halves at once, and the
two comparers divided almost exactly in half along that line — `BilinearGRUComparer`
was a 789,504-parameter GRU plus a 196,608-parameter bilinear form, and
`TransformerCrossAttentionComparer` was two 2.3M decoder stacks. So a rung that
swapped one for the other changed the message encoder *and* the comparison in one
move, and "does attention help compositionality" could not be attributed to
either. The two new cells are what separate *an encoder that reads the candidate
set helps* from *a comparison built on attention helps*.

**Exactly one message encoder, always.** `AttentionDiscriminator` carries an
internal bilinear path, and that is a second *comparison*, not a second encoder:
it reads whatever the language model produced, whichever language model that is.
No key turns on a second encoder, and if one ever looks necessary the slot
contract is wrong rather than the configuration.

### The slot contract

**The language model returns a sequence,** `(batch, slots, width)` always, so
either discriminator can consume either language model. `ReceiverGRULM` returns
its final state as a length-1 sequence; `ReceiverCrossAttentionLM` returns one
position per message slot. `BilinearDiscriminator` means over that axis — the
identity for the GRU, and for a bidirectional stack the honest analogue of "the
last position", which has no meaning there. `AttentionDiscriminator` takes it as
cross-attention memory, where a length-1 memory is legal.

**The signature is uniform:** `language_model(messages, referents)`. The GRU
ignores `referents`, and pays that deliberately — an unused argument is cheaper
than dispatching on class at the call site.

**The discriminator is sized from the language model,** not from a config key.
`build_models` passes `language_model.output_size`, which is `2 * d_model` for a
bidirectional GRU and `d_model` for the decoder stack. No arithmetic makes those
agree, so a key restating one in the other's table could only ever be wrong.

**The slots declare, `Receiver` delivers.** Every swappable module exposes
`referent_input_size` and `message_input_size`; `None`, or the attribute being
absent, means the module does not take that input at all.

| module | `referent_input_size` | `message_input_size` |
| --- | --- | --- |
| `ReceiverGRULM` | `None` — ignores referents | `None` — it *is* the encoder |
| `ReceiverCrossAttentionLM` | its `d_model` | `None` |
| `BilinearDiscriminator` | its `referent_embedding_size` | its `message_width` |
| `AttentionDiscriminator` | its `d_model` | its `d_model` |

For each declared width `Receiver` builds a `model_util.LinearInterface` — the
same class the speaker's `adapter` is — and hands the input
over in a stated distribution:

> - **Referent interfaces:** `dropout(norm(adapter(referents)))`
> - **Message interfaces:** `norm(adapter(message_repr))` — adapter and norm, **no
>   dropout**
>
> Adapters are plain `nn.Linear`. Norms are affine-free `LayerNorm`. Masks are
> drawn independently per referent interface. Referent adapters are sized from
> the backbone's `final_feat_dim` and message adapters from
> `language_model.output_size`.

**This inverts what was here before**, and the paragraph it replaces is worth
keeping in view because its objection was a good one. The arrangement was:
`Receiver` held one adapter of its own and one dropout, each slot owned its own
projection and norm, `BilinearDiscriminator` owned no projection at all and took
the referent width as its own output width, and `AttentionDiscriminator` consumed
the referents **twice at two widths** — its stack at `d_model`, and the raw
tensor handed to the `BilinearDiscriminator` it composes internally. That last
consumer was invisible to `Receiver`, which is what made the arrangement hard to
change at all. The stated reason for not sharing was that `Receiver` would have
to work out which slots want `d_model` and whether to build one at all, which is
reaching into slot internals. Under a declaration it works nothing out: it reads
what the slot states. That is the whole of the change.

**`bias=False` on referent interfaces, bias on message interfaces.** `LN(W(cx)) =
LN(W(x))` holds for a homogeneous `W`; `LN(W(cx) + b)` does not. A bias on the
referent side would break the listener's invariance to the scale its backbone
happens to emit at, which is pinned over seven orders of magnitude by
`test_scores_are_independent_of_the_referent_magnitude` — and not only at
initialisation, which a zero init would cover, but for the whole run, during
which a `BatchNorm` trunk's output scale really does drift. A bias would also
buy little: the norm subtracts the mean, so a uniform one is annihilated
outright and what survives is a constant direction added to every candidate,
which is what `score_bias` already is. No cross-backbone scale claim rides on
the message side, which arrives through the Gumbel channel rather than off a
vision model, so it keeps its bias.

**No dropout on message interfaces.** The message already arrives through the
Gumbel channel, whose noise `uniform_weight` calibrates. A mask on top is a
second and uncalibrated perturbation of a signal that already has one.

The referent mask is element-wise over `(batch, n_objects, features)`, so it
removes features within each candidate rather than removing whole candidates —
which would leak the label ordering.

**One mask per referent interface, drawn independently — and this answers the
prior rejection of per-interface masks rather than dropping it.** The objection
recorded below under [the interface norms](#the-interface-norms) was that two
masks would regularise the listener at a rate no config key names. That is a
correct argument about masking *one* tensor twice, which is not what this is:
each slot reads its own projected copy of the referents, and each copy is masked
exactly once at the rate `[receiver] dropout` names. Independence between the
draws is what makes them the two masks that rate describes rather than a
correlated pair.

*Two consequences, accepted deliberately.* The mask is now **downstream** of each
interface's norm, where every module this has replaced masked upstream of its
own. A LayerNorm after a dropout renormalises the corrupted vector and one before
it does not, so these are genuinely different operations and no earlier numbers
reproduce except at `dropout = 0`. And `Receiver` holds no adapter of its own
upstream of these: two linear maps in series with only a norm between them are
one linear map and a norm, so the separate stage bought only a rank bottleneck
at its output width, and each interface is sized straight from `final_feat_dim`
instead.
`tests/test_receiver_slots.py` still pins the bilinear arm against the pre-split
module at `dropout = 0`, and records the four places it now deliberately
diverges.

### `ReceiverGRULM`

A GRU reads the message and its final state is returned as a length-1 sequence.
`referents` is accepted and ignored: this is an absolute encoding of the message,
with no view of what it is being compared against.

**Default 1 layer, unidirectional, 1024 wide** — jayelm's listener exactly, and
4,687,872 parameters. Parameter parity with the transformer arm is bought at
*that* width, by taking `ReceiverCrossAttentionLM` to 6 blocks on rungs 15 and 16
for 4,702,646, which is +0.3%.

That was +2.1% before the interfaces were hoisted, and the gap was almost
entirely one thing: `ReceiverCrossAttentionLM` owned a `referent_adapter` and
this module had nothing corresponding, so the ladder was comparing an encoder
against an encoder-plus-a-projection. Parity is measured over the **slots
alone**, and the hoist is what makes that the right boundary — counting the
interfaces back in would put the backbone's `final_feat_dim` into a comparison
about message encoders.

Parity used to be sought the other way round, at a shared width of 256, with this
key at 2 and `bidirectional = true`: 1,972,224 against
`ReceiverCrossAttentionLM`'s 2,318,427 at rung 11's widths, where a 1-layer
bidirectional GRU is 789,504. (2 layers bidirectional is 2.5× 1 layer and not 2×
— the second layer's input is the first's concatenated output, so its `weight_ih`
is double.) But nothing in the ladder set both widths, so every rung up to 14
inherited those keys at the defaults' own 1024 and 500 and got a 28,262,400-
parameter listener encoder — 49% of the pair, reading a 7-symbol message from a
14-word vocabulary, at most 26.6 bits — against 2,466,139 on the transformer arm.
The rung 14 → 15 comparison spanned an 11.5× listener. Deepening the arm under
test is the cheaper distortion, because it leaves the baseline listener the
published one.

Parity remains a property of the pair of widths and not of the key: set both
widths together, or set neither. And parameter parity is not interface parity —
`output_size` is 1024 here against 256 on the transformer, so the message
interface and the discriminator downstream of it differ in size between the arms
even at matched encoder parameters.
See the note beside `layers` in DEFAULT.toml.

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

### `BilinearDiscriminator`

A bias-free `nn.Linear` projects the message representation into referent space;
the score is the dot product with each referent, divided by `sqrt(d)`.

**Why no bias in the projection.** With a bias the score expands to
`obj·W·m + obj·b`. That second term is a message-independent *prior* that would
make the model prefer certain objects regardless of what the message said. The
bilinear form is deliberately pure.

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

**`ScoreVolume`** is the listener's one degree of freedom over its own
confidence, and the counterpart of the speaker's `GumbelChannel`. Both are
mixins for the same reason: the scalar stays registered on the module itself, so
`split_out_parameter`'s suffix match and every checkpoint key see it. Both are
one scalar in front of a normalised quantity. The readout is

```
score = score_scale · raw
```

where `raw` is the bilinear form over two layer-normed operands, divided by
`√referent_embedding_size`. BCE is not scale-invariant, so without a volume the
listener could only ever sharpen by aligning its operands — never by committing
harder to an alignment it already has.

**The normalising is on the inputs, not the output.** Both operands arrive at
per-element unit variance, so `|r| = |m| = √d`; `nn.Linear`'s default init has
standard deviation `1/√(3d)`, which puts the raw score at `√(d/3)`; and the
`1/√d` leaves **`1/√3` = 0.577 at every width and under every backbone**. That
is the same width-independence a normaliser downstream would give, arrived at
without one, and it has the property a downstream normaliser cannot have: it
does not divide each game by anything the listener's own performance moves.

One scalar, not one per operand: `c·LN(p)·LN(r)` and `LN(p)·c·LN(r)` are the
same function. `AttentionDiscriminator` builds its composed bilinear path
without either readout scalar for a related reason — that branch is multiplied
by `1 − mix_weight` and read out through the module's own pair, so a scale on it
would say what `mix_logit` already says. Absent rather than frozen, so a
parameter that could not move never matches an elevated learning-rate group.

`log_score_scale` opens at 0, so the readout opens at 0.577. BCE on a random map
at that spread is 0.725 against `ln 2` = 0.693, where unit spread would give
0.804; the gentler opening is why the scalar is left at 1.0 rather than
calibrated to `√3`. Nothing here has a traverse to cover — the speaker's channel
scale once did, opening at 0.839 against a usable channel of 4 to 6, and is a
constant now.

Stored as a log for the usual reasons: `exp` keeps it strictly positive so
gradient descent cannot walk a volume through zero; halving and doubling a gain
should cost the same step; and it gives `train_score_scale` a known ceiling of
`score_scale_lr × steps` log-units per epoch, which is what makes the column
readable rather than merely present.

**The scale is in the backward pass, and that is fine.** A scalar at the front
of the score multiplies every gradient going back through the message, the token
embedding and the Gumbel channel into the speaker, so a listener going quiet
scales down what reaches the speaker. That was read as a self-sealing loop —
BCE's minimiser is `p = 0.5`, so on a message carrying nothing the scale is
*correct* to fall, and falling then keeps the message carrying nothing — and
answered twice, by deleting the scalar and by hiding it from the backward pass
with a straight-through helper.

Neither was necessary. AdamW updates by `m / √v`, and a uniform factor on a
parameter's gradient scales the numerator and the denominator alike, so it
cancels before it becomes a step. `train.py`'s `clip_gradients` is per-submodule
and renormalises each module to `clip_grad_norm` whenever it binds, which at the
ablation's recorded speaker norms of ~10 against a ceiling of 1.0 it does. What
the straight-through helper *did* add was an inconsistent gradient: `∂L/∂s = x`
is the true partial while `∂L/∂x = J` is not, the truth being `s·J`, so the
machinery behind the scale was shaped for a volume of 1 whatever the forward
used. See [anecdotes.md](anecdotes.md), round seven.

**What the threshold does.** `train.py` decides on `lis_scores > 0` against a
score nothing centres, so the threshold is a fixed origin and the listener has
to place its scores against it. `ScoreVolume.score_bias` is what does that: a
signed scalar, opening at zero, applied *after* the volume so that it is an
offset on the score rather than one `score_scale` rescales — a threshold that
slid every time the listener changed how loudly it spoke would be a second thing
to learn. `AttentionDiscriminator.decision` carries no bias because
`score_bias` is already the module's one constant across candidates and a second
would be degenerate with it. `train_acc` is not comparable across the commits on
either side of the centring.

It replaced `mix_bias`, which lived on `AttentionDiscriminator` alone. That left
the twelve rungs on the bilinear arm with no bias anywhere — `bilinear` is built
`bias=False` and the readout was a bare multiply — so the only way for them to
move all candidates together was for the projected message to align with
whatever direction the candidates have in common, which is data-dependent and
spends discriminative capacity in that direction. `mix_bias` also had no config
key, so it sat at the base `lr`; `score_bias` is at `score_bias_lr` = 2e-3 like
every other lone scalar here, and has a metrics column.

Expect it near zero. Games are balanced 10 positive / 10 negative, so the
loss-optimal *global* offset is about zero and staying there means the scores
already sit where the threshold assumes. A scalar cannot correct a *per-game*
offset — the bilinear score's per-game mean is `mean_j(LN(r_j)) · proj`, which
varies by game — so a bias that moves while accuracy does not says the offset
was per-game, and the answer is a different readout rather than a bigger bias.

**And what the weights carry.** Nothing downstream divides a rescaling of
`bilinear.weight` back out, so it carries volume as well as direction and
`bilinear_weight_norm` reads as volume — see
[measurement.md](measurement.md). Sharing the volume with `log_score_scale` is
not the ambiguity two scalars would be: a 320×320 matrix under Adam spends its
step turning and only a fraction of it radially, which is the measurement that
killed the round where the matrix held the volume alone (1.3% of its norm in
thirty epochs, against the scalar's 59%). The scalar is the fast path; the
matrix is not competing for the job. Inside `AttentionDiscriminator` the
branches also mix at their own magnitudes, so there both weights additionally
set what the score is made of.

#### The two readout keys: `scale_score` and `bias_score`

`[receiver_discriminator] scale_score` builds `ScoreVolume.log_score_scale` and
`bias_score` builds `ScoreVolume.score_bias`, on either discriminator. Both
default `true`, which is bit-identical to this section as written. Each removes
one scalar and nothing else.

**Neither reaches the `1/√d`, which is unconditional.** With both operands at
unit variance the raw score's standard deviation is `√(d/3)`, so the division
leaves `1/√3` = 0.577 at every width and under every backbone. That is the whole
reason this repo can state the listener's opening rather than measure it per
rung, and it is a calibration, not a hypothesis — uncalibrated, the score opens
at sd 18.4 and BCE 6.87 at d = 1024, against `ln 2` = 0.693. There is no
configuration that removes it.

**Why two keys.** They answer two questions. `train.py` decides on
`lis_scores > 0`, so the offset places the scores against a fixed origin, while
the volume says how loudly the listener states a conclusion — and
`[optimiser] score_scale_lr` and `score_bias_lr` already moved them at separate
rates. A listener with a threshold and no loudness is a coherent thing to run,
and under the single key these replace it was unreachable.

**They replace `normalise_score`, which had stopped meaning what it said.** That
key gated four things: `BilinearDiscriminator`'s two operand norms, the
calibration, and both scalars. The norms went first, in the interface hoist —
they are `Receiver`'s interface norms now, unconditional, part of delivering an
input at the width a slot declared rather than part of shaping a score, and
there is one referent interface per slot, so the old arrangement (this key
gating `BilinearDiscriminator`'s norms while `AttentionDiscriminator`'s were
deliberately immune to it) had nowhere left to live. The alternative would have
been a `[receiver_discriminator]` key reaching back into `Receiver` to suppress
its interface norms, which reintroduces exactly the config coupling the hoist
removed.

What that left was an off-state of normalised operands under an uncalibrated
score: not `ce7d6a5`, not jayelm's `CopyListener.compare`, and not the design —
a third arrangement nobody chose and no run has used. So the calibration follows
the norms into the design, and the configuration is the two scalars that were
always the only genuinely optional part.

**The history the old key carried, which is still live.** No ShapeWorld run
has learned shape since 17 August.
`4248fca` added all of this on the 19th, its stated purpose to take the score's
volume "out of the backbone's hands", and `60f9094` moved the backbone to
`ResNet18SmallInput` the day before. The two are perfectly confounded — and
since the coupling `4248fca` removed is exactly the one that differed between
Conv4 and ResNet18, they are two descriptions of one intervention rather than
two independent suspects. `score_scale` also decreases on **100% of steps** in
every arm of both silhouette titrations, 0.996 → 0.21–0.28, never once
reversing. The backbone half is under test in
`experiments/silhouette_titration_conv4/` against
`experiments/silhouette_titration_resnet18/`;
`experiments/silhouette_titration_norms/` was the other half, and its ten
score-arm cells no longer parse: the treatment they name is gone.

**The listener half is no longer a single switch, because three of its four
parts turned out not to be optional.** `normalise_score = false` used to return
the listener to `ce7d6a5`'s arithmetic exactly — the last state a ShapeWorld run
demonstrably learned shape from, and what jayelm's `CopyListener.compare` has
always computed, a dot product on raw backbone output. The interface hoist ended
that, and the calibration's removal completes it. What can still be turned off is
the volume and the offset, separately. The speaker-side counterpart
`[sender_language_model] normalise_logits` is a different case and stays one
key: it is not a calibration but a norm over the vocabulary axis on the quantity
that is actually sampled, so it sets the channel's fidelity budget, and
`logit_scale` only means anything because it multiplies a unit-variance
quantity. See [channel.md](channel.md).

The gradient argument above is also weaker than it was. `4248fca` reasoned about
the straight-through Gumbel Jacobian, and the ladder has run
`estimator = "identity"` since `681ef0b`; Hyperion's GPUs report
`is_bf16_supported()`, so `model_util.scale_without_attenuating` is inert by its
own docstring and where a volume scalar sits relative to the backward pass no
longer changes what the optimiser sees.

**What they do not touch.** `AttentionDiscriminator`'s input and memory norms
stay under every setting, as they always have — but they are `Receiver`'s
interface norms now, so this is no longer an exemption written into a key. The
reason is unchanged: a post-norm stack normalises its own stream but never its
memory. `mix_floor`, `mix_logit` and `mix_logit_init` are untouched, so with both
keys off that module returns `(1 − a)·bilinear + a·attention` unaltered — still
calibrated, because the bilinear path it composes keeps the `1/√d`. On that
module the keys act on the outer readout, downstream of the mix; the composed
path carries neither scalar whatever they say, a volume on it being degenerate
with `mix_logit` and a constant on it with the outer offset.

`score_scale_lr` and `score_bias_lr` stay live and simply have no effect when
their own key is off; `train_score_scale`, `train_score_bias` and
`train_clip_log_score_scale` read NaN, each independently of the other. With the
volume gone `bilinear_weight_norm` is the only volume column left, and with
nothing downstream of the matrix it reads as the listener's whole volume rather
than as the fast scalar's slow partner. Checkpoints do not cross either key: the
scalar it builds is absent from the `state_dict` when it is off.

**Dropout masks the referents only,** and lives on `Receiver`'s referent
interfaces. It used to mask the message operand too, on the argument that a dot
product lets the listener lean on whichever side is left intact. True, but it
assumed the two sides arrive on equal terms and they do not: the message comes
through the Gumbel channel, whose noise is already calibrated by `logit_scale`
and `uniform_weight`, so a mask on top is a second perturbation of a signal that
has one — and the listener cannot tell which of the two it is being asked to be
robust to. The referents arrive clean. This is why message interfaces are adapter
and norm with no dropout, where referent interfaces carry all three.


### `ReceiverCrossAttentionLM` and `AttentionDiscriminator`

Two `TransformerDecoder` stacks, one in each slot, each reading the other's
stream as memory:

1. **`ReceiverCrossAttentionLM.message_decoder`** — `layers` blocks of
   self-attention, cross-attention into the candidate set, then a feedforward.
2. **`AttentionDiscriminator.referent_decoder`** — `layers` blocks of
   cross-attention into the encoded message, then self-attention across the
   candidates, then a feedforward. `cross_first`, so the message comes before
   the candidates compare each other.

Then a plain linear readout scores each one, and the mix below combines that
score with a bilinear one over the same encoding.

`AttentionDiscriminator` declares `message_input_size = d_model`, and `Receiver`
builds it a message interface — an `nn.Linear` from the language model's
`output_size` to that width, followed by a non-affine `LayerNorm`. The
projection is what makes the slot swappable at all, since no arithmetic makes a
bidirectional GRU's `2 * d_model` agree with this stack's width. The norm is
there because a post-norm stack normalises its own stream and never its memory,
and `message_decoder`'s last post-norm used to make that safe by accident where a
GRU state would not.

It owned that adapter and that norm itself, as `memory_adapter` and
`memory_layer_norm`, until the interfaces were hoisted. The same tensors, one
stage upstream — with the consequence that this module is now **the same size on
rungs 13 and 15**, where it used to differ by a `memory_adapter` reading a
1024-wide GRU state against a 256-wide encoded message. The difference has moved
into `Receiver.interfaces`.

**Why two stacks rather than four bare stages.** The structure this replaces
crossed the message into the referent stream exactly once, at a single
cross-attention. Everything common across candidates cancels at the readout, so
the only thing that could separate two of them was the difference between their
attention weights over the message — a small perturbation about a near-flat
softmax at initialisation, where a bilinear `obj·W·m` is first
order and differs per candidate from step zero. Rungs 11 to 14 sat at 0.5000 for
thirty epochs while rung 10, which is rung 12 with the bilinear comparer and
nothing else changed, learned. `comparer_probe.py` shows the old module solving
a fixed noise-free protocol in under 200 steps, so what failed was not the
comparer's capacity but its ability to bootstrap against a speaker that had not
learned yet. `M` crossings instead of one is the response.

Measured at initialisation, as the standard deviation of the change in scores
when the message is replaced with noise, over the standard deviation of the
scores themselves — how much of what separates the candidates comes from what
was said. Mean of five seeds on a 16 × 20 game with correlated referents, both
modules untrained:

| | message share of score sd |
|---|---|
| four stages, 320 wide, 4 encoder layers | 0.299 |
| two stacks, 256 wide, 3 + 3 blocks | 0.450 |

Depth alone does not move it — 1, 2, 3, 4 and 6 blocks a side all land between
0.45 and 0.52 — because DeepNorm damps each branch harder as the stack it is on
gets deeper, and the extra crossings buy back roughly what the damping costs.
The gain above is the structure, not the depth. Pinning `alpha = beta = 1.0`
reaches 0.75 at three blocks, which is the knob if this ever needs to go
further; it is not the default because the pinning gives up what DeepNorm is
for, and because five seeds do not order the depths under it.

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

**Each stack's depth is the `layers` key of its own config table.** A single key
was once a total split between two stacks, which meant asking for one more block
moved two; separate tables make that unstateable rather than merely untested.
Each stack also resolves its DeepNorm constants from its own count and with
`decoder=True`, since a `DecoderBlock` has three residual branches rather than
two: at three blocks that is `alpha = 1.732`, `beta = 0.408`.

**Stochastic depth is suppressed below two layers,** asked of each stack
separately. `depthwise_linear_stochastic_depth` spreads the rate linearly across
layers, so a one-block stack would get a single rate of 0.0 regardless.

**The referent stack is never causal, and that is not negotiable.** In this
codebase referent *order is the label vector*: `data.util.split_spk_lis` writes
positives into the first half of each agent's view and negatives into the
second, and the augmentation permutes only *within* each half. Anything that
could index its own sequence axis could learn "the first half are targets" and
score perfectly while ignoring the message. `DecoderBlock` defaults to
`causal=True` because its other caller is a speaker generating a sequence, so
this stack passes `causal=False` explicitly and takes no positional embedding of
any kind; both are asserted in
`tests/test_cross_attention_comparer.py`. With neither, it is
permutation-equivariant and cannot read the ordering at all.
`BilinearDiscriminator` is immune for a different reason: it scores each referent
in isolation and never sees the set.

**Each referent interface's adapter has `bias=False`, and that is load-bearing
rather than tidy.** The norm after it is what makes the score independent of the
size the vision model happens to emit, and it can only do that exactly if what
reaches it is homogeneous in the input: `W(cx) = cW(x)` gives `LN(W(cx)) =
LN(W(x))`, where `W(cx) + b` does not. With a bias, a backbone emitting features
a hundred times smaller gets a score shaped partly by this layer's bias and one
emitting large features does not — a weaker form of exactly the defect being
removed, and one that would leave the invariance test asserting an
approximation. The following norm subtracts the mean anyway, so most of a bias
here would be annihilated a line later.

The norm that follows is what makes this a claim about the *whole run* rather
than about initialisation. It is `model_util.LinearInterface`'s, unconditional,
on both agents: the speaker's adapter used to end in an affine `RMSNorm` inside
a `FeedforwardBlock`, and when that block became a plain `nn.Linear` the norm
stayed and the learnable gain went. A gain is a route to global magnitude, and
neither agent should be choosing one at a width change. The message interfaces
carry a bias, and the asymmetry is deliberate: no cross-backbone scale claim
rides on the message side.

**The interface norms are parameter-free, and not for the reason originally
given.** broccoli's `project_qkv` RMS-normalises Q and K per head, so the
attention *logits* are already free of the vision model's scale, and
`MHAttention.out_norm` handles a uniformly louder backbone (measured: the whole
set at 10× moves the output by 0.0%). What neither handles is *per-object*
magnitude. In the message stack's cross-attention the referents are the values;
V is not
normed anywhere, so the attention output is a magnitude-weighted mixture: one
candidate 50× larger than its neighbours moves that output by 116% without this
norm and by 0.0% with it, and no downstream norm can undo it because the
averaging has already happened. That is an object winning for being large rather
than for matching. An affine here would also be a route to *global* score
magnitude.

<a id="the-interface-norms"></a>
**The dropout is `Receiver`'s, and it is back downstream of the norms** — one
mask per referent interface, drawn independently, at the end of the interface.
That is the placement the original argument wanted: a mask upstream of a learned
projection is a mask the projection can average away, and a mask upstream of a
LayerNorm has its `1/(1−p)` rescale thrown away and its survivors renormalised
*up*, so the perturbation is neither the size nor the shape the knob names.

It was moved upstream to buy one thing — one mask reaching both slots
identically — on the argument that a mask inside each slot would regularise the
two-adapter combinations twice at a rate no key names. **That argument is
answered rather than dropped.** It is correct about masking one tensor twice.
Under the interface contract each slot reads its own projected copy of the
referents and each copy is masked exactly once at the rate `[receiver] dropout`
names, so two independent masks on two tensors is the rate the key describes and
not double the rate. What is given up is that the two slots no longer see the
same masked referents, which was itself only ever a means to the rate. The
consequence is stated under the slot contract above. Only the referents are
masked; attention dropout is a separate setting
(`receiver_discriminator.cross_attention_dropout`).

**There is no separate norm before the readout, and there used to be.** The
argument for `decision_layer_norm` was that it equalised the candidates' lengths
— otherwise `scores` is `|refined_j| · cos(θ_j)` and an object can be read loudly
for being large rather than for matching, the same defect as the referent-norm
case one stage later. The referent stack's last block ends in a post-norm, which
is `nn.RMSNorm(d_model)` and normalises per position, so the candidates already
reach the readout at equal length and that argument is answered structurally.

What the extra norm also did was sit between the post-norm's learnable gain and
global score volume. Nothing closes that route now, deliberately — a branch is
allowed to be loud or quiet, and `mix_share` against `mix_alpha` is what reads
it. See [anecdotes.md](anecdotes.md) for the attempts to close it here, and what
each of them cost.

**The readout is a plain `nn.Linear(d_model, 1)` with no bias.** The bias has
tracked whether anything downstream subtracts a mean; nothing does now, and it
stays off for a different reason — `ScoreVolume.score_bias` is already the
module's one constant across candidates, so a second one would be degenerate
with it and the pair would be free to drift against each other. Measured: a bias
of `b` on `decision` moves the score by `score_scale · mix_weight · b`, the same
constant for every candidate in the game, which is precisely what `score_bias`
expresses directly. `decision_spread` and
`decision_kurtosis` read the magnitude and the shape of what comes out — see
[measurement.md](measurement.md).

### The mix, and why the attention arm opens as the bilinear one

`AttentionDiscriminator` does not return that readout. It returns

```
score = score_scale · ( (1 − a) · bilinear + a · attention ) + bias
a     = mix_floor + (1 − mix_floor) · sigmoid(mix_logit)
```

so the volume — and the offset — are the same `ScoreVolume` the bilinear arm
carries.

**Neither branch is standardised,** and that is a choice with a cost on each
side. Standardising per branch would make `a` mean *composition* exactly, and
would close the escape of turning an uninformative branch down rather than
making it informative. Leaving them alone keeps the single volume knob without
pinning the branches to equal spread, which reopens that escape — so `mix_share`
is reported beside `mix_alpha` to watch for it: the first is the share the score
is actually made of, the second the share `mix_logit` asked for, and they come
apart exactly when a branch is loud or quiet rather than useful.

This module's opening is therefore not the bilinear arm's calibrated `1/√3`:
the attention branch arrives at whatever magnitude `decision` gives it, and the
mix is a weighted sum. That is a fixed number per architecture rather than a
moving one — measure it with a forward pass if a rung needs its openings
matched.

**Why.** The attention path alone does not bootstrap. Under a nuisance level
where the bilinear comparison reaches 0.938, the two decoder stacks reach 0.469
with the speaker's polarity tag barely moving — and the cause is not the
listener. Handed a message that names the concept, the same module reaches 0.988
and holds its between-candidate share at 0.90; handed a scrambled one it
collapses to 0.40. Uniformity is *correct behaviour* when there is no pattern,
and at initialisation nothing in the pair is a pattern yet. So the pair needs
something that already works at step zero, and the attention path can take over
if it earns it. That is the recipe `AttentionPrototyper` already follows: open at
pooling that *is* the mean, and depart only if it pays.

At `a = mix_floor` the discriminator is essentially the bilinear comparison,
which is the configuration measured bootstrapping. `mix_logit_init = −4.0`
against a floor of 0.1 opens it at 0.116.

**The floor is in the parameterisation and must never become a `clamp`.**
`clamp`'s gradient is zero below its bound, so a weight that drifted under the
floor would weld there permanently and the attention stack could never come back.
That bug cost an afternoon in the prototype. What the floor buys is that the
attention path always contributes and so always receives gradient — at `a = 0`
the whole stack would get nothing and could never earn its way in.

**The pair can go quiet, and that is deliberate.** `score_scale` is downstream
of the mix, unbounded and log-parameterised, so a listener with nothing to say
can say it quietly. That is the whole of why this is not the fixed-gain readout
coming back: that one closed the collapse exactly as designed and stopped four
rungs learning at all, because a pair forced to commit through a fixed volume
from step zero commits before the message carries anything.

What made going quiet dangerous — that it turned the speaker down at the same
time — is gone rather than prevented; see `ScoreVolume` above.

**Three columns come out of it,** and they have to be read together.
`mix_alpha` is how much of the score `mix_logit` asks the attention path for,
which is the chapter's question stated as a number. `mix_share` is how much of
it the attention path actually supplies. `path_agreement` is the within-game
correlation between the two standardised paths, and it is necessary because an
attention path that is never used and one that has learned to imitate the
bilinear path look identical from accuracy and from `mix_alpha` alone. See
[measurement.md](measurement.md) for how to read the combinations.

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
pooling_kernel_size   = largest even number ≤ (max_side / 32) × 3
pooling_kernel_stride = kernel_size
pooling_padding       = enough to cover the image, split symmetrically
```

That is 6px patches on an 11×11 grid at ShapeWorld's 64px, and 20px patches on a
12×12 grid at CUB's 224px.

**The two datasets no longer run the same ViT.** `[sender_feature_model]` is
ShapeWorld's — 128 wide, 6 layers, 4 heads, `ff_inner_size` 256, GELU, 876,599
parameters — and `[birds.sender_feature_model]` pins CUB's, which is the 320 /
10 / 5 / 576 SwiGLU stack both used to share, at 11,332,626. Each is matched to
its own baseline backbone rather than to the other dataset's: `ResNet56` at
852,368 on ShapeWorld and `ResNet18` at 11,176,512 on CUB. See
[the CIFAR ResNet](#resnet56-the-cifar-resnet) below for why the ShapeWorld
backbone shrank.

The activation differs for the same reason the width does. SwiGLU's `linear_in`
is double width because it produces the gate alongside the value, so a block's
feedforward costs `3·d·f` where GELU's costs `2·d·f`; at 128 / 6 / 256 that is
1,107,774 against 876,599, and only the second is within 3% of the CNN.

**The tiling does not overlap, and used to.** The old rule ran stride at half the
kernel with a matching pad, which put both datasets on a 17×17 grid of 289
tokens. Because `pooling_type` is `"concat"` the tokenizer is a space-to-depth,
so at stride = kernel it is an exact tiling and every pixel still reaches the
transformer exactly once — the overlap was duplicating each pixel four times
rather than adding information. What it bought was a locality prior and a finer
positional grid; what it cost was 289 tokens against 121.

On an A100 at 640 images of 64px, fwd+bwd in bf16 and compiled, that is 303ms
against 118ms, where the `ResNet18SmallInput` these backbones were compared
against at the time ran in 81ms. The ViT was 3.75× the baseline's wall clock and
is now 1.46×. Those numbers were measured at the shared 320-wide stack and rank
the *geometries*, which is what they are for; ShapeWorld's ViT is much smaller
now and the ResNet it is compared against smaller again.
`scripts/vit_geometry_sweep.py` is the harness and can re-derive them.

Stride appears in no weight shape, so a geometry change moves ShapeWorld's
parameter count not at all — which matters, because the fairness claim the
ablation rests on is stated in parameters. CUB's does move, since a 20px patch is
1,200 values against a 28px one's 2,352 and above `d_model` that difference is
carried by `ResizeAndPadPatches`. It moves the right way: 101% of `ResNet18`
where the old geometry was 113%.

The padding is what makes the tiling cover the image. Without it the final
partial patch is silently cropped, which is a strip of the image the model cannot
see.

`image_classes` is `d_model`, so what broccoli calls the logits is this
backbone's output *embedding*. It is left unnormalised — `SequencePool` into a
plain `Linear` — which is the intended state, not an oversight. Whichever
consumer needs the referent at a controlled magnitude normalises it where the
score is formed. See [broccoli.md](broccoli.md) for `batch_norm_logits=False` and
the rest of the pinned arguments.

### `ResNet56`: the CIFAR ResNet

ShapeWorld's backbone on both agents, and the third one this repository has had:
`Conv4` at 113,088 parameters, then `ResNet18SmallInput` at 11,168,832, and now
`CifarResNet(9)` at **852,368**.

It is He et al. 2015 §4.2 rather than a modified ImageNet ResNet: a 3×3 stride-1
convolution at 16 channels, three stages of nine blocks at 16 / 32 / 64 channels,
stride 2 at the first block of the second and third stages, a global average pool
and nothing else. `6n + 2` weighted layers at n = 9 is 56, of which 55 are
convolutions here — the 56th is the classifier this repository does not build.
`final_feat_dim` is 64.

**Why it replaced an 11.2M network.** The reason is the learning rate, not the
accuracy. jayelm maps ShapeWorld to `Conv4` and his `--lr` default is 1e-4, so
the rate was tuned at 113,088 parameters; this repository restored 1e-4 on
2026-09-05 as "jayelm's own rate" while running `ResNet18SmallInput` at
11,168,832 from random init. His CUB runs are not a counterexample —
`run_cub.sh` passes `--pretrained_feat_model` on every one, so 1e-4 there is
fine-tuning from ImageNet weights rather than training 11.2M from scratch. The
rate had never been measured at the size the ladder runs, and 99× is too far to
assume it travels. `experiments/baseline_lr_sweeps/` measures it; this backbone
is the other half, bringing the scale back to somewhere the inherited rate is at
least arguable.

It also closes a hole. The backbone stable ran 113k and then 11.2M with nothing
between, so "the rate is wrong for this scale" and "the rate is wrong for this
architecture" could not be separated — and the `Conv4`/`ViT2` comparison the
ablation is built on was 0.11M against 10.3M, a 91× mismatch on rungs whose whole
claim is that only the architecture moved. At ~0.85M the CNN and the ViT are
within 3% of each other.

**Option A shortcuts, which is what "faithful" means here.** Where the widening
shortcut in `SimpleBlock` is a 1×1 convolution with a BatchNorm after it —
option B — `CifarBlock` subsamples the spatial axes by taking every other pixel
and zero-pads the channel axis, split evenly. That carries no parameters at all,
which is exactly why He et al. chose it for CIFAR: the residual network then has
the same parameter count as the plain network it is being compared against, so
the comparison is about the shortcut and not about capacity. It is also why this
is a sibling class rather than a channel list passed to `ResNet` —
`tests/test_backbones.py` pins that the shortcut holds nothing.

**Resolution, which was the small-input stem's argument and is still the point.**
The stock ImageNet stem discards 4× before any residual block runs — 7×7 stride 2
then a 3×3 stride-2 maxpool — which on ShapeWorld's 64px images leaves a 2×2 map
for the adaptive pool to average over. What survives that is colour, and what
does not is shape, which is precisely the wrong bias for a study whose known
failure mode is the speaker learning to name colours. `ResNet18SmallInput`
reached 8×8 by replacing that stem with a 3×3 stride-1 convolution and no
pooling, as SimCLR does for CIFAR-10 (Chen et al. 2020, arXiv:2002.05709).
`ResNet56` reaches **16×16**, because it downsamples twice in total rather than
four times.

`CifarResNet.reset_parameters` recurses over `self.modules()` for the reason
`ResNet.reset_parameters` does — see [anecdotes.md](anecdotes.md) — and here that
is 54 convolutions of 55 that a `self.trunk` walk would miss.

**`ResNet18SmallInput` is gone**, and `ResNet18` is untouched: it remains CUB's
backbone on both agents and stays pinned tensor-for-tensor against
`torchvision.models.resnet18`. Configs naming the deleted class — the
`silhouette_titration_resnet18` arm and one arm of `conv4_silhouette_asymmetric`
— now raise at construction, which is the intended loud failure; they are records
of runs against a network this repository no longer has.

**The two datasets are no longer parameter-matched to each other**: 0.85M on
ShapeWorld against 11.2M on CUB, on both agents. That is deliberate — CUB's 224px
photographs do not want a CIFAR network — and what each dataset preserves is the
match *within* it, which is the comparison a rung actually makes. A difference
between an odd rung and the even rung beside it covers two different networks at
a 13× size ratio as well as the dataset, and is not an architecture result.

### `ResNet` and the ImageNet stem

`ResNet18` is CUB's backbone on both agents and the only user of the `ResNet`
class since `ResNet18SmallInput` was removed. Its `small_input_stem` flag is
still there and nothing selects it: the stem swap it performs is what
`ResNet56` supersedes on ShapeWorld, and the flag is kept because a factory
naming it is a two-line addition if a 224px small-stem arm is ever wanted.

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

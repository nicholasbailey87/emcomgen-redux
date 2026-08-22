# Anecdotes

Findings and failures, with the numbers. Several current design choices only make
sense as the survivors of something that did not work, and this is where those
are recorded.

## The listener readout: two attempts and a revert

The longest story in the codebase. It concerns
`TransformerCrossAttentionComparer.decision`, currently a bare
`nn.Linear(d_model, 1)` — which is exactly where it started.

### The problem

A bare linear readout makes one vector both the *direction* the head reads out
and the *volume* it reads out at. BCE will always reduce a loss it cannot
otherwise reduce by becoming less confident, and that pressure is first-order
where learning a useful direction is not, so the volume collapses first.

On CUB it did exactly that: scores fell from sd 0.42 to sd 0.016 inside one epoch
and stayed there for thirty, with `train_loss` pinned at `ln 2 + 2e-5`.

The reasoning against that was: every gradient reaching
`referent_self_attention`, both cross-attentions, `encoding`, both adapters, both
vision models and the entire speaker is proportional to the readout's magnitude,
so a quiet listener starves the machinery that would make it informative. **That
reasoning is sound and the conclusion drawn from it was wrong.**

### Attempt one — normalise the direction, learn the volume

Normalise `decision.weight` to a unit vector and move the volume into a single
learnable scalar, `log_score_scale`.

It made the collapse *legible* — one column, with a known ceiling of
`score_scale_lr × steps` log-units an epoch — but it did not remove the pressure.
`issue.csv` is that round: rung 12 at 30 epochs with `train_loss` pinned at
`ln 2`, `train_acc` at 0.4998 and `score_scale` sliding 0.914 → 0.273, monotone,
sign-consistent, never recovering. Rung 11 did the same.

The accuracy column could not see it either: `train.py` reads the decision as
`lis_scores > 0`, and a strictly positive scale leaves `s·(u + b) > 0` equivalent
to `u + b > 0`.

### Attempt two — remove the volume as a parameter

`decision` called directly, its output standardised by a
`BatchNorm1d(1, affine=False)` over the flattened batch, the result multiplied by
a fixed `decision_gain`.

That **closed the collapse as designed**. Scaling `decision.weight` by `c` scaled
the pre-norm logits, their mean and their standard deviation alike, so the
quotient did not move; a constant readout came out at 0 and sigmoid 0.5; a
shrinking spread was renormalised straight back out.

It also **stopped the run learning**. `receiver-cross-attention-birds.csv` and
`receiver-cross-attention-shapeworld.csv` held `train_acc` at 0.5000 to four
places for all thirty epochs, and rungs 11, 12, 13 and 14 — every rung with this
comparer — failed together.

### Why removing the volume was the wrong lesson

The premise was that the `log_score_scale` collapse is fatal. **It is not.**

Rung 10 is this rung's own control — the same speaker, channel, optimiser and
receiver ViT, differing by five config lines that swap in `BilinearGRUComparer` —
and its `score_scale` falls monotonically 0.856 → 0.238 across its whole run
while `train_acc` climbs 0.4975 → 0.6351. It does the collapse and learns anyway.

`diagnostics/bootstrap_probe.py` isolated the difference: the whole pair, real
speaker, real Gumbel channel, real comparer, with only the vision models replaced
by frozen prototypes so it runs on a laptop. At 2500 steps and the config's own
1e-4:

```
rung 10, bilinear         acc 1.000   polarity_separation 9.50
rung 12, standardised     acc 0.606   polarity_separation 3.64
rung 12, plain readout    acc 0.863   polarity_separation 8.17
```

The third row is the module with the norm stripped to an identity, the gain at
1.0 and `decision`'s bias restored — i.e. what is written today. It takes off at
step ~1600 by the same route rung 10 takes: `polarity_separation` crosses 6–8,
the speaker's logit scale traverses, `realised_survival` jumps 0.22 → 0.82,
accuracy follows.

**So a readout the listener cannot turn down is a readout it must commit through
from step zero, before the message carries anything. What that costs the speaker
is larger than what the collapse costs it.** The volume is a parameter again,
deliberately, and the collapse route is open again with it — watched by
`decision_spread` rather than closed.

`BilinearGRUComparer` keeps its `log_score_scale`, as it always has. It is the
ablation's baseline listener and the control that produced the evidence above,
and it works.

### What is left of attempt two: `decision_kurtosis`

The standardised readout was measured to escape through the *fourth* moment,
which is where the column came from. BCE against a coin-flip label costs
`|s|/2 + ln(1 + exp(−|s|))` — quadratic near zero, linear far out — while
variance costs `s²`, so under a pinned variance a few enormous scores absorb the
budget cheaply and the bulk sits at sigmoid 0.5. That specific arbitrage needed
the pinned variance and is gone with it.

The column is not, because what it reads is more general. Driven against this
module on a synthetic game, an informative message gave −2.0 at 100% accuracy and
a scrambled one +11..+23 at chance, while `decision_spread` overlapped between
the two (2.7–5.1 against 1.4–2.1) and could not tell them apart.

Also left behind: `score_scale_lr` is now gated on `BilinearGRUComparer` rather
than applying to both comparers. That gate is an artefact of the parameter's
absence, not an argument against the parameter.

## Frozen logit spread: NaN through a masked gradient

`logit_spread` bit-identical across epochs is the signature of the AMP
`GradScaler` skipping every step.

Cause: scaling the *already-masked* logits by `logit_scale`.
`d(logits · scale)/d(scale)` is the logits themselves, so the `-inf` at the
reserved slots enters the gradient w.r.t. the scale; the upstream gradient at
those slots is zero, and `-inf × 0` is NaN. `GradScaler` reads that as an
overflow and skips the step — every step, so the whole pair sits frozen at
initialisation. The loss just idles, so nothing else shows it.

Fixed by scaling the *unmasked* logits and re-masking. Harmless while the scale
was a constant, since there was no gradient path to it at all. `fccba0f` fixed
the same class of failure for a differentiable `sampling_tau`, which divides the
`-inf` logits and puts `inf` into the gradient the same way.

## The birds run that flattened itself for 35 epochs

A birds run started at 0.62 retained entropy (the old `ln V` scheme's 0.66
coefficient), then annealed *upwards* for 35 epochs to about 0.94 retained,
before accuracy left chance on the way back down at around 0.82–0.85.

Read as a policy that is annealing rather than one that is stuck, the descent is
a cost: the run spent 35 epochs travelling to an entropy it could have been
started at. `init_energy` now defaults to 0.9 — near where it chose to go, short
of the 0.94 extreme where messages may carry too little for the listener to learn
from. A design decision from a single run, not a derived bound.

## A speaker whose sharpness was normalised away

With a constant `logit_scale`, the birds speaker spent 55 epochs growing
`logit_spread` from 0.41 to 1.62, saw every bit of it normalised away by
`layer_norm_logits`, and held `realised_survival` at 0.18 with train accuracy at
chance for the whole span. This is why sharpness had to become a *post-norm*
parameter (`87c1027`).

## The LayerNorm epsilon runaway

A fresh GRU speaker emits pre-norm logits at sd ~0.24. At `F.layer_norm`'s 1e-5
default the normaliser starts giving out below ~0.01 — a margin of roughly 24×,
*not* the three orders of magnitude an earlier version of this note claimed.

Shrinking that speaker's output layer 1000× drops realised survival from 0.43 to
0.09; a channel that noisy then starves the gradient that would restore the
logits, so it runs away. Observed on a birds run whose `realised_survival` fell
0.47 → 0.17 over 22 epochs.

At `eps = 1e-12` the same 1000× collapse leaves survival at 0.43, unchanged to
four decimal places, and the normaliser holds down to sd ~1e-6.

## `reset_parameters` that reset almost nothing

**`ResNet`.** It used to walk `self.trunk` and call `reset_parameters()` on
anything that had one. `SimpleBlock` defines no such method, so the eight
residual blocks — 11.1M of the 11.18M parameters — were skipped entirely, and the
two layers that *were* reached (the stem conv and BN) got PyTorch's defaults,
which for `Conv2d` is kaiming *uniform* rather than the fan-out normal
`init_layer` applies at construction. **One tensor of sixty was reset, with the
wrong distribution.** It now recurses over `self.modules()` and goes through
`init_layer`, and resets BatchNorm running statistics too.

**`ConvBlock`.** Going straight to `init_layer` skipped the conv biases, which it
does not touch, and the BatchNorm running statistics, which are buffers rather
than parameters — both were carried across a reset. It now reproduces
construction exactly: PyTorch's own initialisation, then `init_layer` overriding
the weights.

**`ViT2`.** It had no `reset_parameters` at all. That made
`Receiver.reset_parameters` raise `AttributeError` for any rung using a ViT
listener, while `Sender.reset_parameters` — which guarded on `hasattr` — silently
left the whole speaker backbone untouched. The method now exists and the guard is
gone.

**`TransformerCrossAttentionComparer`.** The two adapters were missing from its
reset, so a reset listener kept the projections that map referents and messages
into `d_model` — most of what it had learned about its inputs — while everything
downstream of them was re-drawn.

Parameter-free norms are listed in every `reset_parameters` anyway, for the
mirror reason: turning `elementwise_affine` back on must not leave a reset
listener holding trained gains.

## The polarity tag: a speaker that could not read its own input order

`SenderTransformerLM`'s encoder cross-attention carries no positional or rotary
embedding on its key side, so its output is a weighted *sum* over the two
prototypes and is **bit-identical under swapping them**. The ordering was in the
tensor; no parameter could condition on it.

What survived the symmetry was a content cue — positives are a tight cluster and
negatives a diverse one, so the negative prototype sits nearer the global mean
with a smaller norm — and `referent_layer_norm` divides that norm difference out
before the attention ever sees it. Only direction was left, and at initialisation
not even that: an untrained backbone makes both prototypes the mean of noise.

So the cost was heaviest exactly during bootstrapping, where this speaker started
with zero polarity information while the GRU had it for free (`init_h` reads the
concatenation, so each polarity gets its own weight columns).

The `bootstrap_probe` numbers above show the tag doing its job:
`polarity_separation` crossing 6–8 is what precedes takeoff.

## Straight-through gradient collapse at fixed tau

Measured over unit-variance logits at V = 20: with `tau` held constant while the
speaker sharpens, the effective number of tokens carrying gradient falls from 4.9
at the opening scale to 1.6 by a scale of 6, with the winner holding 0.80 of the
mass. Scaling `tau` with the scale gives 9.0–9.5 effective tokens across the same
range.

And in the other direction, at scale 0.35: coupling *below* the opening scale
takes the surrogate from 4.9 effective tokens to 2.0 while the token it favours
matches the noiseless argmax only 9% of the time — a confident gradient pointing
at noise. Hence the floor at ratio 1.

## `ln(V)` was the wrong correction

The old closed form was `coefficient × ln(V)`, on the argument that a winner must
beat the largest of `V` Gumbel draws and `E[max_i g_i] = ln V + γ`. Right for
holding a *survival rate* constant; badly wrong for holding *entropy* constant.

Over V = 8..256, the scale that holds entropy fixed varies only 1.2–1.4× while
`scale / ln(V)` varies about 2× — so dividing by `ln(V)` introduces roughly four
times more vocabulary dependence than it removes. The residual is logarithmic
with a much smaller coefficient: at 80% retained,
`scale ≈ 0.87 + 0.12·ln(V)` fits to about 2% over that range. Replaced by a
numerical bisection.

## Gradient clipping dominated by one module

At init the listener's comparer holds ~90% of the pair's squared gradient norm,
against ~0% for the speaker's vision model — whose gradient reaches it through
the whole language model and the straight-through Gumbel sample. A single global
`clip_grad_norm_` therefore hands every module a coefficient set by the comparer,
carrying its batch-to-batch fluctuation into the weakest gradients in the pair.
Clipping is now per submodule.

## Two floors that were tried and removed

`e3fcabd` put a floor under the listener's score scale. It cost fifteen epochs.
There is deliberately no floor on either agent's scale now; `29b18ea` measured
the separation that distinguishes a productive dip from a collapse instead — a
healthy speaker fell ~0.2 log-units and returned within a few epochs, where the
arm that died fell 0.94 and never did.

## The stage-3 residual carries referent identity

Without the residual around `referent_cross_attention`, a candidate reaches the
score only through near-uniform attention weights. Measured at init, that stage
halved the between-object share of the variance: 0.415 going in, 0.221 coming
out.

## Per-object magnitude, and what normalisation does not fix

broccoli's `project_qkv` RMS-normalises Q and K per head, and
`MHAttention.out_norm` handles a uniformly louder backbone — measured, the whole
referent set at 10× moves the output by 0.0%. What neither handles is *per-object*
magnitude: V is not normed anywhere, so at `message_cross_attention` one candidate
50× larger than its neighbours moves the attention output by **116%** without
`referent_layer_norm` and by 0.0% with it. No downstream norm can undo it, because
the averaging has already happened.

## The ResNet stem discards shape

The ImageNet stem discards 4× resolution before any residual block runs. On
ShapeWorld's 64px that leaves 16×16 into stage 1 and a 2×2 map at the end, which
the adaptive pool averages to a single position. What survives is colour; what
does not is shape — precisely the wrong bias for a study whose known failure mode
is the speaker learning to name colours. SimCLR's small-image stem leaves an 8×8
final map, so the pool has 64 positions rather than 4.

## Parameter costs, for the record

- Absolute position embeddings: ~190k parameters a rung, most of it two
  289-position tables in the `ViT2` backbones.
- A decoder block costs about `4·d_model²` more than an encoder block:
  1,354,951 against 944,711 at 320 wide with `ff_inner_size = 554`. The ablation
  runs the decoder arm at 4 layers and the latent arm at 5 to put both within 2%
  of the GRU baseline.
- The small ResNet stem: 1,728 parameters against 9,408.

# Anecdotes

Findings and failures, with the numbers. Several current design choices only make
sense as the survivors of something that did not work, and this is where those
are recorded.

## The listener readout: six attempts, and what each one was actually about

The longest story in the codebase. It concerns the attention listener's
`decision`, a bare `nn.Linear(d_model, 1)` — which is exactly where it started.

Written before the listener was split into `ReceiverCrossAttentionLM` and
`AttentionDiscriminator`, and left in the names it happened under.
`TransformerCrossAttentionComparer` below is both of those, and
`BilinearGRUComparer` is `ReceiverGRULM` plus `BilinearDiscriminator`. **There
is a fourth act, at the bottom of this section, and it changes what is true
today.**

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

Also left behind: `score_scale_lr` is gated on `BilinearDiscriminator` rather
than applying to both. That gate is an artefact of the parameter's absence, not
an argument against the parameter.

### Attempt three, which is not attempt two again

`AttentionDiscriminator` standardises the attention readout. That is, on its
face, the thing that stopped four rungs learning — so the difference is worth
stating precisely, because reading this section and stopping above would suggest
it had been forgotten.

The standardised readout failed because the *listener as a whole* could not go
quiet. It had to commit through a fixed gain from step zero, before the message
carried anything, and the speaker never got a gradient worth having.

What is standardised now is each of two mixed operands, and the volume is
`log_mix_scale`, downstream of both, unbounded, log-parameterised, with no floor.
So the listener as a whole can still go quiet — the property that mattered —
while neither path can go quiet *on its own*. That second half is the new thing,
and it is deliberate: the attention path cannot escape being learned by turning
itself down, because turning down costs it its whole contribution and buys
nothing back.

The other half of why this is affordable is that the bilinear path is there. At
`a = 0.116` the discriminator is essentially the bilinear comparison, which is
the arm measured reaching 0.938 under a nuisance level where the attention
stacks alone reach 0.469 — so nothing has to be confident early. The fixed gain
had no such companion.

And the listener was never broken, which is what made all of this the wrong
place to look for a while. Handed a message naming the concept, the
cross-attention listener reaches 0.988 and holds its between-candidate share at
0.90; handed a scrambled one it collapses to 0.40. Uniformity is *correct
behaviour* when there is no pattern. Three hypotheses were tested against that
and failed, and are recorded so they are not revisited: message share at
initialisation (raised 0.299 → 0.741 by undamping the referent stack's
cross-attention branch — no effect); readout volume collapse (`|W|` fell 10% in
1500 steps and the bias never moved); and a bounded listener scale (the working
bilinear arm's own `score_scale` *falls* 0.856 → 0.238 and still reaches 1.000).

### Attempts four and five, and why the fifth was the wrong lesson

The two above are attempts one to three. What followed is the reason this section
is now a sequence rather than a story with an ending.

**Attempt four** put the volume in a named scalar on each arm —
`log_score_scale` on the bilinear one, `log_mix_scale` downstream of the
attention arm's standardised mix — both at an elevated 2e-3 so a lone scalar
could move fast enough to matter.

**Attempt five (`a9a6a9c`) took both scalars away and gave the volume to the
weight matrices.** The complaint was legitimate and it is the one this whole
section keeps circling: the listener spent that mobility squashing its own
logits, 0.9021 → 0.3731 on rung 09 and 0.9377 → 0.4072 on rung 11, monotone and
never returning. That is *correct behaviour* on a message carrying nothing, since
BCE's minimiser is `p = 0.5` everywhere. But the scalar factors to the front of
the score, so shrinking it multiplies down every gradient going back through the
message and the channel into the speaker. Going quiet starved the speaker that
would have made it worth listening to.

The conclusion drawn was that the scalar was too cheap to move, and that a matrix
— which has to turn as well as shrink — would not collapse the same way. It did
not collapse. It did not move at all:

| | epoch 0 | epoch 29 |
|---|---|---|
| `bilinear_weight_norm`, rung 09 | 13.055 | 12.889 |
| `bilinear_weight_norm`, rung 10 | 13.049 | 12.968 |
| `log_score_scale`, rung 09, attempt four | 0.9021 | 0.3731 |

1.3% and 0.6% against 59%. A 320×320 matrix under Adam spends its step turning;
the radial component is a small fraction of `lr` per step where a lone scalar
moves about `lr` per step whatever its gradient. And `score_scale_lr` went with
the scalar, so the matrix also dropped from the elevated group to the 1e-4 base.

Rung 10 — the one rung on this ladder that has ever ignited — then sat at
`train_loss` 0.7298 → 0.7006 for a whole run. Above `ln 2` throughout, which is
worse than a constant predictor, with `realised_survival` collapsing 0.545 →
0.190 and `pool_score_norm` climbing 0.0299 → 0.1876: a listener scoring hard on
something that does not discriminate.

**That is attempt two wearing a different hat.** Freedom that cannot be exercised
at the available learning rate is a fixed gain, and it produces the same failure.
The lesson of attempt five is not that the volume should live in a matrix; it is
that "make the cheap move expensive" and "close the collapse" are the same
intervention, and this readout has now been punished twice for it.

### Attempt six: keep the scalar, remove the coupling

`7b10d47`. The objection to the scalar was never that it shrank — it was that
shrinking it also turned the speaker down. Those are two effects of one
multiplication, and they are separable.

`ScoreVolume` standardises the score per game and applies one `log_score_scale`
through `model_util.scale_without_attenuating`: the forward is
`s · standardise(u)`, `∂/∂u` does not carry `s`, and `∂/∂s` is unchanged. The
listener can go as quiet as BCE asks — the freedom attempt two proved is
required — and going quiet costs the speaker nothing. The same treatment went on
the speaker's `logit_scale` in the same commit.

What is deliberately *not* removed is the saturation. The gradient reaching the
message is `σ(s·z) − y` where the plain product gives `s·(σ(s·z) − y)`: both go
quiet on candidates already scored correctly once `s` is large, and only the
uniform factor differs. At initialisation, where `z ≈ 0`, both are `≈ 0.5 − y`,
which is to say this does not invent a bootstrap regime — it stops `s` taking
away the one you start in.

Two things in the record above are now wrong and are left in place rather than
edited, because the sequence is the point. `score_scale_lr` is no longer gated on
`BilinearDiscriminator`: `ScoreVolume` puts the same scalar on both, so one key
reaches both and `mix_scale_lr` has no successor. And `standardise` is back in
the forward path — on the *mixed* score rather than on each branch, which keeps
the single volume knob without pinning the branches to equal spread. That
reopens the escape attempt three closed: a branch can go quiet alone again, and
`mix_share` against `mix_alpha` is what watches it rather than the structure
forbidding it.

Whether any of this makes a rung ignite is unknown at the time of writing. It
removes a coupling that could hold the bootstrap shut; it does not supply what
opens it.

### Attempt seven: the coupling was never reaching the optimiser

Attempt six had a premise five rounds deep and nobody had measured it: that a
scalar at the front of the score, by multiplying the backward pass, changes what
the parameters behind it do. Under this repo's optimiser it does not.

**AdamW divides it out.** The update is `m / √v`. A uniform factor `c` on a
parameter's gradient scales `m` by `c` and `√v` by `c`, so the step is
unchanged. Attempts four, five and six were all arguing about a quantity the
optimiser normalises away before it becomes a step.

**And clipping divides out whatever is left.** `train.py`'s `clip_gradients` is
per-submodule, not global: each module is renormalised to `clip_grad_norm`
whenever its norm exceeds it. The ablation recorded speaker gradient norms around
10 against a ceiling of 1.0, so it binds, and a module whose gradient is clipped
arrives at the optimiser at a fixed norm regardless of any factor upstream.

What `scale_without_attenuating` did add is an inconsistent gradient. `∂L/∂s = x`
is the true partial; `∂L/∂x = J` is not, the truth being `s · J`. The pair is not
the gradient of any function, so nothing guarantees the joint `(x, s)` dynamics
descend the loss, and concretely the machinery behind the scale is shaped for a
volume of 1 whatever the forward uses. At `score_scale` 0.085 — where
`brand-new-birds.csv` ended — that is a listener trained as though it were ten
times louder than it is.

So the helper is gone from both agents and the scalars are plain products again.

**The standardise went with it, and for a better reason than the one first
given.** It was redundant: `BilinearDiscriminator` already layer-norms both
operands of its bilinear form, so the score is already backbone-independent, and
normalising it again downstream bought nothing while costing the exact
`/√referent_embedding_size` that `7b10d47` deleted on the grounds that a
standardise divides any constant out. Restoring it puts the opening back at
`1/√3` = 0.577 at every width and under every backbone — analytic, rather than a
number to measure per rung.

The reason first given was that `standardise` divides each game by the spread of
its own candidate scores, and that spread is the *margin* — so it hands every
game the same magnitude whether its message carried signal or noise, damping the
informative games relative to the uninformative ones. At bootstrap the games that
accidentally do better are the only signal there is, so that weighting is exactly
backwards.

**Measured, it is real and far too small.** On `BilinearDiscriminator`:

| regime | score spread, informative vs noise | effect of `standardise` |
|---|---|---|
| `bilinear.weight` at random init | 0.567 vs 0.567 | uniform 1.77×, which AdamW cancels |
| `bilinear.weight` = identity | 4.20 vs 0.98 | damps informative by ~1.4× |

At initialisation the listener cannot read the message yet, so there is no margin
to divide by and the effect is absent from exactly the regime it was argued
from. It appears only once discrimination exists — which, read forwards, predicts
takeoff followed by collapse rather than a flat failure, and `shapeworld-no-mup.csv`
does exactly that: test accuracy 0.590 and colour accuracy 0.610 at epochs 6–10
with `unique_message_fraction` falling 0.833 → 0.276, then epoch 11 at `uniq`
0.120 and chance.

**What is not explained.** `standardise` is present in exactly the runs whose
*sender* parameters froze — `pool_score_norm` and `polarity_separation` moving
382–445× slower than in the rung 10 that learned — and absent from every run that
was merely dead at chance with those parameters still moving:

| run | `standardise` | `accumulator_steps` | sender pre-channel |
|---|---|---|---|
| rung 10 at `1b512f0` | no | 1 | moving, learned |
| `new_10.csv` | no | 1 | moving, dead |
| `mid_run_birds.csv` | no | 1 | moving, dead |
| `brand-new-birds.csv` | yes | 1 | frozen from epoch 5 |
| `shapeworld-no-mup.csv` | yes | 4 | frozen from epoch 11 |
| `birds-accumulator-2.csv` | yes | 2 | not frozen by epoch 30 |

A 1.4× reweighting, active only after takeoff, is not obviously a 400× freeze.
The correlation is recorded because it is what prompted the change; it is not a
diagnosis, and the removal should not be reported as one.

**The lesson, which is not the one attempts four to six were reaching for.** Five
rounds went into where the volume should live and how it should reach the
backward pass, and the answer to the second half was that the optimiser had
already decided. Before designing around a gradient magnitude, check whether the
optimiser can see it: Adam normalises per parameter, clipping normalises per
module, and `weight_decay = 0.0` removes the one mechanism that would have made
absolute scale matter. What survives all three is a gradient's *direction*.


## Frozen logit spread: NaN through a masked gradient

`logit_spread` bit-identical across epochs is the signature of the AMP
`GradScaler` skipping every step.

Cause: scaling the *already-masked* logits by `logit_scale`.
`d(logits · scale)/d(scale)` is the logits themselves, so the `-inf` at the
reserved slots enters the gradient w.r.t. the scale; the upstream gradient at
those slots is zero, and `-inf × 0` is NaN. `GradScaler` reads that as an
overflow and skips the step — every step, so the whole pair sits frozen at
initialisation. The loss just idles, so nothing else shows it.

Fixed by scaling the *unmasked* logits and re-masking. Harmless whenever the
scale is a constant, since there is then no gradient path to it at all — which is
true again as of 2026-08-30, though the ordering is kept because the trap returns
the moment anything downstream of the mask becomes differentiable. `fccba0f` fixed
the same class of failure for a differentiable `sampling_tau`, which divides the
`-inf` logits and puts `inf` into the gradient the same way.

## The birds run that flattened itself for 35 epochs

A birds run started at 0.62 retained entropy (the old `ln V` scheme's 0.66
coefficient), then annealed *upwards* for 35 epochs to about 0.94 retained,
before accuracy left chance on the way back down at around 0.82–0.85.

Read as a policy that is annealing rather than one that is stuck, the descent is
a cost: the run spent 35 epochs travelling to an entropy it could have been
started at. `init_energy` was then set to 0.9 — near where it chose to go, short
of the 0.94 extreme where messages may carry too little for the listener to learn
from. A design decision from a single run, not a derived bound.

That key is gone. The opening is now `logit_scale`'s own opening of 1.0, which
puts unmixed survival at 0.280 on ShapeWorld and 0.226 on birds, and the scale
learns from there under a 2.0 ceiling. The finding above survives every one of
these rounds and is part of why the opening is left low — a speaker that has to
anneal its way somewhere has spent epochs getting there, and this one anneals
*upward*, which is the direction the run wants anyway. See the entry below.

## The channel had one knob too many

`shapeworld-post-silhouette-update.csv` died at epoch 21 with the speaker's
gradient attenuated ~130,000×. The largest single term was the straight-through
Jacobian, and chasing it produced four things worth keeping.

**The Jacobian is in the *unmixed* `p`.** `flatten_logit_distribution` is a
convex mixture in probability space, so `dy/dz = (1 − w)(diag(p) − p pᵀ)` with
`p` taken before the mixture. `uniform_weight` caps what the listener receives at
0.907 and does nothing whatever to the backward pass. That is why the run read
`realised_survival` 0.9067 against a cap of 0.90714 while the probability
shaping its gradient was 0.99951 — the mixed column could not show it.

**Rank, not magnitude, is the cost.** A uniform factor on a parameter's gradient
cancels in AdamW and is renormalised by `clip_gradients`. What does not come back
is a rank: at `p → 1` the Jacobian is rank ≈ 1, the per-token gradients are
summed before they reach the language model and the vision trunk, and the trunk
hears one token's opinion. This is the argument the identity estimator rests on,
and it is a different argument from the one `7b10d47` was making.

**The normaliser was already a cap, and the scale was moving it *down*.**
`layer_norm_logits` permits a margin of at most `V/√(V−1)`, so at a scale of one
the unmixed winner is capped at 0.789 for `V = 14` — above the 0.703 that
`init_energy = 0.9`'s solved 0.882 allowed. The knob whose job was to open the
channel was closing it, and nobody had computed the free bound to notice.

**A learned scale is a one-way ratchet.** Sharper always helps the current batch
and nothing in the objective pushes back. The tell was predicted in advance:
`0603e27` named "the speaker's stack stalling after the scale is high" as the
cost of removing the tau coupling, and the run fired it exactly, `pool_score_norm`
frozen at 0.2720 and `polarity_separation` at 24.0450 from epoch 21 while the
scale went 3.018 → 3.046. The traverse that scale had been given its own learning
rate to cover was never the constraint either: 4% of its travel bound used.

So `init_energy`, `log_logit_scale` and `logit_scale_lr` were replaced by one
key, inverted in closed form against the sharpest shape the normaliser permits.

**That lasted a day.** The monotone-incentive argument above is a property of the
*gumbel* Jacobian, which collapses to rank ~1 as `p → 1`; `681ef0b` put the whole
ladder on `estimator = "identity"`, whose Jacobian is `I` at any sharpness, so
there is nothing left for a climbing scale to shut. `log_logit_scale` and
`logit_scale_lr` came back on 2026-08-31, opening at 1.0 with no floor and a 2.0
ceiling applied by projection. What is *not* restored is the closed-form bound:
it was bounding a quantity that is no longer a gradient hazard. See
docs/channel.md.

**One float32 detail nearly went in unnoticed.** The identity surrogate is
`onehot.detach() + (z − z.detach())`, and the brackets matter: written
`onehot + z - z.detach()` Python associates left and computes `(1 + z) − z`,
which lands on 1.0000001 — a perturbation of the winning token on every step, in
the one place the change is supposed to alter nothing. It was caught by a test
asserting the forward value bit-for-bit against the gumbel branch, which is the
argument for pinning exactness rather than a tolerance when exactness is what is
claimed.

## A speaker whose sharpness was normalised away

With a *pre-norm* `logit_scale`, the birds speaker spent 55 epochs growing
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

## The Transformer speaker's first symbol was nearly a constant

Rung 9 never ignites, and neither does rung 11. Rung 7 has a different speaker
and ignites at about epoch 3; rung 13 has the *same* speaker and a different
listener and ignites at about epoch 20; rung 10 is rung 9 on CUB, architecturally
identical, and ignites. Every single-factor change from rung 9 rescues it, which
is the signature of a marginal bootstrap rather than a broken component.

`scripts/ignition_audit.py` measured what was marginal. It recorded the sign of
`log_logit_scale`'s gradient per optimiser step — a reading the script no longer
takes, having dropped it when the parameter was deleted on 2026-08-30 and not
restored it when the parameter came back on 2026-08-31 — which matters because
AdamW moves
a lone scalar about `lr` per step whatever the gradient's magnitude — so its
travel over a run is `lr × steps × net sign consistency`, and *only* the sign
consistency is free to vary. Rung 9 sat at **49–52% negative over 1,100 steps**,
a coin flip. Rung 7 sat at the same place for 450 steps and then burst to ~97%
over one 25-step block, settling near 75%, taking its scale from 0.87 to 1.25 and
its `realised_survival` from 0.26 to 0.53 while the loss was still 0.68. Ignition
is a crossing, not a ramp.

Two configuration interventions were tried against that and both failed, which is
worth recording because both had good arguments behind them.

**`silhouette_p_receiver = 0.0`.** ShapeWorld strips colour from the receiver's
view on half of all training games while the speaker always sees it, and the roll
is per game, so a batch is a mix and the receiver's `ResNet18SmallInput`
normalises both distributions against one blended batch statistic. That looked
like it was scrambling the referent side of the map on the listener's side at
exactly the point the listener has to learn it. It is not what stops rung 9:
30 epochs at 0.0 and `pool_score_norm` settled at 0.0324 by epoch 2 and stayed
there, `logit_scale` slid 0.890 to 0.787, and `test_acc` sat at chance throughout — while `train_acc`
climbed to 0.565, which is the colour shortcut arriving by a route the silhouette
was evidently also blocking.

**`accumulator_steps = 1`.** ShapeWorld runs an effective batch of 128 against
CUB's 16, and in a noise-dominated regime `sign consistency − 0.5` grows as
`√batch` while the optimiser step count falls as `batch`, so accumulation is a
straight loss for a lone Adam scalar. The arithmetic held exactly and the
conclusion did not: at four times the steps per epoch the scale travelled about
four times as far — 0.868 to 0.592 over 20 epochs against 0.890 to 0.787 over 30
— in the *same* direction. Net sign consistency came out 1.30% and 1.54% in the
two runs, indistinguishable. Both interventions changed the rate and neither
touched the sign.

### Why the sign is negative

BCE's convexity applied to a message that carries nothing. `L(z) = softplus(z) −
y·z` has `L''(0) = 0.25`, so `E[L(z̄ + ε)] ≈ L(z̄) + 0.125·Var(ε)`. In the
straight-through backward path the listener's score is a function of
`Σ_k y_k E_k`: at small scale `y` is near-uniform and the message embedding
collapses to the mean, `Var(z) → 0`; at large scale `y` sharpens onto a token
chosen mostly by the Gumbel draw and `Var(z)` grows. While the message is
uninformative, raising the scale is a pure variance injection and BCE charges for
it. The gradient is telling the speaker something true: *your message makes the
listener noisier without making it better, so send less.* That is a property of
the loss at p ≈ 0.5, not of the dataset or the architecture, which is why it
survived both interventions. It also unifies two things that had been read
separately — the listener shrinking `score_scale` and the speaker shrinking
`logit_scale` are the same hedge from opposite ends. `7b10d47` acted on that
reading directly, making both scalars apply their gain without carrying it into
the backward pass. Only the listener's half of that survives: the speaker's
scale is no longer a parameter, so it cannot hedge, and the freedom it was given
turned out to be the freedom to saturate its own estimator.

Ignition is therefore a race: the first-order covariance term overtaking that
penalty, which requires the listener to have learned the speaker's **accidental
code** — the random but fixed concept-to-symbol map a freshly initialised speaker
already has. So the thing to measure is how much there is to learn at step zero.

### The measurement

Five seeds, 256 gaussian prototype pairs, both speakers fresh. Two statistics per
content position, both closed-form from the post-mixture distribution rather than
sampled:

- `code_signal` = `1 − P(two different concepts share the intended symbol)`.
- `channel_signal` = `p_same − p_diff`, where `p_same = E_c[Σ_k p_c(k)²]` is the
  probability two draws for the same concept agree and `p_diff` the same across
  different concepts. Zero exactly when every concept emits the same distribution,
  i.e. when the listener can learn nothing.

The Transformer speaker had a hole, and only at **position 0**:

| speaker | position-0 `code_signal` | summed `channel_signal` |
|---|---|---|
| `SenderGRULM` (rung 7) | 0.866 ±0.029 | 0.2261 |
| `SenderTransformerLM` (rung 9, as it was) | **0.485 ±0.284** | 0.1922 |

Positions 1–4 were fine at about 0.88 on both. Per seed, position 0 came out
0.740, 0.323, 0.770, **0.098**, 0.493 — one seed in five emitting essentially the
same first symbol for every concept, and a run-to-run lottery that on its own
predicts rung 9 should sometimes ignite.

The cause was structural. The causal arm's sequence began at SOS, one learned
vector shared by every example, so symbol 0's residual stream carried nothing
about the concept and the referents reached it only through cross-attention
branches scaled by DeepNorm's `beta / alpha` = 0.20. `SenderGRULM` has no such
hole because `init_h` builds its hidden state *from* the prototypes.

### What was tried, and what it cost

| design | DeepNorm | `alpha = beta = 1` |
|---|---|---|
| cross-attention kept, SOS-seeded (as it was) | 0.1922 | 0.2552 |
| cross-attention kept, slots seeded from the latents | 0.2388 | 0.2570 |
| no cross-attention, message = latent tail | **0.2236** | 0.1979 |
| no cross-attention, no free slots (multiplier 1.0) | 0.2059 | 0.1358 |

Two things fall out of that table. **DeepNorm flips sign** on whether the
referents arrive through a branch or through the input: with cross-attention
re-injecting them at every block, `alpha` is a drag; with them in the residual
stream and nothing to refresh them, `alpha` is what keeps them alive and removing
it costs 34%. And **the free slots ahead of the message do the memory's job** —
they are never overwritten, so every message slot reads them at every step, which
recovers about half of what dropping cross-attention costs. More of them buys
almost nothing (0.2254 at multiplier 3.0), and making them bidirectional with a
prefix-LM mask buys +0.1% (0.2239), which does not pay for dropping off SDPA's
fused kernel.

The shipped design is the third row: the message is the tail of the latent array,
there is no cross-attention inside the blocks, and DeepNorm works with the grain.
It scores below the alternatives that keep cross-attention, and was chosen anyway
— it is the one where the residual scaling is protective rather than something to
be worked around, and it collapses the two arms into one stack under one mask.
At six blocks and `ff_inner_size = 512` it holds the parameter match at 1.015x.

### What this does not establish

The whole measured range across nine speaker variants is 0.192 to 0.257 — about
±15% around the GRU. **Nothing here connects initialisation-time code strength to
a gradient sign fraction**, and the variance penalty holding the sign down is a
property of BCE that none of it touches. The seed lottery disappearing is the
strongest result and the one worth checking against reality: if rung 9 has been
run at several seeds and all of them stalled, most of this line is falsified. It
is also all measured at initialisation, on gaussian prototypes, which are
higher-rank than real ones; it says nothing about trainability or the endpoint.

## Parameter costs, for the record

- Absolute position embeddings: ~190k parameters a rung, most of it two
  289-position tables in the `ViT2` backbones.
- A decoder block costs about `4·d_model²` more than an encoder block:
  1,354,951 against 944,711 at 320 wide with `ff_inner_size = 554`. This is why
  the ablation used to run the speaker's two arms at 4 and 5 layers to put both
  within 2% of the GRU baseline. Neither arm builds decoder blocks any more, so
  both run at 6 layers and `ff_inner_size = 512` — 5,854,089 against the GRU's
  5,764,923 on ShapeWorld, 1.015x — and the arms are now the same size as each
  other to within `token_embedding`'s 5,760.
- The small ResNet stem: 1,728 parameters against 9,408.

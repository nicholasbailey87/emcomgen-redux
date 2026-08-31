# The Gumbel channel

Everything in this file lives in `models/sender.py`. It is the noisy channel
between the speaker's logits and the symbol the listener receives, and the
parameters that set how noisy it is.

The pipeline, in order, is:

```
outputs2vocab  →  layer_norm_logits  →  mask_reserved_tokens
               →  × logit_scale      →  flatten_logit_distribution
               →  gumbel_softmax(hard=True, tau=tau)
               →  [estimator: gumbel | identity]
```

`logit_scale` is a learned scalar, opening at 1.0 and bounded above at
`MAX_LOGIT_SCALE` = 2.0 by projection; the estimator chooses only what the
backward pass sees — the forward is the same one-hot either way.

At eval there is no scale, no mixture and no noise: the argmax of the masked,
normalised logits. Eval measures the learned policy rather than a deliberately
noised one. This mirrors jayelm's emergent-generalization, which zeroes
`uniform_weight` whenever the split is not `train`.

## `layer_norm_logits`

Normalises the *emittable* vocabulary logits to zero mean and unit variance, per
example and per position. Only the last `vocabulary` columns are normalised; the
leading four reserved slots (PAD/SOS/EOS/UNK) are concatenated back untouched.
They are masked to `-inf` immediately afterwards so their values are irrelevant,
but they must not be allowed to pollute the mean and standard deviation of the
tokens that can actually be emitted.

This replaces an `nn.BatchNorm1d` over the same columns. LayerNorm is the right
normaliser here because the property wanted is that every speaker arrives at the
exploration gain with logits of comparable *magnitude*, and LayerNorm delivers
that per example rather than on average over a batch. It is also
position-invariant for both speakers by construction — BatchNorm annihilated
per-position offsets in the GRU, which sees one position per call, but preserved
them in the Transformer, which sees all of them at once — has no running
statistics so train and eval agree, and does not couple to `accumulator_steps`.

It is functional and has neither affine parameter, so the transform is
argmax-preserving (it changes no eval-time message) and nothing is added to the
`state_dict`.

### The prior and the sharpness sit on opposite sides of it

That is the whole design. Both are learned; this section is partly a record of
the round in which one of them was not.

**The token prior** is `outputs2vocab.bias`, **pre-norm**. It is divided by the
incoming standard deviation along with everything else, so its influence stays
proportional to the input-dependent signal rather than competing with it
outright. That bound is the reason it goes there. A post-norm beta would have
nothing holding it and could grow until it beat the signal outright — which is
the always-emit-one-token language these runs keep collapsing into
(`test_unique_message_fraction` of 0.005 across 200 games). The price of the
bound is that the prior is weakest late and strongest at initialisation, when
`Wh` is still small; treat it as scaffolding, since a trained `W` can carry token
preferences in its row norms without help.

**The sharpness** is `logit_scale`, **post-norm**, a single number per speaker.
It has to be post-norm to mean anything at all, since this function pins the
variance and would divide any pre-norm scaling straight back out. That is not
hypothetical: the birds speaker once spent 55 epochs growing `logit_spread` from
0.41 to 1.62, saw every bit of it normalised away, and held `realised_survival`
at 0.18 with train accuracy at chance for the whole span.

A scalar rather than LayerNorm's gamma vector, because sharpness is one degree of
freedom and a per-token gamma spreads it over `vocabulary` of them — which then
also have to serve as a token prior, and the shape that suits the listener is not
the shape that maximises sharpness. One parameter per job. It also keeps
argmax-preservation, which a per-token gain would cost.

**It is learned, and for one day it was not.** It is `log_logit_scale`, an
`nn.Parameter` stored as a log so `exp` keeps it positive, with a learning rate
of its own (`logit_scale_lr`). `44767b2` deleted it on 2026-08-30 in favour of a
constant solved from a `token_max_probability` key; `2026-08-31` put it back.

Two reasons were given for deleting it, and neither survives.

*"A learned scale climbs until the straight-through estimator is shut."* That is
a property of the **gumbel** Jacobian `diag(p) − p pᵀ`, which collapses to rank
~1 as `p → 1`. The same commit added `estimator = "identity"`, whose Jacobian is
`I` at any sharpness, and `681ef0b` put the whole ladder on it. Saturation now
costs nothing in the backward pass, so there is nothing for a climbing scale to
shut.

*The ratchet is one-way.* It is not. The section below records the scale sliding
**down** monotonically in failing runs — 0.9094 → 0.6547 on rung 10, 0.8648 →
0.7784 on rung 9. A speaker with nothing to say is pushed flatter, because
confidently emitting the wrong symbol is worse than hedging. It self-regulates.

The bound that replaced it is not restored in another form, because it was
bounding a quantity that is no longer a gradient hazard. What bounds the scale
now is a ceiling, `MAX_LOGIT_SCALE` = 2.0, applied by projection rather than by
a `clamp` — see [the ceiling](#the-ceiling-and-why-projection-and-not-a-clamp)
below.

**And it goes through the helper again.** `7b10d47` first put the gain through
`model_util.scale_without_attenuating` — same forward, `∂/∂normalised` forced to
1 — so that a scale sliding down would not multiply down every gradient reaching
`outputs2vocab`, the stack and the vision model. Round seven took it out on the
grounds that the coupling never reached the optimiser; `b72e5e6` put the
listener's volume back through it and the speaker's gain follows.

The argument for taking it out is right as far as it goes. AdamW updates by
`m / √v`, so a uniform factor on a parameter's gradient scales the numerator and
the denominator alike and cancels; `train.py`'s `clip_gradients` is per-submodule
and renormalises each module to `clip_grad_norm` whenever it binds, which at
recorded speaker norms of ~10 against a ceiling of 1.0 it does. Do not reinstate
the "it changes a ratio between modules" reasoning — AdamW normalises each
parameter separately, so that ratio is exactly what it removes.

What it does not cover is AMP. Both AdamW and `clip_gradients` act *after* the
backward pass, and under `float16` a gradient the scale has divided down can
underflow to zero before either sees it. That is the failure
[anecdotes.md](anecdotes.md) records as skipped steps, and it is the reason the
helper is on both ends of the channel now.

**This is what makes an unfloored scale safe**, and it is the invariant the whole
design rests on. With `∂scaled/∂normalised = 1` from the helper and `∂y/∂scaled =
I` from the identity estimator:

```
dL/dnormalised       =  dL/dy                        (independent of the scale)
dL/dlog_logit_scale  =  ⟨dL/dy, normalised⟩ · scale   (real and nonzero)
```

A scale that slides quiet makes the channel *noisier*, not the stack behind it
*starved*, so there is nothing to floor. `tests/test_exploration.py` pins the
first line as a bit-identical equality across the whole range the parameter can
occupy; it is the assertion most likely to be broken by a later edit.

**The gain sits between `layer_norm_logits` and `mask_reserved_tokens`,**
upstream of the sampler; `gumbel_softmax(hard=True)` keeps its own
straight-through untouched. The
gradient into the raw logits is a product of three factors:

```
dL/draw  =  J_gumbel(scaled)  ×  d(scaled)/d(normalised)  ×  d(normalised)/d(raw)
```

and the helper changes the middle one from `logit_scale` to 1, at every scale.
That is the whole of what it does, and it is exact.

The end-to-end number is not flat, because `J_gumbel` is itself a function of
`scaled` — the soft surrogate is `softmax((scaled + g) / tau)`, which saturates
as the scale grows. That is the saturation, it belongs to the sampler, and it is
deliberately kept. Measured on the decoder arm at a fixed seed, gradient norm
into the raw logits:

| `logit_scale` | 0.05 | 0.25 | 1.0 | 4.0 | 20.0 |
|---|---|---|---|---|---|
| plain product | 3.3e-8 | 1.5e-7 | 4.9e-7 | 1.7e-7 | 5.7e-8 |
| through the helper | 6.6e-7 | 6.0e-7 | 4.9e-7 | 4.4e-8 | 2.8e-9 |

Downwards — the direction every failing run travels — the plain product loses an
order of magnitude as the scale falls 20× and the helper does not.

Upwards the plain product looks better, and the tempting reading of that is
wrong. The helper does not attenuate more at high scale; it does the same thing
it does everywhere. What the plain product has above ~1.0 is a factor
`logit_scale` that happens to *offset* the sampler saturating, so removing it
exposes an attenuation that was always the sampler's. No run on this ladder has
been there.

### `eps = 1e-12`, and why that is load-bearing

`F.layer_norm` divides by `sqrt(var + eps)`, so scale invariance holds only while
the incoming variance is large against `eps`; below that the normaliser quietly
stops normalising and the emittable logits come out *smaller* than unit variance.
`logit_scale` can in principle learn its way out of that, but only slowly, and
only if the gradient survives the noisier channel in the meantime — where the
per-batch solve it replaced absorbed it immediately and silently. A collapsing
speaker essentially gets a weaker channel, which is the honest behaviour and the
reason `eps` is set where it is.

The headroom is much smaller than raw logit scales suggest. A freshly built GRU
speaker emits pre-norm logits with a standard deviation of ~0.24, and at the
`1e-5` default the normaliser starts giving out below ~0.01 — a margin of roughly
24×. Shrinking that speaker's output layer 1000× drops realised survival from
0.43 to 0.09; a channel that noisy then starves the gradient that would restore
the logits, so it runs away. Observed on a birds run whose `realised_survival`
fell 0.47 → 0.17 over 22 epochs.

At `1e-12` the same 1000× collapse leaves survival at 0.43, unchanged to four
decimal places, and the normaliser holds down to a standard deviation of ~1e-6.
`tests/test_exploration.py` pins both the invariance and where it finally stops.
`logit_spread` in `metrics.csv` is the column that makes a collapse visible
rather than something inferred after the fact.

`receiver.LAYER_NORM_EPS` mirrors this constant for the same reason: at the 1e-5
default a referent at RMS 0.01 comes out 4.5% off, taking the *relative* scores
between candidates back out of the listener's hands and putting them in the
backbone's. Not currently binding — ViT2 emits RMS 0.23 — but closing it costs
nothing. The score's overall magnitude is no longer at stake there: the
listener normalises both operands of its bilinear form, so what a backbone emits
reaches the score only through its *direction*. `referent_layer_norm` is half of
that, and it is also what stops a large candidate being read loudly for being
large — which is the half no downstream normalisation could have undone. See
`ScoreVolume` in [architecture.md](architecture.md).

## `mask_reserved_tokens`

Sets the four reserved tokens to `-inf` so they can never be emitted mid-message.
SOS and EOS are attached by the caller instead, so messages are fixed-length.

Out of place, because it runs directly on the output of the vocabulary
projection, and writing `-inf` into that in place would be modifying a tensor
autograd still needs.

It runs **before** the exploration noise so that the uniform mixture is spread
over the emittable tokens only.

## `flatten_logit_distribution`

Mixes a uniform distribution into the logits at weight `w`, in log space.

The uniform component is spread over the emittable tokens only, i.e. those not
already masked to `-inf`. Spreading it over all `vocabulary + 4` slots and
masking afterwards would throw away the `4/(V+4)` of it that landed on reserved
tokens, so a nominal weight of 0.1 would deliver 0.078.

Masked entries are `-inf` in both components, and `logsumexp` of two `-inf`
backpropagates NaN, so they are mixed as a finite placeholder and the mask is
restored afterwards. `torch.where` routes the gradient to the selected branch
only, so the placeholder never reaches the speaker.

## `logit_scale` — the speaker's channel scale

`F.gumbel_softmax(..., hard=True)` emits `argmax(logits + g)` with
`g ~ Gumbel(0, 1)`, whose standard deviation is a fixed 1.283, so how much of the
speaker's distribution survives the noise is set by the size of the logits
relative to that. LayerNorm pins them to unit variance for every speaker, and the
scale says what that unit is worth. Larger scale, sharper distribution, less
entropy.

It is `exp(log_logit_scale)`, a 0-d `nn.Parameter` on the speaker's language
model, in `state_dict`, with a clip group and a learning rate of its own. It
opens at **1.0**, has **no floor**, and is bounded above at **2.0**
(`sender.MAX_LOGIT_SCALE`).

### The ceiling, and why projection and not a `clamp`

`train.py`'s `optimiser_step` calls `GumbelChannel.project_channel()` after
`scaler.step`, which clamps the *parameter* — not the forward value — back to
`ln 2`. Three parameterisations were considered:

| | positivity | ceiling | gradient at the bound |
| --- | --- | --- | --- |
| `exp(x).clamp(max=2)` | free | exact | **zero — welds permanently** |
| `2·sigmoid(x)` | free | asymptotic | live, but saturates *both* ends |
| **`exp(x)` + projection** | free | exact | live right up to it |

`clamp` is rejected for the reason `receiver.py` already gives about the mix
floor: *the floor is in the parameterisation and never a `clamp`, because
`clamp`'s gradient is zero past the bound and is not directional*. The
derivative is `1 if x < 2 else 0`, so a parameter that overshoots gets no
gradient in **either** direction and cannot come back; `weight_decay` is 0.0, so
nothing else would pull it.

`2·sigmoid(x)` avoids the weld but its derivative, `scale·(1 − scale/2)`, peaks
at 0.5 at the opening and falls away at both ends — recovery from a low scale is
~5× slower than the descent that got there. Under `exp` the rate is
proportional, which is the right behaviour for a scale.

Projection keeps `exp`'s proportional traverse and an exact 2.0, at the cost of
the constraint being applied outside the forward pass. That cost is paid by
giving the mixin a method, so the module still owns the rule.
`scaler.step` may skip on inf/nan; projection is idempotent, so that is harmless.

**Expect the ceiling to bind, early and often.** At `logit_scale_lr` = 6e-3 and
156.25 optimiser steps an epoch, a sign-consistent gradient covers `ln 2` in 0.74
of an epoch. That is the case the design is for and is *not* a fault — sitting at
the bound costs nothing and leaving it is free. But it is what makes
`train_logit_scale` worth watching in the first runs: whether it pins at 2.0
immediately, and if so whether 2.0 is the wrong ceiling rather than 6e-3 the
wrong rate; and whether it ever comes back **down**, which is the behaviour a
`clamp` would have made impossible.

### There is no floor, and that is deliberate

The scale slides down in every run that fails: 0.9094 → 0.6547 on rung 10,
0.8648 → 0.7784 on rung 9, monotone. That is not a runaway to be floored. A
speaker with nothing to say is pushed flatter because confidently emitting the
wrong symbol is worse than hedging — BCE's minimiser is `p = 0.5` everywhere on a
message carrying nothing — so the slide is the objective working, and it reverses
when the speaker has something to say.

What made it *look* like a runaway is that a plain product multiplies the whole
speaker's stack by the scale on the way back, so a quiet speaker starved the
gradients that would have given it something to say.
`model_util.scale_without_attenuating` removes exactly that, and it is why the
scale is bounded above and not below. Read `train_logit_scale` as
`train_score_scale` is read: a dip and a return is a speaker declining to commit,
a monotone slide with no return is a collapse.

### Why the shape budget matters more than the scale

The scale is one route to fidelity and never was the only one.
`layer_norm_logits` pins the logits to unit variance but leaves a **shape**
budget: the sharpest arrangement it permits is one token at `√(V−1)` and the rest
at `−1/√(V−1)`, whose margin is `V/√(V−1)` — 3.883 sd at V = 14, 4.588 at V = 20
(`sharpest_logit_margin`). At a scale of *one* that alone caps the unmixed winner
at 0.789 for ShapeWorld and 0.838 for birds. Runs do traverse it with the scale
pinned; the 2026-08-29 ShapeWorld run had spent 86% of it.

So saturation is set by `logit_scale · logit_margin`, and each factor has its own
ceiling. Their product, 0.9945 at V = 14, is the sharpest channel the design
permits at all. `logit_margin` (docs/measurement.md) is the column that says
which of the two routes a run took, and it is not substitutable for
`logit_scale`.

### The bound that was here, and why it is not restored

Between 2026-08-30 and 2026-08-31 the scale was a constant, solved in closed form
from a `sender_language_model.token_max_probability` key against exactly that
sharpest shape, so that no speaker could ever hold a token above a configured
probability. It is gone, and it is deliberately **not** reinstated in another
form.

The quantity it bounded is `p`, the winner's probability before the uniform
mixture — which is the `p` in the gumbel estimator's Jacobian
`(1 − w)(diag(p) − p pᵀ)`. That Jacobian collapses to rank ~1 as `p → 1`, and
that collapse was the hazard. The ladder now runs `estimator = "identity"`, whose
Jacobian is `I` at any sharpness, so `p` no longer reaches the backward pass at
all. Bounding it would be bounding a diagnostic.

What remains true, and is why there are still two survival columns:
`uniform_weight` mixes in probability space, so `m = (1 − w)·p + w/V` and the
mixture contributes a *constant* to the backward pass. It caps what the listener
receives at `1 − w + w/V` = 0.907 and does nothing to the gradient. That gap is
how the 2026-08-29 run read `realised_survival` 0.9067 against a cap of 0.90714
while the probability shaping its gradient was 0.99951. `unmixed_survival` is the
column that reports `p`.

### Where a run opens

At initialisation the normalised logits are i.i.d. standard normal — random
weights through a linear projection whose rows are independent, so nothing
correlates the vocabulary dimension yet — and the opening scale of 1.0 fixes the
opening from there. Unmixed survival, at the two ends of the scale's range:

| | scale 1.0 | scale 2.0 |
| --- | --- | --- |
| ShapeWorld, V = 14 — fresh speaker | 0.280 | 0.511 |
| ShapeWorld, V = 14 — sharpest legal shape | 0.789 | 0.995 |
| birds, V = 20 — fresh speaker | 0.226 | 0.455 |
| birds, V = 20 — sharpest legal shape | 0.838 | 0.998 |

**Bootstrapping is why the opening wants to stay modest.** A fresh speaker's
argmax is very nearly input-independent: it has learned nothing, so its preferred
token barely varies with the referent. If that argmax is transmitted reliably,
the speaker emits one message for every input, confidently, from the first batch,
and the listener co-adapts to that degenerate language before the speaker's
embeddings are worth grounding anything on. Near-random messages carry no
premature structure to co-adapt to, and the pair sharpens together as the
embeddings become worth using.

Both datasets open at the same multiplier, so the vocabulary difference shows up
where it belongs — in what that multiplier buys — rather than in a different
bound for each. A larger vocabulary is the slightly *louder* channel at equal
scale, because `V/√(V−1)` grows with V.

### What the other end is

`uniform_weight` (w) owns the trained end from the *listener's* side: mixing caps
a slot's winner at `1 − w + w/V` however sharp the logits get, which at w = 0.1
is 0.907. So at least `w·(1 − 1/V)` of symbols are flipped no matter what — a
permanent per-symbol corruption rate training cannot reduce, which is the point.

`uniform_weight` bounds what arrives; `MAX_LOGIT_SCALE` and
`sharpest_logit_margin` bound the two factors the gradient is written in. Only
the mixture survives into the message, and under `estimator = "identity"` none
of the three reaches the backward pass at all.

Where a run actually lands between the opening and the ceiling is a **finding**,
reported by `realised_survival`, `unmixed_survival`, `logit_margin` and
`logit_scale`, not a design input.

## `tau`

The temperature handed to `gumbel_softmax`, flat at its configured value for the
whole of any run.

`hard=True` emits `argmax(logits + g)`, which is invariant to any positive `tau`,
so this shapes the *soft* sample and nothing else. On the `"gumbel"` branch that
soft sample is what the estimator differentiates, so `tau` is a pure backward
knob there. On the `"identity"` branch the soft sample is discarded, so `tau`
does nothing at all.

### It used to be coupled to the scale, and cannot be again

`17ae9f9` tied it to `logit_scale / initial_logit_scale` on a cosine schedule
over training, so that a sharpening speaker got a correspondingly softened
surrogate and the straight-through Jacobian could not collapse. `0603e27`
commented that out; the lines and the argument stood in this file until
2026-08-30, when the learned scale was removed and there was no longer a ratio
for the schedule to track. Both are gone.

The record of what it traded is worth keeping, because it is the reason the
channel now looks the way it does.

**Why it was turned off.** The coupling is a *pin* on the scale, not a floor.
Under it the surrogate reduces to `softmax(L + g/scale)`, and `layer_norm_logits`
holds `L` at unit variance, so the scale leaves the signal term entirely: the
only gradient `log_logit_scale` still received was through `g/scale` — "these
particular Gumbel draws would have hurt less had I been louder" — which is a
different answer every batch. Rung 9 moved `log_logit_scale` by −0.008 over ten
epochs, 0.2% of its travel bound, at chance throughout.

**Why turning it off cost something.** `0603e27` predicted that cost with a named
tell: *the speaker's stack stalling after the scale is high —
`pool_score_norm` and `polarity_separation` flattening while `logit_scale` and
`realised_survival` keep climbing.* `shapeworld-post-silhouette-update.csv` fired
it exactly. From epoch 21 `pool_score_norm` was frozen at 0.2720 and
`polarity_separation` at 24.0450, while `logit_scale` went 3.018 → 3.046 and
survival closed the last 0.0003 to its cap. The uncoupled surrogate saturated,
and the speaker's four gradient norms fell to ~2e-7.

**And why neither side of that trade is the answer.** The coupling removes the
*scale* route to saturation and leaves the shape route open, since
`softmax((z + g/s)/c)` still saturates if `z`'s top-two gap grows — which is the
route that run actually took, at 86% of the shape budget. So a reinstated
coupling would not have caught it either.

None of this is a live hazard any more. The saturation it was managing is a
property of the gumbel Jacobian, and `estimator = "identity"` removes it
outright; `tau` shapes a surrogate that branch discards. Both bounds on
sharpness — `MAX_LOGIT_SCALE` and `sharpest_logit_margin` — are there to keep
the channel legible rather than to protect a backward pass.

The same run says the traverse `logit_scale_lr` was raised to buy was never
step-limited: `log_logit_scale` moved 0.05–0.09 log-units in epochs 1–4 and
0.010–0.014 in the stall at 5–6, against a bound of at least 0.28 — 4% of it. It
was gradient-limited. Worth remembering before reading a flat
`train_logit_scale` as a rate that is too low.

## The two estimators

`sender_language_model.estimator` selects what the speaker learns through. The
forward pass is identical on both branches — one shared `_gumbel_sample`, so at
the same seed they emit not similar messages but *identical* ones. Any difference
between two runs is the backward pass and nothing else, which is what makes an
A/B between them a control rather than two experiments.

```
"gumbel"    dy/dz = (1 − w)(diag(p) − p pᵀ)      the soft sample's Jacobian
"identity"  dy/dz = I                             y = onehot.detach() + (z − z.detach())
```

### The argument is rank, not magnitude

The soft Jacobian's cost is not that it is small. It is that it is **low rank**.

At `p` near one-hot, `diag(p) − p pᵀ` has rank ≈ 1: thirteen of fourteen
directions carry 0.8% between them at `p = 0.992`. The per-token gradients are
then *summed* into one vector before they reach the language model and the vision
trunk, so all but one direction is destroyed before any optimiser or clipper gets
a look, and the trunk hears a single token's opinion about what the message
should have been.

Magnitude, by contrast, largely cancels. AdamW updates by `m / sqrt(v)`, so a
uniform factor on a parameter's gradient scales both; `clip_gradients`
renormalises each submodule to `clip_grad_norm` whenever it binds. For
`outputs2vocab.weight`, where each row is its own set of parameters, a
consistently 1000x-attenuated token still takes a full `lr`-sized step. No
per-parameter normaliser recovers a rank.

That argument does have a floor, and it is `[optimiser] eps`: `m / sqrt(v)` is
only scale-free while `sqrt(v) ≫ eps`. At the dead run's ~2e-7 speaker norms
AdamW's 1e-8 default was already damping the update by about 9%, so the epsilon
is set to 1e-12 — see DEFAULT.toml for the trade.

### What the identity estimator is

Sample faithfully, then let the backward pass go straight to the logits:

```python
with torch.no_grad():
    onehot = self._gumbel_sample(normalised)

emittable = normalised[..., 4:]
onehot = torch.cat(
    [onehot[..., :4], onehot[..., 4:] + (emittable - emittable.detach())],
    dim=-1,
)
```

The sample is unchanged and still faithful: `argmax(z + g)` *is* a categorical
draw from `softmax(z)`. The speaker's gradient becomes `dL/dy`, which is the
receiver's per-token embedding sensitivity `⟨dL/dm, Eᵢ⟩` — full rank, the same
size at `p = 0.95` as at `p = 0.2`, and completely blind to how sharp the speaker
has become.

Three details in that are load-bearing.

**The surrogate is built on the emittable slice, not on the masked logits.**
`masked` holds `−inf` in the four reserved columns and `−inf − (−inf)` is NaN.
Slicing also stops those columns receiving gradient at all: they are constants
from the sampler's point of view, so `outputs2vocab` rows 0–3 and the stack
behind them are never trained toward tokens that cannot be emitted.

**It taps the *scaled* logits**, and has to. `_gumbel_sample` runs under
`no_grad` on this branch, so the surrogate is the only path back to
`log_logit_scale`; an unscaled tap would leave the parameter with no gradient at
all. That this costs the speaker's stack nothing is the composition described at
the top of this file: `scale_without_attenuating` gives `∂scaled/∂normalised = 1`
and the estimator gives `∂y/∂scaled = I`, so `dL/dnormalised = dL/dy` exactly, as
it was under the unscaled tap, while `dL/dlog_logit_scale` is
`⟨dL/dy, normalised⟩ · logit_scale`.

**The bracketing is not cosmetic.** `onehot + z - z.detach()` associates left, so
it computes `(1 + z) − z`, which in float32 is 1.0000001 rather than 1 — a
perturbation of the winning token on every step, in the one place the estimator
is supposed to change nothing. Forming the zero first makes the addition exact.
`test_the_identity_surrogate_forwards_exactly_the_one_hot` pins it against the
gumbel branch bit for bit.

### What to read on each branch

`unmixed_survival` is a gradient diagnostic on `"gumbel"` and a fidelity reading
on `"identity"`. On the first, `1 − p` is the factor the estimator's Jacobian
turns on; on the second the Jacobian is `I` and the column reaches the gradient
not at all. It is still worth watching there — it says how much of the message
arrives — but a run that saturates it is not thereby in trouble.

The saturation signature in docs/training.md — the speaker's stack flattening
while survival climbs — therefore **cannot fire on the identity branch**. On the
gumbel branch it is bounded rather than impossible: `MAX_LOGIT_SCALE` and
`sharpest_logit_margin` together cap `p` at 0.9945 at V = 14, so the collapse is
bounded at `(1 − 0.9945)/(1 − p_open)` rather than unbounded.

## Scale the unmasked logits, then re-mask

Both decode loops do this:

```python
logits = mask_reserved_tokens(
    model_util.scale_without_attenuating(normalised, self.logit_scale)
)
```

rather than scaling the already-masked tensor. `d(logits · scale)/d(scale)`
is the logits themselves, so scaling *after* the mask sends `-inf` into the
gradient w.r.t. the scale; the upstream gradient at those slots is zero, and
`-inf × 0` is NaN. The AMP `GradScaler` reads that as an overflow and skips the
step — every step, so the whole pair sits frozen at initialisation. Invisible in
the loss, which just idles; `logit_spread` bit-identical across epochs is the
tell. Harmless while the scale was a constant, since there was no gradient path
to it at all.

Order within the loop matters throughout:

1. **Normalise before scaling and mixing** — normalisation is what fixes the
   magnitude the scale is expressed against, and it would otherwise upset the
   mixture's bounds.
2. **Mask before mixing** — so the uniform component is spread over emittable
   tokens only.
3. **Scale before mixing** — the scale sets how much of the fixed 1.283-sd Gumbel
   noise the logits stand up to; scaling *after* the mixture undoes the bounds
   the mixture exists to impose.

## `mean_winning_probability`

The fraction of symbols that survive the Gumbel noise, averaged over slots, and
the source of **both** the `realised_survival` and `unmixed_survival` columns —
the same call with `uniform_weight` passed and with it zeroed.

The pair is there because the mixture caps the reported number without capping
the channel. `realised_survival` cannot exceed `(1 − w) + w / V`, 0.90714 at
ShapeWorld's settings, so a speaker that has committed entirely still reads 0.91.
On the `"gumbel"` branch the gradient runs through the soft sample, whose
Jacobian is `diag(p) − p pᵀ`, and the `p` there is pre-mixture — so
`1 − unmixed_survival` is the factor the estimator turns on, and it is the column
with the dynamic range. On `shapeworld-post-silhouette-update.csv` a reported
0.90670 inverted to 0.99951, a 510× attenuation against epoch 0 that the mixed
column could not show. Nothing bounds this column directly; `MAX_LOGIT_SCALE`
and `sharpest_logit_margin` bound the two things it is bought with, and on the
`"identity"` branch it leaves the gradient altogether.

By the Gumbel-max identity, the probability that a slot's argmax is unchanged by
the noise is exactly the winning token's softmax probability. So survival can be
read straight off a softmax: no Monte Carlo over noise draws, no assumed logit
distribution, and no seed. `tests/test_exploration.py` pins the identity.

It applies the real sampling pipeline in the real order — scale first, then the
uniform mixture — so that the mixture's bounds hold.

This is purely a measurement. It used to be the inner loop of a solve that chose
the scale to hit a requested rate, and later read a learned scale; the scale is a
constant now, so what varies in this column over a run is the logit *shape*
alone. That is worth keeping in mind when reading it against `logit_margin`,
which reads the shape directly.

Both speakers pool the measurement over positions once per batch rather than per
position, so it reads the batch's statistics rather than each position's alone.
The parallel arm can take it straight off its logits because it emits every
position in one shot; the GRU and the causal arm have to stack theirs.

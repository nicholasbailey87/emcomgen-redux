# The Gumbel channel

Everything in this file lives in `models/sender.py`. It is the noisy channel
between the speaker's logits and the symbol the listener receives, and the
parameters that set how noisy it is.

The pipeline, in order, is:

```
outputs2vocab  →  layer_norm_logits  →  mask_reserved_tokens
               →  × logit_scale      →  flatten_logit_distribution
               →  gumbel_softmax(hard=True, tau=sampling_tau)
```

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

### The two learnable channel parameters sit on opposite sides of it

That is the whole design.

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

**The sharpness** is `log_logit_scale`, **post-norm**, a single scalar per
speaker. It has to be post-norm to mean anything at all, since this function pins
the variance and would divide any pre-norm scaling straight back out. That is not
hypothetical: with a constant scale the birds speaker spent 55 epochs growing
`logit_spread` from 0.41 to 1.62, saw every bit of it normalised away, and held
`realised_survival` at 0.18 with train accuracy at chance for the whole span.

A scalar rather than LayerNorm's gamma vector, because sharpness is one degree of
freedom and a per-token gamma spreads it over `vocabulary` of them — which then
also have to serve as a token prior, and the shape that suits the listener is not
the shape that maximises sharpness. One parameter per job. It also keeps
argmax-preservation, which a per-token gain would cost.

**And it is a plain product, briefly not.** `7b10d47` put the gain through
`model_util.scale_without_attenuating` — same forward, `∂/∂normalised` forced to
1 — so that a scale sliding down would not multiply down every gradient reaching
`outputs2vocab`, the stack and the vision model. It does slide in every run that
fails: 0.9094 → 0.6547 on rung 10 and 0.8648 → 0.7784 on rung 9, monotone. The
same reasoning put `ScoreVolume` on the listener.

Both were answering a coupling that never reached the optimiser. AdamW updates by
`m / √v`, so a uniform factor on a parameter's gradient scales the numerator and
the denominator alike and cancels; `train.py`'s `clip_gradients` is per-submodule
and renormalises each module to `clip_grad_norm` whenever it binds, which at
recorded speaker norms of ~10 against a ceiling of 1.0 it does. What the helper
added instead was an inconsistent gradient — `∂L/∂scale = x` is the true partial
while `∂L/∂x = J` is not, the truth being `scale · J` — so the stack was shaped
for a channel of volume 1 whatever the forward used. See
[anecdotes.md](anecdotes.md), round seven.

The forward was never in question either way, so the fidelity against the fixed
1.283 noise floor — the scale's real job — and `∂L/∂log_logit_scale`, which is
what `scripts/ignition_audit.py`'s covariance reads, are unaffected by the
removal.

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
`log_logit_scale` is learned, not solved per batch, so it can only absorb that at
whatever rate gradient descent manages — where the per-batch solve it replaced
absorbed it immediately and silently.

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

## `logit_scale` — the exploration control

`F.gumbel_softmax(..., hard=True)` emits `argmax(logits + g)` with
`g ~ Gumbel(0, 1)`, whose standard deviation is a fixed 1.283, so how much of the
speaker's distribution survives the noise is set by the size of the logits
relative to that. LayerNorm pins them to unit variance for every speaker, and the
scale says what that unit is worth. Larger scale, sharper distribution, less
entropy.

`logit_scale(init_energy, vocabulary, uniform_weight)` resolves the *initial*
scale by numerical bisection. Each speaker then stores its log in
`log_logit_scale` and learns from there, so `init_energy` fixes where a run opens
— and so still equalises the opening channel across architectures — but not where
it settles. Where it settles is reported by `realised_survival`.

Stored as a log so that `exp` keeps it strictly positive: a negative scale would
invert the speaker's preferences rather than flatten them, and only a positive
one is argmax-preserving.

### Why entropy, and why at initialisation

The point of a noisy channel here is not fidelity, it is **bootstrapping**. A
fresh speaker's argmax is very nearly input-independent — it has learned nothing,
so its preferred token barely varies with the referent. If that argmax is
transmitted reliably, the speaker emits one message for every input, confidently,
from the first batch, and the listener co-adapts to that degenerate language
before the speaker's embeddings are worth grounding anything on. High entropy at
the start is what prevents this: near-random messages carry no premature
structure to co-adapt to, and the pair sharpens together as the embeddings become
worth using.

So the knob is deliberately expressed as *entropy retained*, not as channel
capacity or as a symbol error rate. Both of those were tried and both mislead:

- **Capacity** (mutual information over its maximum) runs the wrong way round.
  High capacity means a sharp, low-entropy speaker, i.e. *less* room to
  bootstrap — so a config asking for "80% capacity" is asking for the opposite of
  what it sounds like.
- **"Fraction of symbols flipped"** presupposes a correct symbol. At
  initialisation there is no correct symbol: argmax is not an intended message,
  it is an accident of the initialisation. Counting departures from it as errors
  imports a notion of correctness that does not exist yet.

Entropy has neither problem. It is a property of the distribution alone, it needs
no reference symbol, and it runs the way intuition does: higher means flatter
means more room to explore.

`initial_energy` computes `H(p) / log2(V)`, so 1.0 is a uniform speaker that has
committed to nothing and 0.0 is one that emits a single token with certainty.

### Why a numerical solve, and why no `ln(V)` term

An earlier version used a closed form, `coefficient × ln(V)`, on the argument
that a winner must beat the largest of `V` Gumbel draws and
`E[max_i g_i] = ln V + γ`. That is the right correction for holding a *survival
rate* constant across vocabularies, but it badly overshoots for holding *entropy*
constant. Measured over V = 8..256, the scale that holds entropy fixed varies
only 1.2–1.4×, while `scale / ln(V)` varies about 2× — so dividing by `ln(V)`
introduces roughly four times more vocabulary dependence than it removes. The
residual really is logarithmic, but with a much smaller coefficient: at 80%
retained, `scale ≈ 0.87 + 0.12·ln(V)` fits to about 2% over that range.

Rather than fit that, the scale is solved for numerically. It costs one bisection
at construction, it is exact for any `(V, w)` instead of approximate over the
range someone happened to check, and it puts the design decision itself in the
config rather than a coefficient that encodes it.

`initial_logit_sample` makes the solve deterministic: a fixed draw of the logits
a *freshly initialised* speaker produces, reused at every bisection step. After
`layer_norm_logits` the emittable logits are zero mean and unit variance, and at
initialisation they are also i.i.d. standard normal — random weights put the
referent through a linear projection whose rows are independent, so nothing
correlates the vocabulary dimension yet. **This is the one place in the scheme
where the Gaussian assumption is a fact about the model rather than a proxy for
one, which is why the operating point is defined at initialisation and not
anywhere later.**

Reusing one sample also makes the solve exactly monotone in the scale, which is
what makes bisection valid. The constants: 2^16 samples put the resolved scale
within about 0.5% of its large-sample limit; the bracket spans six orders of
magnitude, searched geometrically, and 48 steps close it to better than one part
in 10^4.

### What the other end is

`uniform_weight` (w) owns the trained end: mixing caps a slot's winner at
`1 − w + w/V` however sharp the logits get, which at w = 0.02 is a floor of about
0.05 on retained entropy. The two knobs barely interact. Mixing only matters when
some token holds much more than `w/V`, so at the flat end the scale is the whole
story and `uniform_weight` changes nothing measurable; at the sharp end the cap
binds and the scale stops mattering.

Where a run actually lands between the two is a **finding**, reported by
`realised_survival` and `logit_spread`, not a design input. Do not calibrate the
scale against an assumed trained shape — that number is unmeasured,
`uniform_weight` already bounds it, and letting it into the chain sets the
operating point from a guess.

If `init_energy` is below the floor `uniform_weight` imposes, `logit_scale` warns
and the scale pins at `ENERGY_SCALE_MAX`; lower `uniform_weight` to ask for less.

### Rederiving the default

Reference points for birds (V = 20, w = 0.02), all computable from
`initial_energy`:

| retained entropy | 0.94 | 0.90 | 0.85 | 0.77 | 0.62 | 0.57 |
| --- | --- | --- | --- | --- | --- | --- |
| scale | 0.64 | 0.84 | 1.05 | 1.37 | 1.99 | 2.23 |
| argmax probability | 0.14 | 0.19 | 0.23 | 0.31 | 0.45 | 0.49 |

The default of 0.9 is set from the one trajectory that has been measured. A birds
run started at 0.62 retained (the `ln V` scheme's 0.66 coefficient), then
*flattened itself* for 35 epochs, reaching about 0.94 retained, before accuracy
left chance on the way back up at around 0.82–0.85. Read as a policy that is
annealing rather than one that is stuck, the descent is a cost: the run spent 35
epochs travelling to an entropy it could have been started at. 0.9 starts it near
where it chose to go, and short of the 0.94 extreme, where messages carry so
little that there may be nothing for the listener to learn from.

That is a design decision taken from a single run, not a derived bound, and it
should be revisited when there are more. What is *not* a free choice is the
direction: lower than about 0.6 reproduces the premature-sharpening failure this
scheme exists to avoid.

## `sampling_tau`

The temperature handed to `gumbel_softmax`: the configured `tau`, adjusted
towards `tau × logit_scale / initial_logit_scale` by a cosine schedule over
training.

```
ratio  = max(logit_scale / initial_logit_scale, 1)
weight = (1 + cos(π · training_progress)) / 2
tau    = configured_tau · (1 + weight · (ratio − 1))
```

So a run opens fully coupled and ends at exactly the configured `tau`, where the
surrogate is an honest picture of the sharpness the listener actually receives.
The cosine is flat early, which is what makes a single schedule work across
datasets that take off at very different times: at 100 epochs the weight is still
0.97 at epoch 11, where ShapeWorld leaves chance, and 0.65 at epoch 40, where
birds does. Both take off fully coupled and the coupling retires over the last
third.

**Being open-loop is the point**, not a compromise. Driving the schedule from
`realised_survival` or from accuracy would track the model rather than the clock,
but it would also hand every rung of the ablation a different tau schedule — so
arms would differ in their estimator as well as their architecture, which is the
confound the ladder exists to avoid.

### Why the coupling at all

`tau` shapes the soft sample the straight-through estimator differentiates; it
leaves the hard forward sample alone, which is an argmax and so invariant to it.
With `tau` held constant while the speaker sharpens, the surrogate sharpens with
it and the gradient collapses onto whichever token won the Gumbel draw: measured
over unit-variance logits at V = 20, the effective number of tokens carrying
gradient falls from 4.9 at the opening scale to 1.6 by a scale of 6, with the
winner holding 0.80 of the mass. The losing tokens then receive almost nothing,
so the speaker stops being told what the alternatives were worth exactly as its
channel becomes good enough to use them.

Scaling `tau` with the scale holds that open instead — the same measurement gives
9.0 to 9.5 effective tokens across the range, since the surrogate reduces to
`softmax(L + g/scale)` and `L` is pinned to unit variance by `layer_norm_logits`.

### Why the ratio is floored at 1

Below the opening scale the coupling runs backwards: the surrogate becomes
`softmax(s₀·L + s₀·g/s)`, whose noise term grows as the scale shrinks. Measured
at V = 20, scale 0.35, that takes the surrogate from 4.9 effective tokens to 2.0
while the token it favours matches the noiseless argmax only 9% of the time — a
confident gradient pointing at noise. `tau` is monotone and cannot change *which*
token is favoured, only how hard the gradient commits to it, so there is nothing
to be gained in that direction.

### Why against `initial_logit_scale` rather than raw

So that a fresh speaker samples at exactly the configured `tau` and the coupling
changes nothing at initialisation. It diverges only as the speaker moves its own
scale, which is the behaviour being added.

### Why detached, and that is not optional

`gumbel_softmax` divides the logits by it, and the reserved slots are `-inf` by
then, so a differentiable tau puts `inf` into the gradient with respect to the
scale and NaN into the step — the same failure `fccba0f` fixed for the scale
itself. It is also the right semantics: the scale is learned through the forward
channel, while tau only shapes the estimator, and a speaker that could tune its
own gradient estimator would have every reason to soften it rather than to
communicate better.

The cost is straight-through bias: the surrogate stays softer than the hard
sample the listener actually receives, so the speaker differentiates a policy
slightly different from the one it plays. Straight-through is biased at any `tau`,
so this is a change of degree.

`training_progress` is a fraction of training elapsed, set once per epoch by
`train.py`. It is a *position in a schedule*, not state: recovered from the epoch
counter on resume rather than checkpointed, and 0.0 until told otherwise, so a
speaker used outside a training loop — ACRe, the tests, an interactive session —
samples at the configured `tau`.

## Scale the unmasked logits, then re-mask

Both decode loops do this:

```python
logits = mask_reserved_tokens(normalised * self.logit_scale)
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
the source of the `realised_survival` column.

By the Gumbel-max identity, the probability that a slot's argmax is unchanged by
the noise is exactly the winning token's softmax probability. So survival can be
read straight off a softmax: no Monte Carlo over noise draws, no assumed logit
distribution, and no seed. `tests/test_exploration.py` pins the identity.

It applies the real sampling pipeline in the real order — scale first, then the
uniform mixture — so that the mixture's bounds hold.

This is purely a measurement. It used to be the inner loop of a solve that chose
the scale to hit a requested rate; now the scale is the speaker's own learned
`logit_scale`, so what it reports is the joint result of that scale and the logit
*shape* it is applied to. The scale is passed detached — this is a diagnostic and
should not be on the graph.

Both speakers pool the measurement over positions once per batch rather than per
position, so it reads the batch's statistics rather than each position's alone.
The parallel arm can take it straight off its logits because it emits every
position in one shot; the GRU and the causal arm have to stack theirs.

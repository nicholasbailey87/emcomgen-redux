# Measurement

Only two things are measured in this codebase: **generalisation accuracy**, taken
straight off the listener's predictions in `train.py`, and **topographic
similarity** (topsim) — the Spearman correlation between pairwise distances in
meaning space and pairwise distances in signal (message) space.

Everything else `metrics.csv` carries is a diagnostic: a column that exists so
that a particular failure mode is a row rather than something inferred after the
fact.

## The topsim family (`code/emergence.py`)

Classic topsim uses a single signal distance, Levenshtein, which is sensitive to
both the order of the symbols and their identity. A language with free symbol
order, or one with synonyms, is therefore scored as non-compositional even when
it is perfectly compositional. This module implements topsim as a *family* of six
variants, one per signal set S1–S6, differing only in the signal distance
function:

| key | set | signal distance | characterised by |
| --- | --- | --- | --- |
| `topsim_s1` | S1 | soft MoverScore, 1-grams | free order + synonymy |
| `topsim_s2` | S2 | soft MoverScore, 2-grams | blockwise free order + synonymy |
| `topsim_s3` | S3 | soft Levenshtein | strict order + synonymy |
| `topsim_s4` | S4 | hard MoverScore, 1-grams | free order, no synonymy |
| `topsim_s5` | S5 | hard MoverScore, 2-grams | blockwise free order, no synonymy |
| `topsim_s6` | S6 | hard Levenshtein | strict order, no synonymy |

S6 is the classic topsim. **"Soft"** means synonymy-tolerant: the cost of
aligning two symbols is a function of the sender's own contextual symbol
embeddings (`Sender.speak`). **"Hard"** is the same function under a fixed,
embedding-free 0/1 ground cost, so only symbol identity matters.

Keys are named by *signal set*, not by distance function, because the set is what
the value licenses a claim about.

### Two meaning spaces

The meaning distance is held constant across all six within one reading, and
`topsim_report` takes two readings:

| prefix | meaning distance |
| --- | --- |
| `topsim_` | cosine between the sender's concept vectors (the third output of `Sender.speak`) — the chapter's semantic distance |
| `topsim_gt_` | word-level edit distance between the ground-truth logical forms — the concept distance of the original paper, so `topsim_gt_s6` is comparable to its reported rho |

The first asks whether the language tracks what the sender *represents*; the
second, whether it tracks the *latent variables the dataset was generated from*.
They come apart when the sender has collapsed onto a subset of the visual
features: a sender that has collapsed onto one visual feature scores well on the
cosine space for faithfully encoding that feature, and only the second reading
shows how much of the latent structure it left out.

That is a difference in *coverage*, not in compositionality, and `topsim_gt_` is
not the better of the two readings. See "Interpreting topsim" below before
drawing a conclusion from either.

`formula_distance_condensed` interns words to integers so the compiled
`rapidfuzz` path can be used; it is exact on any sequence of hashables. The
`topsim_gt_` family is omitted for datasets whose concepts have no logical form
(CUB, whose keys are bare class ids).

### The `_static` control

The three soft variants are additionally reported as `_static`: recomputed on
per-token mean embeddings.

The soft signal distances are parameterised by the sender's *contextual* symbol
embeddings, which differ from a fixed lookup table in two ways at once. They
tolerate synonymy, which is the point. They are also sensitive to the concept
being described, which is not: `SenderGRULM` initialises its hidden state from
the concept vector, and the SOS input is a constant, so the embedding behind the
*first* content symbol of every message is a function of the concept alone with
no token in it at all. A soft variant can therefore read a correlation straight
out of the meaning space without the language taking any part.

Every speaker here has that property in some form, so this is not a GRU-specific
control. `SenderTransformerLM`'s causal arm reaches the first content symbol with
nothing committed anywhere in the latent array, so everything it has is
prototype-derived and that position is a function of the concept alone in exactly
the same way. (It used to reach it with nothing in its sequence but SOS and the
utility tokens, which was the same statement about a different architecture — and
a good deal more literally true than anyone wanted; see anecdotes.md.) Its
parallel arm is the extreme case: *no* position depends on a sampled token, since
they are all emitted at once, so there every symbol's embedding is a function of
the concept alone.

Averaging every occurrence of a token id into one vector removes exactly that
second sensitivity and keeps the first. What survives is a non-contextual
embedding table learned from the sender's own usage: two tokens used for similar
meanings still sit close together, so synonymy is still detected, but no single
symbol carries the concept it was emitted for. **`raw − static` is the
contextuality inflation** — the thing a soft-versus-hard gap cannot by itself
rule out.

The hard variants need no such control; they see nothing but token identities, so
there is no leakage for them to suffer.

#### Two rejected alternatives

**An adjustment against the untrained model's topsim.** That baseline could not
bound the leakage, because the leakage is created by training: an untrained
`SenderGRULM` has a random `init_h` and a saturating tanh which between them
destroy the concept signal, so the baseline read ~0 for every variant while the
trained model did not.

**A permutation null** — reassigning messages between concepts. Permuting the
messages while each utterance keeps its own embeddings does not decouple form
from meaning, because the embeddings still encode the tokens they were originally
emitted with, and in a compositional language those track the concept. On a
perfectly compositional toy language such a control reports ~0.6 for S1 and S3
and would subtract away most of a true reading.

### Reference implementations and deliberate divergences

The MoverScore variants follow `moverscore.py` (v1) from
<https://github.com/AIPHES/emnlp19-moverscore> — the version with n-gram support
and `score = 1 − emd(...)`. We take the raw transport cost `emd(c1, c2, D)` as
our distance, i.e. the quantity that repo subtracts from 1. IDF weighting is
dropped (emergent languages violate Zipf's Law of Abbreviation), so the masses
are uniform over n-gram positions.

Two deliberate divergences:

1. **2-gram embeddings are concatenated, not summed.** `load_ngram` builds an
   n-gram vector as an IDF-weighted *sum* over the window. With IDF dropped that
   degenerates to a plain mean, which is order-blind *within* the n-gram: "AB"
   and "BA" would embed identically, collapsing S2 into S1 and destroying the
   exact distinction the variant exists to draw. We concatenate the two unit
   vectors in order and re-normalise instead, so the resulting cost is a monotone
   function of the mean cosine similarity of the aligned constituents.
2. **Zero ground cost on token match.** Symbol embeddings here are *contextual*,
   so two different games emitting the same token sequence would otherwise be at
   non-zero distance from each other, violating the Identity property topsim
   needs. Wherever the two n-gram ids match, the ground cost is overridden to 0.
   This applies to the soft Levenshtein substitution cost too, and it is what
   guarantees that soft is never more expensive than hard on the symbols the two
   languages agree about.

The reference's `_safe_divide` zero-guard on the masses would only rescale them
by a constant, and would rescale the two sides differently when the messages
differ in length, so the masses are normalised exactly instead.

The Levenshtein variants use `strsimpy.weighted_levenshtein` from
<https://github.com/luozhouyang/python-string-similarity> for the soft case, and
`rapidfuzz` for the hard case (an exact, compiled equivalent).

### Implementation notes

**`hard_mover_condensed` is computed in closed form.** Under a 0/1 ground cost
the optimal plan matches as much mass as possible at zero cost, so for normalised
n-gram histograms `p` and `q` the transport cost is `1 − Σ_g min(p_g, q_g)`,
which is exactly `0.5·||p − q||₁`. `tests/test_emergence.py` asserts this against
`ot.emd2`.

**`soft_levenshtein_condensed`** hands strsimpy `(utterance_index, position,
token_id)` tuples so that the cost callback can reach the contextual embedding
for that specific position. That defeats strsimpy's internal `s0i != s1j`
short-circuit — which is exactly why the token-id equality check has to live
inside our cost function. Note `1 − cos ≤ 2 = insertion + deletion`, so
substitution is never dominated by a delete/insert pair and the DP stays
well-formed.

**`hard_levenshtein_condensed`** reports raw edit counts, not length-normalised:
messages are fixed-length in this setup, so normalising by mean length is a no-op
that only obscures the metric. `tests/test_emergence.py` asserts equality against
the strsimpy DP the soft variant uses.

**NaN propagation.** EMD between an empty measure and anything is undefined, so a
message with no n-grams (shorter than n) propagates NaN rather than inventing a
distance. `spearman_topsim` drops NaN pairs pairwise and returns NaN if fewer
than two finite pairs remain or if either side is constant (Spearman is
undefined at zero variance).

### Message geometry in this codebase

Both senders mask the reserved tokens out of the content logits, so EOS never
fires early and every message is exactly `message_length − 2` content symbols.
Insertions and deletions therefore never arise, length normalisation is a no-op,
and a 5-symbol message has only 4 two-grams.

## Concept prototypes (`train.compute_language_metrics`)

Measurement is per **concept prototype**, not per image. One point per concept is
what the signal sets are about — a language is compositional to the extent that
the form of the utterance for a concept tracks that concept's meaning. Pairing
individual images would put many pairs at meaning distance zero (same concept,
different image) against a non-zero signal distance, which only depresses the
correlation; and the soft signal distances are O(n²) in a Python loop, so 2000
images is ~2M pairs.

A concept's prototype is:

- the **modal token sequence** its instances emitted;
- the **mean of the contextual symbol embeddings** of *every* instance that
  emitted that sequence (the soft distances need embeddings paired with real
  symbols, and averaging over the instances that emitted them rather than taking
  one arbitrary instance is what keeps a single epoch's reading stable);
- the **mean of its instances' concept vectors**.

Concepts are capped at `max_concepts`; capping images instead would starve each
concept of the instances the modal message is drawn from.

Spearman needs at least two pairs, i.e. at least three prototypes; below that the
report is empty.

Concept keys come from the ground-truth language: a CUB concept is its integer
class id, a ShapeWorld concept is the logical-form string with SOS/EOS stripped.

Topsim is computed on the eval passes only. It is a property of the language, so
there is no point computing it mid-training-pass, and the extra tensor is wasted
work. When it is computed, the sender is driven through `speak` so that message,
symbol embeddings and concepts all come from one forward pass.

## Interpreting topsim

The sections above are what the numbers *are*. This one is how to read them, and
most of it exists because it was got wrong first.

### Topsim is the definition of compositionality here, not an estimator of it

There is no separate ground truth about how compositional a language is that
topsim approximates well or badly. The measure *is* the claim.

Three things follow, and they are the ones that go wrong:

**It is not normative.** A language can be highly compositional and grounded in
one latent variable. You do not become more compositional by representing the
*correct* concepts. A sender that has collapsed onto colour and emits a
systematic colour language is compositional — over an impoverished semantics.

**`topsim_gt_` is not a gold standard.** Read it as *what topsim would look like
if the language encompassed all the latent variables*. It is coverage-conditioned,
not a paragon that `topsim_` is a degraded copy of. A run reporting `s6 = 0.454`
and `gt_s6 = 0.099` is reporting narrow coverage, not failed compositionality —
and on ShapeWorld you can show which by rebuilding the gt meaning space over a
filtered formula (below).

**`topsim_gt_` cannot be built for CUB, in principle.** ShapeWorld is synthetic,
so its latents *are* the generative variables and are known by construction. CUB
is natural images; the 312 attributes shipped with it are human annotations, not
intrinsic latents, and a topsim against them measures agreement with an
annotator. So `topsim_` on birds is not a second-best proxy for a missing gt
reading — there is no gt reading to miss. Birds comparisons work rung against
rung on the same meaning space, and never against an absolute.

The pairing that does work is `topsim_` for compositionality relative to what the
sender represents, and **accuracy** for how much of the task that representation
covers. Read jointly.

### `_static` is not a stand-in for `topsim_gt_`

They fix different things, on different sides of the correlation. `_static`
removes signal-side leakage from one meaning space; `topsim_gt_` changes the
meaning space. A near-zero `raw − static` gap says the soft reading is clean; it
says nothing at all about coverage.

The two are independent in practice, not just in principle. Across the ShapeWorld
rungs of the August 2026 ablation the `gt_s1 − gt_s1_static` gaps were +0.004,
+0.007, −0.003, +0.004, +0.001 and +0.009 while `gt_s1` itself was 0.087, 0.086,
0.108, 0.096, −0.001 and 0.066. Gap at zero, coverage at zero, no contradiction —
and the *tightest* gap in that set belongs to a run that never left `ln 2`.

### Read more than one column

There are 6 signal sets × 2 meaning spaces + 3 `_static` per meaning space = 18
per split, over `test` / `test_same` / `test_avg` — 54 columns on ShapeWorld, 27
on CUB. Reading `test_topsim_s1` alone is reading one of them.

**`test_avg` is `np.mean` of the `test` and `test_same` values**, not a third
eval pass, which is why it always sits between them.

**S4–S6 touch no embeddings at all.** Only S1–S3 read the sender's contextual
symbol embeddings, which is why only they get a `_static` control. "The topsim is
inflated by the embeddings" is a claim about S1–S3 and is simply false of S4–S6;
S6 is leak-free by construction.

**`test` is far noisier than `test_same`.** In the August 2026 ablation the
median epoch-to-epoch |Δ| on birds ran 0.09–0.20 on `test` against 0.037–0.059 on
`test_same`, over a range about 0.6 wide. A final-epoch topsim is a draw from
that spread, not a property of the run. Take a median over a window of epochs and
quote a spread; a difference of 0.05 between two rungs is noise.

### Topsim rises as the message set collapses

Always read topsim beside `unique_message_fraction`. Fewer distinct messages means
fewer distinct points in the signal space, and the correlation flatters itself.
Down the birds ladder of the August 2026 ablation, `test_same` unique-message
fraction fell 0.286 → 0.298 → 0.188 → 0.107 → 0.031 → 0.012 while static topsim
climbed 0.808 → 0.846. The rung at the top of that table was scoring 1.2% unique
messages. That is a smaller message set, not a better language.

The same effect runs the other way at the top of the range. Whole-message
statistics — topsim included, but mutual information especially — inflate when
nearly every message is unique, because the whole-message variable approaches an
identity function. One dead rung reported whole-message NMI of 0.371 with colour
while its per-slot NMI was 0.008.

**A NaN column is a symptom, not missing data.** Near-random messages sit at
almost the same distance from each other, so the signal distance vector has ~zero
variance and Spearman is undefined (see "NaN propagation" above). A run whose
topsim columns are NaN has a channel carrying noise.

### Topsim cannot see slot specialisation

A smooth *holistic* code — each whole message an unanalysable label, but similar
concepts getting similar labels — scores high topsim and has no positional
structure whatsoever. Topsim cannot separate the two, and neither can the
`_static` gap: a holistic code has perfectly context-independent symbols, so its
gap vanishes too.

The complement is per-slot mutual information: NMI between each message position
and each attribute. If the language is compositional, different positions carry
different attributes. If it is holistic, every position carries the same
information. In the August 2026 ablation every live rung was the latter —
ShapeWorld rung 1 gave colour NMI 0.332 / 0.294 / 0.273 / 0.251 / 0.229 across
its five content slots, and shape 0.162 / 0.134 / 0.122 / 0.107 / 0.096. Five
noisy copies of one message.

### Build a grounded topsim by filtering the formula

`formula_distance_condensed` takes any token sequence, so a gt meaning space
restricted to *one* latent is a filter on the formula and nothing else. That is
how "compositional but narrow" gets established rather than asserted.

Rung 1 of the August 2026 ablation, S6 against filtered ShapeWorld formulas:
full formula 0.068, colour terms only **0.512 ± 0.029**, colour plus `and`/`or`/
`not` 0.226, shape terms only −0.046. So `gt_s6 = 0.099` was reporting coverage,
the language was a real colour language, and adding the operators halved the
reading — it tracked *which* colours were in the concept and not how they were
combined. Content words, no syntax.

Two practical notes. All-pairs Spearman on 21,000 utterances is 220M pairs, so
subsample and report a spread over several draws rather than one number. And a
filtered formula is still the gt meaning space, so everything above about
coverage applies to it: a high colour-only rho is not a better result than a low
one, it is a different question.

## Diagnostic columns

### Speaker, on the train pass only

Logging the channel is how it stops being an invisible property of the
architecture.

**`realised_survival`** — what each speaker's logit shape and its learned
`logit_scale` buy it together. A finding rather than a restatement of a target.
Expected to climb over a run as the speaker grows confident, and bounded above by
`uniform_weight`'s mixture. Read it beside `logit_scale` and `logit_margin`,
which say which of the two routes the climb came from.

**`unmixed_survival`** is the same measurement with `uniform_weight`'s mixture
switched off, and it is the one to read for saturation. `realised_survival` is
bounded above by `(1 − w) + w / V`, which is **0.90714** at ShapeWorld's `w` 0.1
and `vocabulary` 14, so a channel that has committed entirely still reads as 0.91
and looks unremarkable next to a healthy 0.85.

The gap matters **on the `"gumbel"` branch**, where the estimator runs through
the *soft* sample, whose Jacobian is `diag(p) − p pᵀ`. That vanishes as `p → 1`,
so `1 − p` is the factor the estimator turns on — and the `p` in it is the one
*before* the mixture, because `flatten_logit_distribution` mixes in probability
space and so contributes only a constant. On
`shapeworld-post-silhouette-update.csv`,
`realised_survival` pinned at 0.90670, 99.95% of the way to the cap, which
inverts to an unmixed 0.99951. Read against 0.4779 at epoch 0, `p(1 − p)` fell by
a factor of 510, and `clip_sender_vision` went from 2.5e-2 to 1.9e-7 alongside
it. None of that was legible in the mixed column.

Expect the two to open close together and separate as the run sharpens. Watch
`1 − unmixed_survival` on a log scale; it is the quantity with the dynamic range.

**Which estimator is running changes what this column means.** Under
`estimator = "identity"` the Jacobian is `I`, so this reaches the gradient not at
all: it is still the channel's fidelity — how much of the message arrives — but a
run that saturates it is not thereby in trouble, and the saturation signature in
[training.md](training.md) cannot fire. Under `"gumbel"` it is a gradient
diagnostic as described, and bounded rather than unbounded: `MAX_LOGIT_SCALE`
and `sharpest_logit_margin` cap the two factors that saturate it, so this column
cannot exceed **0.9945** at V = 14. A run pinned *at* that is at the sharpest
the channel allows; a run pinned at `realised_survival`'s 0.90714 might be
anywhere.

**`logit_margin`** — the winning token's lead over the runner-up, in units of the
logits' own standard deviation, averaged over slots. Taken after
`layer_norm_logits` and before the gain, so it is invariant to `logit_scale` by
construction and comparable across vocabularies without rescaling.

This is the *shape* parameter that decides saturation, and it is one of the two
things that move the channel over a run — the scale is the other. The winner's
probability goes roughly as
`1 − p ≈ (V − 1)·exp(−logit_scale × logit_margin)`, so the scale and the margin
saturate the channel interchangeably, which is why both are logged and why
neither substitutes for the other. `layer_norm_logits` pins the second moment
while pinning nothing about the shape, so the margin is free up to `V/√(V−1)`,
and `MAX_LOGIT_SCALE` bounds the scale at 2.0; their product is the ceiling on
`unmixed_survival` above. This is the route the 2026-08-29 ShapeWorld run took:
`logit_scale` reached only
3.046, well inside what looks safe, but inverting the observed 0.99951 gives a
margin of about **3.35**, against the **0.44** that i.i.d. standard normal logits
over 14 tokens would give (`1/√(2 ln V)`, the Gumbel top-two spacing). Seven
times sharper than Gaussian: the speaker grew kurtosis, not volume.

Do not read `logit_spread` for this. That is the standard deviation of the *raw*
logits taken before the normaliser divides it out, so it reports the head's
output magnitude and the normaliser then discards exactly that. The two can rise
together, as they did on the run that died, and it still says nothing about the
margin: `logit_spread` would read the same for a head whose output is uniformly
larger and for one that has grown a spike.

**`logit_prior_share`** — the fraction of the normalised logits' variance that is
the same for *every input*, from the orthogonal split of the batch into a common
component and the residual that varies. It answers the question a large
`logit_margin` leaves open.

A concentrated distribution is not a failure. One whose peak *moves with the
input* is the best channel available — confident and informative. One that peaks
on the same token whatever the speaker saw is confidence with zero information,
and `realised_survival` and `logit_margin` read **identically** in the two cases.
This is the column that separates them. At 1.0 the speaker emits one message for
every game, which is where
`shapeworld-post-silhouette-update.csv` ended: `test_unique_message_count` of 1
across 1,000 games.

Two reference points, because 0 is not the healthy value. Logits with nothing in
common across inputs sit at about `1 / batch_size` — the mean of B independent
rows carries 1/B of their energy — so 0.031 at ShapeWorld's 32. But a *freshly
initialised* speaker measures around **0.55**, because `outputs2vocab.bias` sits
before the normaliser and survives it as a fixed pattern, and it is on the
weight-decay exclusion list. So the column opens high and the reading is which
way it moves, not its distance from zero. Read it against that bias.

**Why the shape columns exist at all.** `layer_norm_logits` fixes the second
moment, so the speaker has a bounded shape budget and the only question is what
it spends it on. The most concentrated distribution the normaliser permits is one
token at `sqrt(V-1)` and the other thirteen at `-1/sqrt(V-1)` — a margin of
**3.883** sd at V = 14, or `V/√(V−1)` in general. The 2026-08-29 run had used 86%
of that budget. **Fidelity does not have to come from the scale** — at a scale
of one this shape alone reaches 0.789 at V = 14 — which is why a run can traverse
the whole channel with `logit_scale` flat, and why a bound on the scale is not by
itself a bound on saturation.

**None of the four columns above survive `normalise_logits = false` as
comparable quantities.** They are all measured on post-norm logits — "in units
of the logits' own standard deviation" — and `sharpest_logit_margin`'s bound of
3.883 at V = 14 is a property of `layer_norm_logits` that does not exist without
it. They stay computable, read against a raw spread instead, and must not be
pooled across the key. `logit_spread` is taken *before* the norm and is
unaffected. See [channel.md](channel.md).

**`logit_scale`** is a live parameter again as of 2026-08-31, and it is the
primary read on whether the channel is earning anything. Under
`normalise_logits = false` it does not exist and this column reads NaN, as does
`train_clip_log_logit_scale`; the header keeps its shape either way, which is
the same convention the contrast columns follow. It opens at exactly
**1.0** on both datasets and is bounded above at **2.0** by `MAX_LOGIT_SCALE`;
there is no floor.

Three things to look for, in order:

- **Does it pin at 2.0, and how fast?** At `logit_scale_lr` = 2e-3 a
  sign-consistent gradient covers `ln 2` in 2.2 epochs, so the ceiling binding is
  expected and is not a fault — but it should not now happen before the warm-up
  is over. If it pins early and stays, the question is whether 2.0 is the wrong
  ceiling rather than 2e-3 the wrong rate — check `logit_margin` first, because a speaker already spending its shape
  budget does not need the scale as well.
- **Does it ever come back down?** A dip and a return is a speaker declining to
  commit while it has nothing to say, and it is the behaviour the projection
  exists to preserve — a `clamp` at the ceiling would have welded it. A monotone
  slide with no return is the failing-run signature: 0.9094 → 0.6547 on rung 10,
  0.8648 → 0.7784 on rung 9.
- **Is it dead flat?** That is now a finding rather than a check. Bit-identical
  across epochs means the same thing it means for `logit_spread` — AMP skipping
  every step. See below.

It was a constant, at 1.419 on ShapeWorld and 1.283 on birds, between 2026-08-30
and 2026-08-31. Runs from that window read flat by construction and are not
comparable with these; see [channel.md](channel.md).

**`logit_spread`** measures the emittable logits *before* `layer_norm_logits`,
which divides that magnitude straight back out, so it reports the head's output
magnitude rather than the channel's fidelity. Do not read it for saturation —
it would read the same for a head whose output grew uniformly and for one that
grew a spike, and both happened on the run that died. What it is still good for
is AMP: bit-identical spread across epochs is the tell for skipped steps.

**`sampling_tau`** is the configured `tau`, flat for the whole of any run: the
coupling in `17ae9f9` was retired in `3b3b857` and its machinery is gone. It carries no information, and is kept so the
metrics header is stable and so a run records what its `"gumbel"` surrogate was
shaped by. Under `estimator = "identity"` it does nothing at all.

**`pool_effective_examples`** — `1 / Σ p²` over the prototyper's attention
weights, so it reads in examples. It opens at the number of positive examples,
where the pooling is uniform and the prototype is exactly the mean, and falls
towards 1 as the pooler commits to particular images.

**`pool_score_norm`** — the scoring vector that carries that departure, opening
at exactly zero. Without the pair of them, "the pooler found something" and "the
pooler stayed at the mean" are the same row, and the difference between rung 3
and rung 1 would only be visible in a checkpoint. `pool_score_norm` is NaN under
`AveragePrototyper`, which has no scoring vector to report;
`pool_effective_examples` is the example count by construction there, since
averaging *is* pooling with uniform weights.

**`pool_score_sd`** — the standard deviation of the pre-softmax pooling scores
over the examples in a game, which is what `pool_effective_examples` is a
compressed function of. For a spread `σ` the effective count is about
`n / (1 + σ²)`, so the entire range from "faintly structured" to "exactly
uniform" is squashed into the last fraction of a percent below `n`. On the
2026-08-29 run, 9.86323 and 9.99613 differ by 1.3% in the count and by **six
times** in `σ` — 0.118 against 0.020 — and that was precisely the difference
between the collapse the speaker climbed back out of at epoch 17 and the one at
epoch 21 it never did. Keep both: the count is the interpretable column and this
is the one with resolution where the run is decided.

Recovered from the weights rather than recomputed, since `log w = s − logsumexp(s)`
makes the two standard deviations identical and nothing has to reach past
`SequencePool.attention_scores` into broccoli's `Sequential`. NaN under
`AveragePrototyper` for the same reason `pool_score_norm` is.

**`referent_spread` and `referent_spread_backbone`** — how much the referents
within one polarity still differ from each other, relative to what they share:
subtract each polarity's own mean, then take the RMS of the residual over the RMS
of the means. The same decomposition `contrast_within_share` uses, so the two are
read on one basis, and dimensionless, so a global rescale of the embeddings does
not move it.

They exist to break an ambiguity `pool_score_sd` cannot. A flat pooling score has
two causes that call for opposite fixes — the examples genuinely collapsed onto
one point, or the single scoring direction rotated somewhere they do not vary —
and these columns never touch the scoring vector, so a collapse here is the
backbone and a flat `pool_score_sd` alongside a healthy spread here is the pool.

`referent_spread_backbone` is taken on `embed_images`' output and `referent_spread`
after the contrast stage, so they bracket it. They are **equal** on a rung without
the stage, which is the informative reading and the reason neither is NaN there.
A gap between them is the contrast branch homogenising the referents, which is
live enough to be worth watching: `contrast_share` reached 0.32 on the 2026-08-29
run while `contrast_within_share` was 1.6e-4, a branch that was 99.98% a shared
vector plus a per-polarity offset. Adding a common vector lowers this ratio by
construction — that is what it is for.

**`polarity_separation`** — `norm(e_pos − e_neg)` for the Transformer speaker's
`polarity_embedding`. A constant added to both rows shifts every key and value
alike and cannot separate them, so the distance between the rows is the only part
of the tag the cross-attention can act on. Opens at `2 * sqrt(d_model)`: the tag
is an antipodal draw at the scale of the layer-normed prototype it is added to,
which on a 320-wide speaker is 35.8, where under the old zero init the column
opened at 0 and the question was whether the tag was ever used at all.

The trace to read it against is rung 10's, the one rung that both builds this
speaker and learns: 0.098 at epoch 0, crossing 1.0 at epoch 4, 6.1 by epoch 7
and 13.19 by epoch 29, with `train_acc` leaving chance in the same epoch the tag
crosses 1.0 and plateauing when it does. So a learning run wants a tag of order
10, and the antipodal init opens a factor of 2.7 above that rather than the
order of magnitude the dead rungs suggest. Do **not** use rungs 11 and 12's 0.16
to 0.79 as the reference: those runs never learned, so that is where a dead run
leaves the tag. NaN for `SenderGRULM`, which is
handed the distinction by `init_h` and has nothing to learn.

It is a *parameter* norm rather than a per-batch quantity, so it does not depend
on the pass at all; it is recorded inside the decode so every column stays on the
same clock.

**`contrast_gate`, `contrast_share`, `contrast_within_share`** — the speaker's
optional contrast stage, and all three are NaN when `[sender] contrast` is false.
NaN rather than absent because the header has to be the same shape either way or
a run cannot be resumed against a config that toggles the flag; and NaN rather
than zero because "the stage is not there" and "the stage is there and has not
opened" are different rows.

They divide the way the speaker's other columns do, into volume and shape:

| | did it open | how loud | is it doing anything new |
|---|---|---|---|
| `ExampleContrast` | `contrast_gate` | `contrast_share` | `contrast_within_share` |

**`contrast_gate`** is the scalar standing between the branch and the identity.
It opens at exactly zero and takes `contrast_gate_lr`, so as a lone scalar it
cannot travel further than `lr × steps` — 0.1 an epoch at 2e-3 and birds' 62
steps, against 0.0062 at the base rate, which is why it has its own key. A
parameter rather than a per-batch quantity, so it does not depend on the pass.

**`contrast_share`** is `rms(gate × branch) / rms(referents)`: what fraction of
the referent going into the prototyper is contrast. This is the reportable
column, and it is what the gate alone cannot give — the gate's meaning depends on
whatever magnitude `out_projection` happens to emit, and that is neither
configured nor pinned.

**`contrast_within_share`** is the part of the branch that is example-level.
Within each game the branch decomposes by nested means into a single vector
common to all `2n`, a per-polarity offset, and a per-example remainder; the three
are orthogonal, so their sums of squares add to the total, and this is the
remainder's share. It is measured on the branch *before* the gate, so a
well-shaped branch that is still quiet reads as well-shaped — the only time
anyone needs the column is early, when the gate is small.

It exists because a large `contrast_share` is not evidence of contrast. A vector
common to the whole game shifts both prototypes equally and the language model's
`LayerNorm` eats most of it; a per-polarity offset is a learned "I am positive",
which `AttentionPrototyper`'s two separate pools already provide. Only the
remainder is contrast *between examples*, which is the entire reason the stage is
there.

| `contrast_share` | `contrast_within_share` | reading |
|---|---|---|
| ~0 | any | the gate never opened; the arm is its parent rung and says nothing about contrast |
| large | ~0 | a null result dressed up as a departure — the branch is a polarity tag or a global shift |
| small | large | the stage found something example-level but is not being trusted with the decision |
| large | large | the referents are genuinely being read against each other, which is the arm working |

Read all three against `topsim` and not against `acc`. The stage makes messages
distractor-dependent by construction, so a run where the share is large and
`topsim` has fallen is the cost showing up, not a bug — see
[architecture.md](architecture.md).

### Listener, on the train pass only

The listener is two slots — a language model and a discriminator — and it is the
**discriminator** these columns belong to. Dispatch is on its class rather than
by `hasattr`, for the same reason the pooling columns are: a fallback would turn
a rename into a silently-NaN column, and a silently-NaN column is how the
cross-attention listener's collapse went unnoticed for a whole smoke test. A
discriminator with no branch here raises rather than running unmeasured.

Both read out the same way — `ScoreVolume`'s `log_score_scale` and `score_bias`,
a volume then an offset, in front of a score whose two operands are
layer-normed — and differ only in what else they have to report:

| | volume | shape | mix |
|---|---|---|---|
| `BilinearDiscriminator` | `score_scale`, `score_bias`, `bilinear_weight_norm` | — | — |
| `AttentionDiscriminator` | `score_scale`, `score_bias`, `decision_spread`, `bilinear_weight_norm`, `decision_weight_norm` | `decision_kurtosis` | `mix_alpha`, `mix_share`, `path_agreement` |

**`score_scale`** — both classes, one per discriminator.
`AttentionDiscriminator` composes a `BilinearDiscriminator` built with
`score_scale=False`, because that branch is multiplied by `1 − mix_weight` and
read out through the module's own scalar, so a scale on it would say what
`mix_logit` already says and the two would drift against each other.

Under `[receiver_discriminator] normalise_score = false` there is no readout at
all, and `score_scale`, `score_bias` and `train_clip_log_score_scale` read NaN.
The operand norms and the `1/√d` go with it, so on that arm the paragraph below
about a score calibrated to open at 0.577 does not apply and
`bilinear_weight_norm` is the only volume column left. See
[architecture.md](architecture.md).

On the bilinear arm it multiplies a score calibrated to open at `1/√3` = 0.577
at every width and under every backbone, so the column is comparable across
rungs without further arithmetic. On the attention arm it multiplies a mix whose
opening depends on what `decision` emits — a fixed number per architecture, but
not that one.

**`score_bias`** — both classes, one per discriminator, and gated by the same
`score_scale=False` that withholds the volume from the composed bilinear path.
The offset half of the readout: `train.py` decides on `lis_scores > 0`, so this
is the parameter that places the scores against that fixed origin.

Read it *against* `train_acc`, and not as bigger-is-better. Games are balanced
10 positive / 10 negative, so the loss-optimal global offset is about zero, and
a column that sits there is the healthy reading — the scores are already where
the threshold assumes. A sustained drift is a finding: it says they are not.
A scalar cannot reach a *per-game* offset, the bilinear score's per-game mean
being `mean_j(LN(r_j)) · proj`, so a bias that moves while accuracy does not
means the offset was per-game and the readout is what needs changing.

It replaced `AttentionDiscriminator.mix_bias`, which had neither a column nor a
config key — it sat at the base `lr`, which at today's 5e-5 and 156.25 steps an
epoch would bound its entire thirty-epoch travel at 0.23 against a score opening
at 0.577 spread.
An unmeasured parameter is how the readout's collapse went unnoticed the first
time; this is the correction.

`score_scale` says how confidently the listener acts on a message. It used to
have a counterpart on the speaker — a learned `logit_scale`, saying how audibly
the speaker stated one — and the two were read together, both dipping during
bootstrapping for the same reason: neither agent should commit while the message
is still noise. The speaker's half is a constant now, so only this one dips.
`29b18ea` measured the separation that tells a productive dip from a collapse: a
healthy speaker fell ~0.2 log-units and returned within a few epochs, where the
arm that died fell 0.94 and never did. There is deliberately no floor on either;
`e3fcabd` tried one and it cost fifteen epochs.

**Read a slide as a slide.** A monotone descent is alarming because the listener
is turning itself down and not turning back up, which on this ladder has meant a
pair that never found anything to say. What it does *not* additionally mean is
that the speaker is being starved: the scalar does multiply every gradient going
back through the message and the channel, but a uniform factor on a parameter's
gradient cancels in AdamW's `m / √v`, and `clip_gradients` renormalises each
submodule on top of that. Two rounds of design went into removing that coupling
before it was measured — see [anecdotes.md](anecdotes.md), round seven.

**`bilinear_weight_norm`, `decision_weight_norm`** — the branch weights' norms,
and volume columns on both arms. Nothing downstream divides a rescaling of
either back out, so a weight that grows is a listener getting louder, exactly as
`score_scale` growing is.

The two say the same thing at different speeds, and that is the point rather
than a redundancy. A 320×320 matrix under Adam spends its step turning and only
a fraction of it radially: between `a9a6a9c` and `7b10d47`, when the matrices
held the volume alone, `bilinear_weight_norm` travelled 1.3% of its norm in
thirty epochs on rung 09 and 0.6% on rung 10, against the 59% the scalar it had
replaced managed. Read the scalar for what the listener is doing now and the
norms for where it has settled.

On `AttentionDiscriminator` the branches also mix at their own magnitudes, so
there the norms additionally set what the score is made of, and `mix_share`
against `mix_alpha` is where that shows.

**`mix_alpha`** — `AttentionDiscriminator` only, and the column the whole split
exists to produce. The attention path's share of the score:

```
score = score_scale * ( (1 - a) * bilinear + a * attention ) + score_bias
a     = mix_floor + (1 - mix_floor) * sigmoid(mix_logit)
```

Bounded in `[mix_floor, 1)`, so it reads directly as *was attention used* —
which is the chapter's question, and one no previous column could answer. It
opens at 0.116 (floor 0.1, logit −4.0), essentially at the bilinear comparison,
and departs only if attention earns it. `mix_logit_lr` is elevated to 2e-3 so
the traverse is affordable inside a run; see DEFAULT.toml for that arithmetic.

**`path_agreement`** — same class, and it must be read *with* `mix_alpha`, never
instead of it. The within-game correlation between the two paths' standardised
scores. Both operands are already per-game zero-mean and unit-spread, so the
mean of their product is Pearson's r exactly.

The reason it exists: an attention path that is never used and one that has
learned to imitate the bilinear path look identical from accuracy and from
`mix_alpha` alone, and they are different findings. The prototype's pinned arm
ended at 0.811, so this is not hypothetical. Read the four combinations:

| `mix_alpha` | `path_agreement` | reading |
|---|---|---|
| at the floor | high | attention imitates the bilinear path; it is not being used and has nothing else to say |
| at the floor | low | attention is saying something different and losing the argument |
| climbing | high | attention is being taken up but has not yet found anything distinct |
| climbing | low | attention is being taken up *for* something the bilinear path cannot express — the outcome the design is aimed at |

**`mix_share`** — `AttentionDiscriminator` only, and the column to read *beside*
`mix_alpha` rather than instead of it. `mix_alpha` is the weight `mix_logit`
asked for; this is the share the score is actually made of, measured from the
two branches standardised per game — in the telemetry block, under `no_grad`,
which is the only place `standardise` still runs. They agreed only while
`forward` standardised each branch, and it deliberately does not: that would
make `mix_alpha` mean composition exactly, at the cost of closing the escape of
turning one branch down rather than learning it. A gap between the two is a loud
or quiet branch.

**`decision_spread`** — `AttentionDiscriminator` only. The standard deviation of
the returned scores, and independent of `score_scale` again now that the readout
does not pin it: it is the scale times whatever spread the mixed branches
actually have. `score_scale` alone reads the parameter; this reads the parameter
and what the module is doing with it together. It was briefly redundant, while
the readout standardised and the two measured the same thing.

A **monotone descent towards zero** is the finding: BCE reduces a loss it cannot
otherwise reduce by becoming less confident, and nothing in this readout stops it.
Wandering is not that. Nothing in the loss rewards the magnitude in either
direction on a run that is learning; rung 10 carries the identical exposure and
its `score_scale` falls 0.856 → 0.238 across thirty epochs while `train_acc`
climbs. Sign-consistent descent alongside a flat `train_acc` is what to act on.

Note the decision boundary is `scores = 0` and `train.py` reads `lis_scores > 0`,
so accuracy is invariant to any positive rescale of the readout. That is why the
accuracy column could not see the original collapse and still cannot.

**`decision_kurtosis`** — same class, and the one to read first among the shape
columns. Excess kurtosis
of the scores. Negative means bimodal, which is what discriminating looks like
(−2 is the two-point floor); sustained positive alongside `train_acc` at 0.5 is a
listener with nothing to say.

Measured against this module on a synthetic game — the listener handed a message
that names the target concept, versus a scrambled one, everything else identical:

```
informative   acc 1.000   loss 0.127   excess kurtosis  −2.0
scrambled     acc ~0.50   loss ~0.9    excess kurtosis  +11..+23
```

`decision_spread` read 2.7–5.1 and 1.4–2.1 across those same two runs —
overlapping, and so unable to tell them apart on its own.

It is NaN when the scores have collapsed to a constant, where the fourth
standardised moment is 0/0. NaN is the honest value there: the shape of a point
mass is not defined, and a silent 0.0 would read as "Gaussian, nothing to see" at
exactly the moment there is something to see. `decision_spread` is the column
that names that state.

The routes there are `score_scale` and the branch weights alike, now that the
readout does not renormalise: anything that drives the mixed score to a constant
across candidates lands here. A constant attention readout does not, on its own
— the mix still carries the bilinear path, which keeps discriminating.

The column outlived the design it was built for; see [anecdotes.md](anecdotes.md).

The `.item()` calls in `forward` cost a sync and a graph break under
`torch.compile`, which is on. Paid deliberately: `AttentionPrototyper` already
reports `pool_effective_examples` exactly this way, and a metric nobody can read
is how the last collapse ran for a whole smoke test.

### Gradient norms, on the train pass only

**`train_clip_<group>`** — the gradient norm of one clip group, taken *before*
clipping, for every group in `models.builder.GROUP_NAMES`. Twelve named groups —
eight modules and the four scaling scalars — plus `other`. See
[training.md](training.md#the-group-table) for what is in each.

Averaged **per optimiser step**, not per example: a gradient norm is a property
of the step, and the trailing partial accumulation counts once like any other.
`nan` means the group does not exist on this architecture, or exists and holds
nothing with a gradient — `sender_contrast` on a rung with the stage off,
`mix_logit` on a `BilinearDiscriminator`, `sender_prototyper` under
`AveragePrototyper`, which has no parameters at all. Every column is present on
every rung so that the header survives a resume, exactly as the contrast columns
are.

**`train_clip_other` should be `nan`.** It is the catch-all that keeps a module
nobody registered from going unclipped, so a number there is not a reading — it
is a report that something was built without being added to `MODULE_GROUPS`, and
whatever it is has been clipped under a name that says nothing about it. That
was the state of the repo until August 2026, when `other` was silently the whole
`sender.contrast` stage.

**What these are and are not for.** They say how hard clipping is biting on each
group: a norm far above `clip_grad_norm` means that group is renormalised on
every step, and one below it means clipping never touched it. That is worth
knowing about the scaling scalars in particular, which is most of why they were
given groups of their own — the reasoning is in
[training.md](training.md#the-group-table), and these columns are the first
measurement of it, since no run before August 2026 recorded a gradient norm at
all. `clip_gradients` had built them and documented them "for logging" since it
was written, and the call site discarded the return.

They are **not** evidence about a learning rate. AdamW updates by `m̂/√v̂`, so a
uniform rescale of a group's gradients cancels before it becomes a step; a large
norm does not mean the group is learning too fast, and a small one does not mean
it is starved. Read them for the clipping and nothing else.

Two shapes worth watching. A group whose norm climbs steadily while others stay
flat is being renormalised harder over time, which changes the *ratio* between it
and the rest — clipping per group removes the cross-module noise coupling but not
this. And a norm that collapses towards zero on the speaker side while the
listener's holds is the signature the per-module clipping was introduced for: the
speaker's vision model used to take a coefficient set by the listener's comparer,
which supplies ~90% of the pair's squared norm at init.

### Per-epoch, all splits

**`unique_message_fraction`** — how much the language compresses. A speaker that
has learned to say something reuses messages across instances of a concept; one
whose channel is too noisy to learn through emits a near-unique message every
game, so this reads close to 1.0 while accuracy sits at chance.

**`unique_message_count`** — the same quantity before dividing, because the
fraction cannot be read without knowing the split's size and the splits differ by
a factor of twenty. On the 2026-08-29 run `test_unique_message_fraction` 0.001
was one message over 1,000 eval games, i.e. a constant policy; the fraction alone
does not say so. Note also that a small fraction on the *train* pass is not
independent evidence of anything, because the train pass samples through the
Gumbel noise while eval is greedy (`sender.sample_symbols`, line 814): 0.0551
over 20,000 games is 1,102 distinct messages, and a single underlying message at
`realised_survival` 0.9067 over 5 content slots predicts 1,108 — 61% clean, all
65 single substitutions seen, about 908 of the 1,690 doubles, 134 triples. Read
the eval column for the policy and the train column only against that prediction.

**`shuffled_message_acc`** (eval splits only) — the listener's accuracy on the
same candidates when the message is another game's, so the message is in
distribution and carries no information about this game. The gap between `acc`
and this is what the channel is buying; the level of this is the image-only route
the channel has to beat before it earns any gradient at all.

Both halves have bitten. On the 2026-08-29 ShapeWorld run the listener reached
0.588 train and 0.55 `test_same` on shape concepts *while the speaker emitted one
message for every game*, so most of what `acc` reported was never communication —
and there was no way to see that until the speaker had already died, which is too
late to act on. Expect it near chance early. A rise in this column with `acc`
flat is the listener moving its capacity off the message path, which is the
leading indicator of the speaker's gradient collapsing.

Rolled by one along the batch rather than permuted: a permutation leaves about
one game in `batch_size` holding its own message, biasing the baseline up by ~3%
at 32. Eval only, because it costs a second receiver forward pass and the train
pass is where the compute is. Reported as a single scalar, not broken down by
`md`.

**`acc_md_*`** (ShapeWorld only) — accuracy broken down by the metadata class of
the concept: `shape`, `color`, `and_color_shape`, and so on.

**`test_avg_*`** — the mean of the `test` and `test_same` readings of every eval
metric.

**`timestamp`** — wall-clock time the epoch finished, matching vit's `timestamp`
column, so a run's pace can be read off `metrics.csv` after the fact.

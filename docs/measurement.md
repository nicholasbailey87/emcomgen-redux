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
second, whether it tracks the *concepts*. They come apart when the sender has
collapsed onto a subset of the visual features, and only the second notices: a
sender that has collapsed onto one visual feature scores well on the cosine space
for faithfully encoding that feature.

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
control. `SenderTransformerLM`'s decoder arm reaches the first content symbol
with nothing in its sequence but SOS and the utility tokens, and everything else
it has is the prototype-derived memory, so that position is a function of the
concept alone in exactly the same way. Its parallel arm is the extreme case: *no*
position depends on a sampled token, since they are all emitted at once, so there
every symbol's embedding is a function of the concept alone.

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

## Diagnostic columns

### Speaker, on the train pass only

Logging the channel is how it stops being an invisible property of the
architecture.

**`realised_survival`** — at a fixed `logit_scale`, this reports what each
speaker's own logit *shape* buys it, so it is a finding rather than a restatement
of a target. Expected to climb over a run as the speaker grows confident.

**`logit_scale` and `logit_spread`** are read together with it, and say which of
two things a falling survival means: a flatter policy, which is the speaker's own
doing, or a collapsing scale. They are indistinguishable in survival alone.

`logit_scale` is the one that answers it now. `logit_spread` measures the
emittable logits *before* `layer_norm_logits`, which divides that magnitude
straight back out, so since `87c1027` split sharpness off into its own parameter
the spread reports the shape's size rather than the channel's. It remains the
column that makes a scale collapse visible, and bit-identical spread across
epochs is the tell for skipped AMP steps.

`logit_scale` is also the number `logit_scale_lr` exists to move, and the
prediction is quantitative: birds opens at 0.839 and the scale cannot travel
faster than `lr × steps`, so at 2e-3 over 156 steps an epoch its ceiling is 0.31
log-units per epoch. Observed travel against that ceiling reads off directly how
sign-consistent the gradient is, which is not recoverable from when accuracy
happens to move.

**`sampling_tau`** is logged because the coupling in `17ae9f9` and its retirement
in `3b3b857` are both untested. It is a function of the scale and of
`training_progress` alone, so it carries no new information in principle — but it
is the quantity that sets how much straight-through bias the run is paying, and
reconstructing it after the fact from two other columns and a cosine is the sort
of thing nobody does.

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

### Listener, on the train pass only

The listener is two slots — a language model and a discriminator — and it is the
**discriminator** these columns belong to. Dispatch is on its class rather than
by `hasattr`, for the same reason the pooling columns are: a fallback would turn
a rename into a silently-NaN column, and a silently-NaN column is how the
cross-attention listener's collapse went unnoticed for a whole smoke test. A
discriminator with no branch here raises rather than running unmeasured.

The two answer with different columns because they no longer have the same
mechanism:

| | volume | shape | mix |
|---|---|---|---|
| `BilinearDiscriminator` | `score_scale` | — | — |
| `AttentionDiscriminator` | `mix_scale`, `decision_spread` | `decision_kurtosis` | `mix_alpha`, `path_agreement` |

**`score_scale`** — `BilinearDiscriminator` only, and only when it is the whole
discriminator: `AttentionDiscriminator` owns one for its bilinear path but
builds it with `score_scale=False`, because `standardise` runs on that path's
output and divides any positive scale straight back out. `logit_scale` says how audibly
the speaker states a message; this says how confidently the listener acts on one.
Both dip during bootstrapping for the same reason — neither agent should commit
while the message is still noise — and `29b18ea` measured the separation that
tells a productive dip from a collapse: a healthy speaker fell ~0.2 log-units and
returned within a few epochs, where the arm that died fell 0.94 and never did.
There is deliberately no floor on either; `e3fcabd` tried one and it cost fifteen
epochs.

**`mix_alpha`** — `AttentionDiscriminator` only, and the column the whole split
exists to produce. The attention path's share of the score:

```
score = mix_scale * [ (1 - a) * bilinear_hat + a * attention_hat ] + bias
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

**`mix_scale`** — `AttentionDiscriminator` only, and its counterpart of
`score_scale`. Both mixed paths are standardised, so this one scalar is the only
thing that sets the score's magnitude. Read it exactly as `score_scale` is read:
a dip while the message is still noise is a pair correctly refusing to commit; a
monotone descent that never returns is the collapse. No floor, for the reason
given above.

Note what the standardising does and does not close. Neither path can go quiet
on its own, so the attention stack cannot escape being learned by turning itself
down. The *pair* can still go quiet, through this scalar — which is the freedom
the bootstrap needs, and the freedom a fixed gain took away when it stopped four
rungs learning at all.

**`decision_spread`** — `AttentionDiscriminator` only. Simply the standard
deviation of the returned scores, which is `mix_scale` times the spread of the
mixed standardised paths — so it is `mix_scale` read through the mix rather than
off the parameter, and the two moving apart means the two paths have started to
cancel. It opened around 0.57 under the four-stage structure this module
replaced; the current arrangement opens near 1.0 by construction, since both
mixed operands have unit spread per game and `mix_scale` opens at 1.0.

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

Note it now takes *both* paths going flat, or `mix_scale` going to zero, to get
there: a constant attention readout standardises to zero and the mix falls back
on the bilinear path, which keeps discriminating.

The column outlived the design it was built for; see [anecdotes.md](anecdotes.md).

The `.item()` calls in `forward` cost a sync and a graph break under
`torch.compile`, which is on. Paid deliberately: `AttentionPrototyper` already
reports `pool_effective_examples` exactly this way, and a metric nobody can read
is how the last collapse ran for a whole smoke test.

### Per-epoch, all splits

**`unique_message_fraction`** — how much the language compresses. A speaker that
has learned to say something reuses messages across instances of a concept; one
whose channel is too noisy to learn through emits a near-unique message every
game, so this reads close to 1.0 while accuracy sits at chance.

**`acc_md_*`** (ShapeWorld only) — accuracy broken down by the metadata class of
the concept: `shape`, `color`, `and_color_shape`, and so on.

**`test_avg_*`** — the mean of the `test` and `test_same` readings of every eval
metric.

**`timestamp`** — wall-clock time the epoch finished, matching vit's `timestamp`
column, so a run's pace can be read off `metrics.csv` after the fact.

# silhouette_titration

**What rate of receiver silhouetting does the CNN/GRU baseline need to escape
the colour-only minimum, without losing colour instead?**

Four copies of the ablation's rung 1 — ShapeWorld, `ResNet18SmallInput` on both
agents, `AveragePrototyper`, `SenderGRULM`, `ReceiverGRULM`,
`BilinearDiscriminator` — differing in `[data] silhouette_p_receiver` alone, at
0.0, 0.1, 0.2 and 0.3. Everything else comes from `DEFAULT.toml`. 100 epochs,
**three seeds each** — 12 jobs.

The sweep used to run to 0.5, and a white-fill arm sat alongside it at that rate.
Both were deleted on 2026-09-08; see **Why the sweep stops at 0.3**.

```
scripts/run_experiment.sh silhouette_titration 6
```

Without `--rerun` that submits only the incomplete jobs, and completeness is
`metrics.csv` with at least `[scheduler] epochs` rows. Everything on disk for
this experiment predates both the 2026-09-01 transform change and the move to
100 epochs, so every one of the 12 jobs is incomplete and all 12 go. Jobs are
enumerated repeat-major — indices 0–3 are the four rates at seed 0, 4–7 at seed
1, 8–11 at seed 2 — so an array cut short still leaves a complete sweep at the
seeds that finished.

## Why the rate is not already known

`silhouette_p_receiver` repaints the listener's whole view as flat single-colour
silhouettes with this probability, per game, at training time only. It exists to
break the colour-only local minimum Mu & Goodman report (~83% accuracy, appendix
A.1): with the six colours sitting at six distinct luma values a grayscale
conversion would re-encode colour as a scalar rather than remove it, where a flat
repaint removes it outright. See `DEFAULT.toml`'s block on the key for the
mechanism and for why `silhouette_fill` is (149, 149, 106) of 255.

**All five completed runs below predate the 2026-09-01 changes and were run under
the leaky transform** — a flat 0.5 fill, which collided with `gray` so grey
objects passed through untouched, and whose rounding lattice left colour
recoverable from the anti-aliased edges at Kendall tau +0.90. Do not pool them
with anything run after.

Two things moved on 2026-09-01 and neither is the rate. The fill became
(149, 149, 106), and the transform went back to a threshold at half the image's
peak luma rather than blending by coverage — the second on the strength of
`diagnostics/silhouette_shape_probe.py`, which measured that shape learned off a
coverage edge does not transfer to the clean images eval uses. Both change what a
given rate *does*, so this titration wants re-running from scratch under the new
transform rather than extending. See docs/data.md.

Three settings are on the record, and no two of them share an architecture:

| rate | architecture | outcome |
|---|---|---|
| 0.5 | rung 9, Transformer speaker | shape 0.758 and `and_shape_shape` 0.829 — the best on record — with **colour at chance for all thirty epochs** |
| 0.0 | rung 9, Transformer speaker | aggregate 0.661 by epoch 5, colour 0.747 *and* shape 0.646, both still climbing |
| 0.1 | rung 3, ViT speaker | the shortcut back: colour 0.794 against shape 0.509 |

Each of those moved the architecture and the rate together, so none of them
measures the rate. This experiment holds the architecture still.

The answer wanted is **the largest rate at which shape rises without colour
falling**, and it is a number the whole ladder then inherits from
`DEFAULT.toml`.

## Why the baseline agents and not a ViT

Silhouetting suppresses the *channel* as well as the colour feature. At 0.5 half
the listener's games contain nothing worth decoding: `unmixed_survival` sat at
~0.28 against ~0.45 at 0.0, with `logit_margin` ~0.40 against ~0.68. A speaker
whose messages are useless half the time gets a correspondingly weaker gradient
to sharpen on.

That is a second failure mode, and it is one the Transformer arms are separately
prone to — rung 3 had only just escaped a channel collapse when it was last
titrated on, which is exactly the wrong place to read this from. The CNN/GRU
baseline is the simplest thing on the ladder and the architecture whose failure
mode the paper actually documents, so a shape/colour trade measured here is
attributable to the key.

Whether the answer transfers up the ladder is a second question. It is the
ladder's to answer, and the honest expectation is that a ViT speaker wants a
different number.

## Receiver only

`silhouette_p_sender` stays at `DEFAULT.toml`'s 0.0 in all four arms.
Silhouetting the speaker removes colour from what it can *say*; this removes
colour from what the listener can *use*, which is what makes a colour-only
message stop paying. A config that moved both at once was written on
2026-08-31 and is retired into this experiment: two knobs would leave a
difference unattributable to either.

## Three seeds per rate

Every arm runs at seeds 0, 1 and 2, and the sweep is read across seeds before it
is read across rates.

The reason is a pair of runs under the current defaults: 0.2 came out better than
anything the learning-rate sweep reached, and 0.3 — one step along this very axis
— came out poorly. Two readings, one seed each, and no way to tell an effect of
the rate from the spread at a fixed rate. That is the whole experiment's failure
mode in miniature: a titration read off single runs measures the seed as much as
the key.

The failure mode is on the record elsewhere too. Mu & Goodman
(`emergent_generalization_neurips21.tex`, appendix) describe concept-game
training as "less stable" and, in some channel regimes, "often unable to learn";
they plot each run as its own dot and report every number over 5 runs. What they
do not say is whether those 5 are all the runs they launched or 5 that worked, so
treat the paper as a reason to expect run-level failure here — not as a
precedent for dropping a seed that fails.

Three is the smallest count that separates the two questions, and it is not
enough for a significance test. Nothing from this experiment should be reported
as one.

**The seeds go where they can change the answer.** All twelve runs sit on
0.0–0.3; see the next section for why nothing above that is run at all.

**Reading across the three.** Compare rates only where the gap between them
exceeds the scatter within each. A rate whose three seeds disagree with each
other is a rate this experiment has nothing to say about — and that is itself a
finding, because it means the outcome at that setting is not determined by the
setting.

**Do not average first.** Take each seed's `test_acc_md_shape` /
`test_acc_md_color` pair on its own and classify it by the rules below, then
count how many of the three landed where. A mean over one colour-only run and one
shape-only run looks exactly like a language that learned both, which is the
same confusion the aggregate/breakdown rule guards against one level down.

## How to read the result

**On the `test_acc_md_shape` / `test_acc_md_color` breakdown, never on the
aggregate.** The aggregate reads the same for a language with one feature as for
one with both, and that is exactly how the 0.5 result hid a bimodal outcome for
a week.

- **Shape rising, colour holding** — the result the key exists for. Take the
  largest such rate.
- **Shape rising, colour falling to chance** — the mirror-image failure. One
  single-feature language traded for the other. This rate is too strong.
- **Both flat, and `logit_prior_share` climbing towards 1.0** — the key has
  suppressed the channel rather than the feature. Cross-check
  `unmixed_survival`: if it is well under ~0.45 the listener is being handed
  games with nothing in them, and the answer is not more silhouetting.

Eval is never silhouetted, so a colour accuracy of 0.50 at eval is a failure to
communicate colour, not a ceiling this key imposes.

Expect early epochs near chance on every arm, but not for the reason previously
noted here: `warm_up_epochs` has been 0 since the ramp was removed, so nothing in
the schedule holds the first ten epochs down.

## Why the sweep stops at 0.3

There is no 0.4 or 0.5 arm, and no white-fill arm. Both were deleted on
2026-09-08 when the sweep went to three seeds.

The top of the axis is not an open question. 0.5 is already on the record as the
mirror-image failure: under the chromatic fill it ended its last five epochs at
shape 0.606 against colour 0.528, having crossed over around epoch 26 from shape
~0.55 and colour ~0.65. One single-feature language traded for the other, which
is the outcome the rules above say is too strong a rate.

It is also the end of the axis where the seed spread should be widest. More
silhouetting means more of the listener's games carry nothing worth decoding —
`unmixed_survival` ~0.28 at 0.5 against ~0.45 at 0.0, `logit_margin` ~0.40
against ~0.68 — so a weaker, noisier gradient and more run-to-run variation.
Seeds spent up there would buy a wide spread around a conclusion already reached,
which is the opposite of what the repeats are for.

What is open is whether the usable rate is 0.2 or something near it: under the
current defaults 0.2 produced the best run on record and 0.3 a poor one, and a
single run at each cannot separate that from seed noise. Every job in the sweep
now sits on that question.

If these four arms make 0.3 look like the boundary rather than 0.2, 0.4 is the
arm to write back — as three seeds, not one. The white-fill question (whether the
0.5 crossover was the fill or the rate) only becomes live again if the sweep ever
returns to that end of the axis; `diagnostics/silhouette_shape_probe.py` cannot
settle it, having failed to repeat its own reading across fills (0.560 then 0.403
on the same arm, same seed, same GPU — jobs 123354 and 123583 — because cuDNN's
convolution backward accumulates with atomics).

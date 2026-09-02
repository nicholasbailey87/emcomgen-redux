# silhouette_titration

**What rate of receiver silhouetting does the CNN/GRU baseline need to escape
the colour-only minimum, without losing colour instead?**

Five copies of the ablation's rung 1 — ShapeWorld, `ResNet18SmallInput` on both
agents, `AveragePrototyper`, `SenderGRULM`, `ReceiverGRULM`,
`BilinearDiscriminator` — differing in `[data] silhouette_p_receiver` alone, at
0.1, 0.2, 0.3, 0.4 and 0.5. Everything else comes from `DEFAULT.toml`. Thirty
epochs, one seed each.

A sixth config, `06_silhouette_0.5_white.toml`, sits in the folder and is **not**
part of that sweep: it holds the rate at 0.5 and moves `silhouette_fill` to white
instead. It is read against `05_silhouette_0.5` and against nothing else here.
See **The white-fill arm** below.

```
scripts/run_experiment.sh silhouette_titration 6
```

Without `--rerun` that submits only the incomplete jobs, so with the five rate
arms already on disk at thirty rows each it submits the white arm alone.

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

`silhouette_p_sender` stays at `DEFAULT.toml`'s 0.0 in all five arms.
Silhouetting the speaker removes colour from what it can *say*; this removes
colour from what the listener can *use*, which is what makes a colour-only
message stop paying. A config that moved both at once was written on
2026-08-31 and is retired into this experiment: two knobs would leave a
difference unattributable to either.

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

Expect all five arms to sit at chance for the first few epochs: `DEFAULT.toml`
sets `warm_up_epochs = 10`, which is a third of a thirty-epoch run.

## The white-fill arm

`06_silhouette_0.5_white.toml` holds `silhouette_p_receiver` at 0.5 and sets
`silhouette_fill = 1.0`. It is one key away from `05_silhouette_0.5` and is to be
read against that arm only; putting it on the rate ladder would make the ladder
measure two things.

**What it settles.** `diagnostics/silhouette_shape_probe.py` read
`white_threshold` transferring to clean images at 0.560 against 0.486 for the
fill now in `DEFAULT.toml` (job 123354) — the one measurement that pointed away
from the fill that was then chosen. It did not survive its own repeat: the same
arm, same seed, same GPU read 0.403 (job 123583), and six single-fit readings
across three fills all landed between 0.40 and 0.56 with none separable from
another, because cuDNN's convolution backward accumulates with atomics. The
probe cannot order these fills, so this arm asks the question at the level the
answer is wanted at — a full run, read on the shape/colour breakdown.

**What it is read against.** `05_silhouette_0.5` under the chromatic fill ended
its last five epochs at shape 0.606 and colour 0.528, having crossed over around
epoch 26 from shape ~0.55 and colour ~0.65. By the rules above that is the
mirror-image failure, not the result the key exists for. The white arm asks
whether the fill put it there or whether 0.5 is too strong a rate whatever the
objects are repainted to:

- **Shape high, colour recovering off 0.53** — the fill was the problem, and
  white is the better repainting at this rate.
- **The same crossover** — the rate is the problem, the fill is incidental, and
  the titration's answer stays at or below 0.4.
- **Both nearer chance than 05** — white cost more than it bought, which is what
  the BatchNorm argument predicts.

**What white costs.** Both costs are argued in full in `DEFAULT.toml`'s
`silhouette_fill` block and in `docs/data.md`; neither is a reason not to run the
arm, and both are reasons not to move the default on one good result here.
`white` is one of ShapeWorld's six colours and is exactly this fill, so a
repainted white object comes back a white object. Its shape is still treated —
the threshold binarises the anti-aliased edges as it does for every colour — but
no colour is removed, so one colour in six keeps a readable colour label in
silhouetted games and the receiver gets no signal there that colour was taken
away. (The stronger *bit-identical* version of this defect was grey's, under
coverage blending and the old flat 0.5; there are no blended edges left for it to
be true of.) And white is the brightest image the model ever sees: against mean object channels
(148.8, 148.8, 106.3) a rate of 0.5 puts a `BatchNorm2d(3)` over raw RGB at
1.36×, 1.36× and 1.70× the eval distribution, where the chromatic fill runs
+0.1% / +0.1% / −0.3%. That layer is the ViT receiver's, not this rung's, so it
bites here only insofar as the answer is meant to transfer up the ladder.

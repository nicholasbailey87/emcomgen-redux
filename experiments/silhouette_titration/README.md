# silhouette_titration

**What rate of receiver silhouetting does the CNN/GRU baseline need to escape
the colour-only minimum, without losing colour instead?**

Five copies of the ablation's rung 1 — ShapeWorld, `ResNet18SmallInput` on both
agents, `AveragePrototyper`, `SenderGRULM`, `ReceiverGRULM`,
`BilinearDiscriminator` — differing in `[data] silhouette_p_receiver` alone, at
0.1, 0.2, 0.3, 0.4 and 0.5. Everything else comes from `DEFAULT.toml`. Thirty
epochs, one seed each, five SLURM jobs.

```
scripts/run_experiment.sh silhouette_titration 5
```

## Why the rate is not already known

`silhouette_p_receiver` repaints the listener's whole view as flat achromatic
silhouettes with this probability, per game, at training time only. It exists to
break the colour-only local minimum Mu & Goodman report (~83% accuracy, appendix
A.1): with the six colours sitting at six distinct luma values a grayscale
conversion would re-encode colour as a scalar rather than remove it, where a flat
repaint removes it outright. See `DEFAULT.toml`'s block on the key for the
mechanism and for why `silhouette_fill` is 0.5.

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

# diagnostics

> **Rung numbers below are the old ladder.** It was renumbered when it grew to
> sixteen rungs; old 10 is new 12 and old 12 is new 16, both now carrying the
> speaker's contrast stage. See `experiments/README.md` for the full table.

Scripts for taking one suspect at a time out of a stuck run.

A pair that sits at chance offers no evidence about *why*. The speaker may have
nothing to say, the channel may be shut, the vision models may not have learned
a representation, and the loss may have found a way to go quiet without learning
anything — and all four produce the same two columns in `metrics.csv`:
`train_acc` at 0.5 and `train_loss` near ln 2. Nothing in a training run
separates them, because a training run only ever exercises all of them at once.

What these scripts do is build the real modules from a real rung config and then
run them on a synthetic task where the answer is known, so that a failure has
one possible cause instead of five. They need no dataset, no GPU and no cluster.

Nothing here is part of training. They are read when a run is not working.

---

## `comparer_probe.py`

Hands the listener a game it should be able to win, with the speaker, the
channel and both vision models removed. Both slots are built and run — the
language model and the discriminator — since which of them is at fault is
exactly what a probe on one alone could not say.

Each row is `n_examples` slots. Half are one concept and half another, drawn
from a pool of random prototype vectors with Gaussian noise on top; the message
is that concept's own fixed code, noise-free and the same every time the concept
comes up. Slots are shuffled, so position carries nothing. It is the easiest
protocol there is — already converged, perfectly discriminative — which is the
point: anything that fails here fails for a reason that has nothing to do with
emergence.

```
python diagnostics/comparer_probe.py                       # can it use a message?
python diagnostics/comparer_probe.py --message scrambled   # what does failure look like?
python diagnostics/comparer_probe.py --message scrambled --distractors varied
```

`--config` takes any rung. It defaults to
`experiments/ablation/configs/16_birds_receiver_cross_attention_lm.toml`, and works
on the bilinear baseline too. `--lr` defaults to the config's own
`optimiser.lr`, so the timings below are the rate the real run learns at, not a
convenience setting. About a minute on CPU for a thousand steps.

### The three questions

**Can the comparer use a message at all?** (default: `--message informative
--distractors clustered`.) Both halves are clusters of equal size, so clustering
the referents recovers the partition but not which half is positive. Exactly one
bit is missing and only the message carries it. Failure here means the fault is
in the comparer, and no amount of opening the channel or reshaping the loss will
help.

**What does having nothing to say look like?** (`--message scrambled
--distractors clustered`.) Identical in every respect except that the message
names an unrelated concept, so the comparer cannot win. Whatever the metrics
columns do here is the signature of a listener with no information, measured
rather than guessed — which is what makes it possible to recognise the same
signature in a real run.

**Is it clustering instead of talking?** (`--message scrambled --distractors
varied`.) Every distractor is a different concept, so the positives are the only
repeated ones and `referent_self_attention` can find them without reading
anything. High accuracy in this mode is the concept game's own shortcut. It is
deliberately reachable, because it is reachable in the real game too.

### Readings

Rung 12's listener, at the config's own lr of 1e-4. Taken before the split and
before the mix, so these are the two decoder stacks with a plain readout and no
bilinear path; they are the numbers the mix was designed against rather than
readings of what runs today:

| mode | accuracy | loss | excess kurtosis |
|---|---|---|---|
| informative, clustered | 1.000 by step ~175 | 0.13 | **−1.9 → −2.0** |
| scrambled, clustered | 0.39–0.69, no trend | 0.88–1.16 | **+5 to +18** |
| scrambled, varied | 0.93 by step 200 | 0.40 | −0.7 |

Rung 2's listener — the GRU and the bilinear form — informative and clustered:
1.000 by step 200, loss 0.079, kurtosis −1.77.

Read the **kurtosis** column, and read its sign. Accuracy and `score_sd` read
the *size* of the scores; kurtosis reads their shape, and shape is what
separates a listener that is discriminating from one that has nothing to say.
Negative means bimodal, floored at −2. Sustained positive alongside accuracy at
0.5 means the magnitude is being dumped into a handful of outliers while the
bulk sits at sigmoid 0.5.

The readings above were taken while the readout was standardised at a fixed
gain, and the numbers survived its removal because what the column measures is
more general than the arbitrage that first motivated it. That arbitrage —
outliers being cheap per unit of variance under a pinned variance budget — went
with the pin.

The bracketed `[module: …]` figures are the discriminator's own
`decision_spread` and `decision_kurtosis`, the ones that reach `metrics.csv`.
They are computed independently of the probe's, so a disagreement between the
two would mean the logged column is wrong. `AttentionDiscriminator` adds a
second bracket, `[mix a … agree …]`: how much of the score is the attention path
and how far the two paths agree within a game. Read those two together — see
docs/measurement.md.

Note what `decision_spread` alone cannot do: it read 2.7–5.1 in the informative
condition and 1.4–2.1 in the scrambled one — overlapping ranges, no verdict.
That is why `decision_kurtosis` exists.

### What this has established

Run against rungs 11 and 12 after
`receiver-cross-attention-birds.csv` and `receiver-cross-attention-shapeworld.csv`
sat at 0.5000 accuracy for thirty epochs:

- The comparer is not broken. It solves the one-bit game at the real learning
  rate in under two hundred steps.
- The clustering shortcut is reachable — 0.93 accuracy with the message
  scrambled — which is why those runs have to be judged on `test_topsim_*` and
  `unique_message_fraction` rather than on accuracy.

What it has **not** established, and what no version of it can: whether the
speaker has anything to say, or whether the channel can carry it. Those live
upstream of everything this script builds — which is what `bootstrap_probe.py`
is for.

---

## `bootstrap_probe.py`

Runs the whole pair — speaker, prototyper, language model, Gumbel channel,
listener — with only the two vision models replaced by frozen random prototypes.
That is a vision model that has already succeeded, which removes "the ViT has
not learned to separate species yet" without removing anything else, and makes
the loop cheap enough to run on a laptop.

```
python diagnostics/bootstrap_probe.py                        # rung 12
python diagnostics/bootstrap_probe.py --config experiments/ablation/configs/12_birds_receiver_vit.toml
```

It prints `metrics.csv`'s own columns under their own names, because a working
run takes off in a fixed order — the polarity tag separates, the speaker's logit
scale traverses, the channel opens, and only then does accuracy move — and the
question is which of them moves. Steps are not epochs: a rung 10 epoch is 3100
games, so read the order of events rather than the timings. About twenty minutes
for 2500 steps.

### What it settled

This is the script that identified the listener's readout as the cause of rungs
11–14 sitting at chance, and it is why the attention arm no longer batch-normalises
its scores at a fixed gain. At 2500 steps and the config's own 1e-4:

| | acc | loss | polarity_separation |
|---|---|---|---|
| rung 10, bilinear | 1.000 | 0.044 | 9.50 |
| rung 12, plain readout | 0.863 | 0.240 | 8.17 |
| rung 12, standardised readout | 0.606 | 0.826 | 3.64 |

The middle row takes off at step ~1600 by the same route as the top one:
`polarity_separation` crosses 6–8, the speaker's channel scale traverses,
`realised_survival` jumps 0.22 → 0.82, accuracy follows. The bottom row is the
same module with a `BatchNorm1d(1, affine=False)` and a fixed gain on its
readout, and nothing else changed.

The reading: a listener that cannot turn its volume down is a listener that must
commit through that volume from step zero, before the message carries anything.
What that costs the speaker is larger than what the collapse it prevents costs
the listener — rung 10 does that collapse (`score_scale` 0.856 → 0.238 over
thirty epochs) and reaches 1.000 anyway.

### What it is not for

Choosing a learning rate. At 1e-3 *both* listeners saturate their logits inside
a hundred steps and freeze, rung 10 included — accuracy 0.59 against 1.000 at
1e-4. Frozen vision and linearly separable concepts make this a poor model of
where the real run's optimum sits, and two points are not a sweep.

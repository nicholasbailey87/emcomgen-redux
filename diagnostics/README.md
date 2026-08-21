# diagnostics

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

Hands the listener's comparer a game it should be able to win, with the speaker,
the channel and both vision models removed.

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
`experiments/ablation/configs/12_birds_receiver_cross_attention.toml`, and works
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

`TransformerCrossAttentionComparer`, rung 12, at the config's own lr of 1e-4:

| mode | accuracy | loss | excess kurtosis |
|---|---|---|---|
| informative, clustered | 1.000 by step ~175 | 0.13 | **−1.9 → −2.0** |
| scrambled, clustered | 0.39–0.69, no trend | 0.88–1.16 | **+5 to +18** |
| scrambled, varied | 0.93 by step 200 | 0.40 | −0.7 |

`BilinearGRUComparer`, rung 2, informative and clustered: 1.000 by step 200,
loss 0.079, kurtosis −1.77.

Read the **kurtosis** column, and read its sign. The cross-attention comparer
standardises its readout, which pins the variance and leaves the shape free —
and the shape is where a listener with nothing to say goes. BCE against a
coin-flip label costs `|s|/2 + ln(1 + exp(-|s|))`, quadratic near zero but only
linear far out, while variance costs `s²`. So an outlier is cheap per unit of
variance it absorbs, and the cheapest uninformative allocation at a fixed spread
is a handful of enormous scores with everything else at zero, where sigmoid is
0.5 and the cost is ln 2. Negative kurtosis means bimodal scores, which is what
discriminating looks like, floored at −2. Sustained positive alongside accuracy
at 0.5 is the escape. `decision_gain` in `DEFAULT.toml` carries the arithmetic.

`score_sd` is pinned at `decision_gain` by construction and is printed only to
show that it is. The bracketed `[module: …]` figures are the comparer's own
`decision_spread` and `decision_kurtosis`, the ones that reach `metrics.csv`;
they are computed independently of the probe's, so a disagreement between the
two would mean the logged column is wrong.

Note what `decision_spread` alone cannot do: it read 2.7–5.1 in the informative
condition and 1.4–2.1 in the scrambled one — overlapping ranges, no verdict.
That is why `decision_kurtosis` exists.

### What this has established

Run against rungs 11 and 12 after
`receiver-cross-attention-birds.csv` and `receiver-cross-attention-shapeworld.csv`
sat at 0.5000 accuracy for thirty epochs:

- The comparer is not broken. It solves the one-bit game at the real learning
  rate in under two hundred steps.
- The heavy-tailed escape is real and reproducible in the module itself, not an
  inference from the loss curve.
- The clustering shortcut is reachable — 0.93 accuracy with the message
  scrambled — which is why those runs have to be judged on `test_topsim_*` and
  `unique_message_fraction` rather than on accuracy.

What it has **not** established, and what no version of it can: whether the
speaker has anything to say, or whether the channel can carry it. Those live
upstream of everything this script builds.

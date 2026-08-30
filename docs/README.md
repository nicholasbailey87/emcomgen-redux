# emcomgen-redux — design documentation

These files hold the reasoning that used to live in comments and docstrings
inside `code/`. The code itself is now commented only where it does something
that would otherwise read as a mistake; everything that justifies, litigates or
explains a choice is here.

| File | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | The speaker, the listener, the prototypers, and the two Transformer speaker arms |
| [channel.md](channel.md) | The channel: logit normalisation, the constant scale, `token_max_probability`, `uniform_weight`, and the two gradient estimators |
| [measurement.md](measurement.md) | The topsim signal-set family, concept prototypes, and every diagnostic column in `metrics.csv` |
| [data.md](data.md) | ShapeWorld and CUB: splits, the `test_same` holdout, the silhouette intervention, game counts |
| [training.md](training.md) | The training loop, per-module gradient clipping, AMP, resume discipline, config resolution |
| [broccoli.md](broccoli.md) | Conventions for calling into `broccoli`, and the DeepNorm residual scaling |
| [anecdotes.md](anecdotes.md) | The findings and failures behind the current design, with the numbers |
| [dubious-claims.md](dubious-claims.md) | Claims carried over from the code that may be stale, unverified, or self-contradictory |

## Reading order

`architecture.md` then `channel.md` covers the model. `measurement.md` covers
what a run reports and how to read it. `anecdotes.md` is the history: several
current choices only make sense as the survivors of something that failed, and
that file is where the failures are recorded.

## Provenance

Most of this text is lifted from the code with minimal editing, so it carries
the code's own voice, its commit hashes (`87c1027`, `fccba0f`, ...) and its
references to files outside `code/` (`diagnostics/README.md`, `issue.csv`,
`receiver-cross-attention-birds.csv`). Those references have not been verified
while extracting; see `dubious-claims.md`.

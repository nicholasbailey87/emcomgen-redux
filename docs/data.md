# Data

Two datasets: ShapeWorld (synthetic, 64px, logical-form concepts) and CUB
(birds, 224px, species concepts). `data/loader.py` dispatches on the dataset
name.

## Splits

Only the splits a run actually consumes are built: `train`, `test`, `test_same`.

There is **no val split**, because there is no best-epoch selection: training
runs to a fixed endpoint and the per-epoch `metrics.csv` trajectory is the
deliverable.

There are **no cross-game-type eval datasets**. The run trains and evaluates a
single game framing (concept, i.e. `percent_novel = 1.0`), under which the
speaker and listener see fully disjoint targets *and* distractors. That
disjointness is an intrinsic control against context-dependent degenerate codes,
which is what the cross-eval passes were guarding against — so building 12 extra
eval datasets bought nothing but I/O.

jayelm's naming is kept: "same" means the concepts are the same ones training
used, so `test_same` is the paper's **Acc (Seen)** column and `test` is
**Acc (Unseen)**.

ShapeWorld's `load` returns a fourth entry, `train_clean`, and it is not a
split: it is the training games drawn the way eval draws them — no silhouette,
no flip, no rotation, no mixup — sharing the training store rather than copying
it. Nothing is ever scored on it. `train.calibrate_batch_norm` reads it to
re-estimate BatchNorm's running statistics on clean images, and because it gets
no columns in `metrics.csv` it is exempt from the header check that raises when a
split has none. See "The silhouette intervention" below, and
`bn_calibration_batches` in `DEFAULT.toml`. CUB has no equivalent; the
calibration is a no-op where the key is absent.

### ShapeWorld

`test_same.npz` holds freshly generated worlds, so they are unseen *images* of
seen *concepts*, and the seen-concept split comes for free. The file is optional
on disk: `load` tolerates a missing `_same` split and skips it, and `train.py`
skips the corresponding eval pass.

### CUB

CUB classes are 1-indexed. Classes 1–150 are the training species; 151–200 are
held out wholesale and are the novel-concept `test` split — the same 50 species
as jayelm's, so the generalisation number keeps its old footing.

Classes 101–150 were a val split. There is no val split any more, so rather than
sit unused they are folded into training. That is what pays for the image-level
holdout: 150 species at 80% of their images is ~60% of the corpus for training,
against the ~50% that 100 whole species gave, with half again as much species
diversity.

`TRAIN_CLASSES_DEBUG` is `range(1, 5)`. jayelm's was `range(4)`, which asked for
a class 0 that does not exist and so gave debug runs three species rather than
four.

#### The `test_same` holdout

CUB has a finite pool of photographs per species, so the seen-concept property
has to be bought by holding images out of training. Without it, `test_same` would
measure recall of images the sender had already trained on rather than
generalisation to new instances of a seen concept.

`HOLDOUT_FRACTION = 0.2` and `HOLDOUT_SALT = "cub-test_same-v1"` are **module
constants rather than config keys on purpose**. Changing either moves the
boundary between `train` and `test_same`, which invalidates every run recorded
under the old value; that is a version of the dataset, not a knob to turn per
run. Bump the salt's suffix if the partition ever has to change, so that old and
new runs are distinguishable rather than silently pooled.

**`_holdout_rank` uses `hashlib.blake2b`, not an RNG and not the builtin
`hash()`**, because both of the alternatives move. `hash()` is salted per
process, and `numpy.random.Generator`'s stream is explicitly not guaranteed
across numpy releases (NEP 19 freezes only the legacy `RandomState`). A hash
function is fixed by its algorithm, so this partition reproduces on any machine,
in any environment, under any future version.

**`holdout_image_names` touches no RNG at all**, global or local, and that is a
requirement rather than a nicety. `train.py` seeds numpy from `--seed` and
`loader.worker_init` re-seeds it per worker, so an `np.random.choice` here would
hand every seed — and every resume, and every dataloader worker — a different
partition, and an image held out under seed 0 would be trained on under seed 1.
Since the sort key depends on the image *name* alone, the result is also
independent of how many species are passed, of the order they are passed in, and
of npz key order.

**Size is `max(n_examples, round(fraction × n))`.** CUB species carry 41–60
images, so 20% is 8–12 and the floor binds only at the bottom of that range. The
floor is required: `sample_game` draws `n_examples` *distinct* positives from a
single species per game (`replace=False`), so a pool of 8 would raise on every
game played on that species. `fraction × n` cannot land on a .5 for integer `n`
at 0.2, so the rounding rule is not load-bearing.

**`split_images` filters at construction, not at sampling time**, and that is
what makes the `test_same` *distractors* held out too. `sample_negatives` can
only reach what is in `self.imgs`, so a dataset built from the held-out pool
cannot show the listener an image the sender trained on, on either side of the
game.

#### Games per epoch

`n_games(split, n_species, config)`:

- **Train** — `CUBDataset.__getitem__` ignores its index and samples a fresh
  game, so this is the size of an epoch rather than a set of stored games.
  Consecutive epochs are independent draws from a combinatorially large space,
  and nothing is exhausted by raising it. `games_per_epoch` in `[birds.data]`
  sets it; the default is no longer jayelm's 1,000.
- **Eval** — sized *per species* rather than flat. The two eval splits hold 50
  and 150 species, so one shared game count would give them very different
  per-concept coverage — and topsim measures one prototype per concept, built
  from the modal message over that concept's instances, so coverage per species
  is the quantity that has to be held equal for the two splits to be read against
  each other. jayelm's flat 200 gave `test` four games a species and would have
  given `test_same` 1.3.

The eval size follows the species actually present on disk (`len(subset)`), not
the nominal class range.

## The silhouette intervention

`generic.silhouette` repaints each image's object in a flat chromatic fill,
keeping its shape.

ShapeWorld's six colours (red, blue, green, yellow, white, gray) sit at six
distinct luma values — roughly 29, 76, 128, 150, 226, 255 — so a plain grayscale
conversion does not remove colour, it re-encodes it as a single scalar that one
conv filter can threshold. A flat repaint removes it from the *interior*: with a
single object on a black ground, every colour's fully covered pixels render
identically.

**It did not remove it from the edges, and this document claimed otherwise for a
fortnight.** The claim was that the mutual information between colour and pixels
is zero *by construction*. Under the coverage blend that ran from 2026-08-29 to
2026-09-01 it was false and had never been measured: a random forest given only
geometry-invariant summary statistics of silhouetted images recovered the
original colour well above chance, and the count of partially covered edge pixels
alone ordered the six palette colours with a mean Kendall τ of **+0.90** across
600 randomised geometries. There are no partially covered edge pixels now — see
the next paragraph — so that particular mechanism is gone, and what is left of it
is set out under **the leak the edges took with them** below. The history stays
here because the load-bearing sentence was false for a fortnight and nobody
looked.

**Shape is carried by a threshold, not by coverage.** Every pixel above
`peak / 2` is repainted to the fill and every other pixel to zero, where `peak`
is the image's own maximum luma. Per image is what makes it colour-invariant:
blue peaks at 0.114 where white peaks at 1.0, and a fixed threshold would erase
one and keep the other. A pixel at coverage `k` has `luma == k · peak` whatever
the colour, so the threshold falls at `k = 0.5` for all six. An all-black image
has peak 0 and must stay black, so the `peak > 0` guard is load-bearing.

**Why a threshold, when coverage describes the object better.** It does describe
it better, and that is not the criterion. The intervention is training-time only
and eval is never silhouetted, so what a fill is worth is how much of what a
model learns under it survives the trip back to clean images.
`diagnostics/silhouette_shape_probe.py` measures exactly that — a fresh `Conv4`
trained on silhouetted positives, tested both on the same repainting and on the
clean originals (cluster job 123354, 600 games, one seed):

| arm | in_domain | clean |
|---|---|---|
| clean | 0.999 | 0.999 |
| white_threshold | 0.794 | **0.560** |
| white_coverage | 1.000 | 0.483 |
| fill_coverage | 1.000 | 0.486 |

Chance is 0.306 over four bare shape concepts. The coverage arms are perfectly
readable under their own repainting and lose almost all of it at eval; the
threshold is the only arm that gives up in-domain accuracy and keeps more of it
on clean images. That is what put the threshold back in on 2026-09-01, after
three days of coverage blending.

**That table is one fit per arm and it does not survive its own repeat.**
Re-running it unchanged, at the same seed on the same GPU, put `white_threshold`
at 0.403 where it had read 0.560 (jobs 123583 and 123354) — convolution backward
on cuDNN accumulates with atomics — and a third run read the three live fills at
0.412, 0.553 and 0.506. That is six single-fit readings across two transforms and
three fills, every one between 0.40 and 0.56, none separable from another.

Two things survive the repeat, and they are what the section above should be read
as resting on. Every repainting arm sits well above chance and far below `clean`,
so shape does transfer and most of it is lost on the way. And the threshold
reliably costs in-domain readability — 0.79–0.97 against the blend's 1.000, on
every run. **Which fill transfers best is open.** The probe now fits each arm
five times and reports a mean, an sd and a range; run it before quoting a number
from here, and do not read a gap smaller than the ranges overlap.

Two costs come with it, and both are real. The repainted region is **larger**
than the object it replaces, since every partly lit edge pixel is promoted to
fully lit; `test_the_edge_is_promoted_to_the_fill` pins that, and the per-channel
means quoted below run slightly high against it. And a threshold at half the peak
erases the darker of two objects outright rather than merely dimming it — see
docs/dubious-claims.md, which is where the one-object assumption lives.

**The fill is `[data] silhouette_fill`, a per-channel fraction of maximum,
defaulting to the palette's own mean object colour (149, 149, 106).** Which
colour it is is a claim about the receiver's input BatchNorm; *that it is
chromatic* is a claim about the palette, below. The value did not change when
the edge treatment did. `ViT2` passes
`initial_batch_norm=True`, so broccoli puts an `nn.BatchNorm2d(3)` over raw RGB
at `preprocess[0]`, and ShapeWorld gets no other input normalisation — whatever
this function emits lands directly on that layer's statistics. Since eval is
never silhouetted (below), those running statistics are gathered on a mixture
and then applied to clean images, and any gap between the two is a fixed offset
of `(μ_clean − μ_running) / σ` on every eval activation. The learned affine
cannot absorb it, because at train the offset is zero.

That argument is about the input layer because that is where it is easiest to
state, but nothing in it is specific to the input layer: every deeper BatchNorm
in either agent is estimated on the same mixture, and the augmentations added
since — flips, rotation, mixup — shift them too. The top-level
`bn_calibration_batches` is the answer to it. When it is non-zero,
`train.calibrate_batch_norm` re-estimates every BatchNorm's running statistics
on a clean `train_clean` loader between the train pass and the eval passes, so
eval reads statistics gathered on the distribution eval actually sees. It
defaults to `0`, which is what every run on record — the whole silhouette
titration included — was scored under;
`diagnostics/bn_calibration_probe.py` is what says which N to turn it up to.

White was the worst available answer by this measure: the brightest image the
model ever sees. Against the palette as `tests/test_silhouette.py` states it —
mean object channels (148.8, 148.8, 106.3) — a rate of 0.5 put the running means
at 1.36×, 1.36× and 1.70× the eval distribution, worst on blue. Note that the
probe above read its best transfer at exactly that fill, and by a clear margin.
The two changes are confounded in that arm — `white_threshold` differs from
`fill_coverage` in edges *and* intensity — so either the BatchNorm gap is not
what governs transfer, or the threshold buys more than the gap costs. Re-running
the probe on the fill as it now stands is what separates them, and it has not
been run.

Half replaced it, on the argument that 0.5 is the maximum-entropy answer for a
palette the code does not know, and is robust to how that ignorance is
formalised: uniform over the RGB cube, uniform over the six saturated primaries
and secondaries, and uniform over hue at full saturation and value all give 0.5
per channel. **That argument has been abandoned, deliberately, and the fill is
now a constant fitted to this dataset.** It is worth being explicit that
something real was given up: the new value is wrong the day ShapeWorld's palette
changes, and the old one was not. Two measurements forced it.

*A rounding-tie lattice.* Under coverage blending, a stored edge pixel was
`round(coverage × fill × 255)`.
At `fill = 0.5` that is `coverage × 127.5`, and any colour whose maximum channel
is 255 has coverage exactly `n/255`, so the product is exactly `n/2` and every
odd `n` is an exact `.5` tie. Ties resolve by round-half-to-even on a float32
whose last bits differ per colour, because each colour's luma is a different
weighted sum of the same integer. **43 of the 256 stored levels made red, blue,
green, yellow and white disagree** — colour re-encoded in the anti-aliasing, by
the transform that exists to remove it. The fix is that `fill × 255` is an
*integer* per channel: `n·F/255` cannot be a half-integer when `F` is an integer,
since 255 is odd and `2nF` is even, so no **cross-colour** tie can exist. (Grey
keeps ties of its own — its coverage is `m/128`, so `m = 64` gives 74.5 — but
grey is alone on that ramp, so a tie there makes nothing disagree with anything.)
The threshold has since made the whole lattice moot — there are no anti-aliased
edges left to round — but the integer property is kept, on the weaker ground that
it also means the fill is a colour the store can represent exactly.

*A palette collision.* `0.5 × 255 = 128` is exactly `gray`, so silhouetting a
grey object was bit-identical in and out: max delta 0 over 0 pixels. One colour
in six was silently exempt.

Any integer `F` fixes both. The fill is *chromatic* because of the BatchNorm:
the palette is blue-deficient — blue appears in two of six colours where red and
green appear in three — so the mean object colour is (148.83, 148.83, 106.33)
and no achromatic constant can match all three channels. Raising one only trades
R and G against B: 128 runs −14.3% / −14.3% / +19.9% against eval, 135 runs
−9.3% / −9.3% / **+27.0%**, 144 runs −3.2% / −3.2% / **+35.4%**, and
(149, 149, 106) runs **+0.1% / +0.1% / −0.3%**. Those means weight the six
colours uniformly; ShapeWorld's empirical colour frequencies have not been
checked, so the match is approximate rather than exact if they are not.
DEFAULT.toml carries the full argument, including why a per-image random colour
was rejected despite matching the distribution better.

A chromatic fill has **no fixed point**: it is not any palette colour, so no
colour is exempt. Under both previous fills exactly one was — `white` until
2026-08-29, then `gray`.

### The leak the edges took with them

Under coverage blending grey stayed identifiable at ~0.97 recall (chance 0.167)
after every fix above, and it was not repairable at this layer. An anti-aliased
edge pixel is stored as `round(k · C)` per channel, so the ramp ran `0 → max(C)`
and the number of representable coverage levels was

```
levels = 255 · V + 1        where V = max(R, G, B) / 255 is HSV Value
```

Every palette colour has V = 1.0 except `gray`, at V = 0.502. Grey therefore
resolved coverage in 129 steps where everything else used 256, and its output
**skipped specific intensity values** — a structural gap in the value histogram,
identical on every shape at every size, and that is what a classifier found. No
post-hoc processing repaired it: for a given bright-colour stored level the
window of true coverages was `128/255` grey-levels wide, so 128 of 256 levels
were ambiguous and the map was not a function. Stochastic rounding, dithering,
lattice equalisation and per-game jitter were prototyped and did reduce it, the
best combination to chance, but each worked by destroying edge resolution — which
is where shape lives, and which the threshold destroys outright.

**The threshold closes it, by removing the ramp.** The output is two-valued per
channel, `{0, fill}`, for every palette colour, so there is no histogram left for
the bit-depth to shape.

What survives is a different kind of thing and a much smaller one. The threshold
is taken on the *stored* image, and grey stores coverage `k` as `round(128k)`
where every other colour stores `round(255k)`, so grey turns on at `k > 0.5039`
and the rest at `k ≥ 0.502`. Swept over all 256 stored levels, the six colours
repaint identically at 255 of them and differ at exactly one, `n = 128`, where
grey is off and the other five are on. That moves an object's boundary by up to a
pixel; it does not put a colour-dependent gap in every image's histogram.
`test_the_threshold_is_colour_invariant` pins both halves — the agreement and the
exception — so the day the gap widens, somebody notices.

Note that HSL is the wrong coordinate for any of this and should not be used:
red, blue, green and yellow all sit at HSL L = 0.500 and grey at 0.502, so L does
not separate grey at all. V does, and V *was* the edge bit-depth.

dtype and range are preserved — the fill is read against 255 for integer images
and against 1.0 for floating-point ones, and integer output is rounded rather
than truncated. A new tensor is returned; the input is untouched.

### Rolled per game, not per image

`_apply_silhouette` silhouettes each agent's whole view, or neither. With 10
targets in a set, rolling per image would leave ~(1−p)×10 of them coloured and
the colour cue recoverable from the set, which is not the intervention.
Silhouetting the whole view is what makes shape the only available cue.

The two rolls are independent, so the pair of rates selects the regime: `(0, p)`
silhouettes only the receiver, `(p, 0)` only the sender, `(p, p)` either or both.
Each agent's view is a full side of the game, so one roll already covers both
polarities and, in the concept game, the disjoint half that agent sees.

**Training-time only.** Eval is never silhouetted, so the reported numbers stay
comparable to the paper's and to the `probe_shape.py` sweep, which measures the
sender on un-augmented images.

### The rates are 0.0 sender, 0.1 receiver

Both were 0.0 for a few hours on 2026-08-30 and the receiver has been at 0.1
since. It ran at `silhouette_p_receiver = 0.5` from 2026-08-25, and on the
constant channel that was too strong: rung 9 learned shape better than any ShapeWorld run on record
(0.758 on shape concepts, 0.829 on `and_shape_shape`, against ~0.52 for every
run that had never escaped the colour shortcut) and left colour at exactly chance
for all thirty epochs. Since eval is never silhouetted, that 0.50 is a failure to
communicate colour rather than a ceiling the repaint imposes — the mirror image
of the failure the intervention exists for.

At 0.0 the same rung reaches 0.661 aggregate by epoch 5 with colour at 0.747
*and* shape at 0.646, past the silhouetted run's epoch-29 numbers on both
features. Colour arrives first and shape follows, so on this channel colour leads
shape rather than blocking it.

The likely reason 0.5 is too strong here is that it suppresses the *channel* and
not only the colour feature: half the receiver's games contain nothing worth
decoding, and `unmixed_survival` sat at ~0.28 against ~0.45 at 0.0. DEFAULT.toml
carries the full note, and `experiments/silhouette_titration/` is the sweep.

The rate did not move when the fill went back to a threshold on 2026-09-01, so
that is the only change between the next ShapeWorld run and the last. The
titration's earlier rungs ran under the coverage fill and are not comparable
across it.

In the reference game, `percent_novel = 0.0` hands back the *same* tensor for
both agents; `silhouette` returns a new one, so an independent roll per agent is
still safe there.

## Copy-on-write and in-place mutation

`ConceptDataset.__getitem__` copies the row (`img = np.array(img)`) before any
shuffle or re-assignment. The shuffles write in place, and when the store is in
memory (`load_shapeworld_into_memory`), `self.x[i]` is a *view* onto the shared
array — so writing through it would permanently permute the dataset and, worse,
break copy-on-write for forked dataloader workers, which would each end up
copying every row they touch. The copy is ~0.5 MB and costs nothing next to the
forward pass. `ShapeWorldDataset.get_reference_game` copies for the same reason.

## Shape descriptors and world files

`extract_shapes` returns per-*image* descriptors, which disagree with a
subsampled (40-image) store. They are consumed solely by
`get_reference_game`/`shapes_to_idx`, which concept games never call, and they
are the only consumer of the world JSON files — so concept games skip parsing
them altogether (`need_shapes=config['reference_game']`). Reference games use the
separate `shapeworld_ref` dataset.

## Encoding details

hdf5 hands language back as bytes; npz hands it back as numpy unicode scalars.
`load_split` dispatches per element rather than on the array dtype, which reports
neither cleanly.

`ConceptDataset.to_idx` adds SOS and EOS, so `lang_len` is the token count plus
2, and the index array is padded to the longest.

## Image transforms

`data/image_util.py` is the few-shot-learning transform stack: `Resize` at 1.15×
the target followed by `CenterCrop` for eval, `RandomResizedCrop` +
`ImageJitter` + `RandomHorizontalFlip` for training, ImageNet normalisation
either way. `ImageJitter` applies Brightness/Contrast/Color enhancement factors
drawn uniformly in `1 ± alpha`.

CUB images are stored per species as `img.npz` under
`CUB_200_2011/images/<class>/`, built by `save_cub_np.py`. Metadata is either
per-image attributes (312 binary attributes per image) or per-class attributes
(thresholded at 50% of the continuous class-attribute table), selected by whether
the run is a reference game.

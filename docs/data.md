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

`generic.silhouette` repaints each image's object in a flat achromatic fill,
keeping its shape.

ShapeWorld's six colours (red, blue, green, yellow, white, gray) sit at six
distinct luma values — roughly 29, 76, 128, 150, 226, 255 — so a plain grayscale
conversion does not remove colour, it re-encodes it as a single scalar that one
conv filter can threshold. A flat repaint does remove it: with a single object on
a black ground every colour renders identically, so the mutual information
between colour and pixels is zero *by construction* rather than by hoping two
distributions overlap.

**Shape is carried by coverage, not by a threshold.** Every pixel is scaled by
`luma / peak`, its object coverage. For a flat object on black a fully covered
pixel has `luma == peak` and a pixel at coverage `k` has `luma == k * peak`, so
that ratio recovers the coverage exactly and does so independently of colour.
The peak is per image, which is what makes it colour-invariant: blue peaks at
0.114 where white peaks at 1.0, and a fixed divisor would render one far dimmer
than the other. An all-black image has peak 0 and must stay black, so the
`peak > 0` guard is load-bearing — without it the ratio is a NaN rather than a
zero.

This replaced a `luma > peak / 2` threshold on 2026-08-29. The threshold was
colour-invariant too, but it promoted every partly-lit edge pixel to fully lit,
so the repainted region came back larger than the object it replaced.
`test_coverage_is_preserved` pins the difference.

**The fill is `[data] silhouette_fill` of maximum in all three channels, half by
default.** That is a claim about the receiver's input BatchNorm. `ViT2` passes
`initial_batch_norm=True`, so broccoli puts an `nn.BatchNorm2d(3)` over raw RGB
at `preprocess[0]`, and ShapeWorld gets no other input normalisation — whatever
this function emits lands directly on that layer's statistics. Since eval is
never silhouetted (below), those running statistics are gathered on a mixture
and then applied to clean images, and any gap between the two is a fixed offset
of `(μ_clean − μ_running) / σ` on every eval activation. The learned affine
cannot absorb it, because at train the offset is zero.

White was the worst available answer: the brightest image the model ever sees.
Against the palette as `tests/test_silhouette.py` states it — mean object
channels (148.8, 148.8, 106.3) — a rate of 0.5 put the running means at 1.36×,
1.36× and 1.70× the eval distribution, worst on blue. Half of maximum is the
maximum-entropy answer for a palette the code does not know, and is robust to
how that ignorance is formalised: uniform over the RGB cube, uniform over the
six saturated primaries and secondaries, and uniform over hue at full saturation
and value all give 0.5 per channel. DEFAULT.toml carries what that costs against
a measured constant, and why a per-image random colour was rejected despite
matching the distribution better.

One palette colour is always the transform's fixed point, and the fill decides
which: at 0.5 it is `gray`, where under the white fill it was `white`.

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

### Both rates are 0.0 as of 2026-08-30

The intervention is built and tested and currently switched off. It ran at
`silhouette_p_receiver = 0.5` from 2026-08-25, and on the constant channel that
was too strong: rung 9 learned shape better than any ShapeWorld run on record
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
carries the full note and the titration plan if the colour-only minimum returns.

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

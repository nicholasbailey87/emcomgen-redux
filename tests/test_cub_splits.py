"""
Tests for the CUB `train` / `test_same` / `test` split construction.

Runnable without pytest:  python tests/test_cub_splits.py

Nothing in `tests/` covered `code/data/` before this file, and the properties
below are all ones that fail *quietly*.

`test_same` only means anything if training never saw its images. The whole
split is bought by holding photographs out of the training species, so if the
partition slips -- between seeds, between resumes, between dataloader workers --
`test_same` silently degrades from "generalisation to new instances of a seen
concept" into "recall of images already trained on", and the number it reports
goes *up*. That is why `holdout_image_names` uses a hash rather than an RNG, and
why four of these tests do nothing but pin its determinism. `train.py` seeds
numpy from `--seed` and `loader.worker_init` re-seeds it per worker, so any
reliance on global RNG state would give each seed a different partition and every
seed would train on some other seed's test set.

The floor at `n_examples` is the second quiet failure. CUB species carry 41-60
images and `sample_game` draws `n_examples` *distinct* positives from one species
per game (`replace=False`), so a flat 20% of the smallest species holds out 8 of
41 and raises on every game played on it -- but only for the handful of species
at the bottom of the range, and only once training reaches them.

The third is the distractor path. Filtering images at dataset construction rather
than at sampling time is what keeps `sample_negatives` inside the held-out pool;
a `test_same` split whose targets are held out but whose distractors are training
images is a subtler leak than it looks, and no accuracy curve would reveal it.

All of it runs on synthesised dicts. `cub.load` reads per-class npz and three CSV
metadata files off disk, so the split logic is deliberately factored into pure
module-level functions (`holdout_image_names`, `split_images`, `n_games`) that
take plain dicts -- the same approach `tests/test_backbones.py` takes to building
a pair without a dataloader.
"""

import os
import random
import tempfile

import numpy as np
import torch

import _bootstrap  # noqa: F401

from data import cub

# `parse_config` is imported lazily, in `_cub_config` alone. The split logic
# under test is pure -- dicts in, dicts out -- so everything here but the two
# epoch-size tests runs anywhere numpy and torch do, without needing a TOML
# parser or a loadable DEFAULT.toml.

# `[birds.data] n_examples`. Asserted against the real config in
# `test_eval_length_is_sized_per_species`; hard-coded here so the unit tests do
# not depend on DEFAULT.toml being loadable.
N_EXAMPLES = 10

# The observed per-species image counts in CUB-200-2011: 11,788 images over 200
# species, minimum 41, maximum 60.
CUB_SIZES = range(41, 61)


def _names(cl, n):
    """Image names in the form `cub.load` produces: paths under the class dir."""
    return [
        os.path.join("cub", "CUB_200_2011", "images", f"{cl:03d}.Species",
                     f"{cl:03d}_{i:04d}.jpg")
        for i in range(n)
    ]


def _img_names(classes, n_images):
    """class id -> its image names, sized by the `n_images(cl)` callable."""
    return {cl: _names(cl, n_images(cl)) for cl in classes}


def _imgs(classes, n_images, size=4):
    """
    class id -> {image name: array}, plus a colour -> name index.

    Each image is a single constant colour, unique to it, so an image found in a
    sampled game can be traced back to the name it came from. That is what
    `test_test_same_games_never_show_a_training_image` needs.

    The index is spread over the three channels, base 256, because these stay
    `uint8` like the real per-class npz arrays and a one-channel counter would
    overflow after 256 images -- ten species is already more than that.
    """
    imgs = {}
    by_colour = {}
    index = 0
    for cl in classes:
        imgs[cl] = {}
        for name in _names(cl, n_images(cl)):
            colour = (index // 65536, (index // 256) % 256, index % 256)
            imgs[cl][name] = np.full((size, size, 3), colour, dtype=np.uint8)
            by_colour[colour] = name
            index += 1
    return imgs, by_colour


def _cub_config():
    """
    The real birds config. `parse_config.get_config` merges the `[birds.*]`
    family off the *name* of the dataset directory, so a two-line TOML is enough
    to get DEFAULT.toml's actual birds values -- which is the point: these
    numbers should fail here if someone edits them by accident.
    """
    import parse_config

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write('name = "test_cub_splits"\n[data]\ndataset = "../data/cub"\n')
        path = f.name
    try:
        return parse_config.get_config(path)
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------
# Holdout size
# --------------------------------------------------------------------------

def test_holdout_size_is_the_fraction_with_a_floor_at_n_examples():
    """
    20% of a species, except where that would leave fewer than `n_examples`
    images to draw a game's distinct positives from. At CUB's sizes the floor
    binds for species with 41-47 images.
    """
    img_names = _img_names(CUB_SIZES, lambda cl: cl)
    holdout = cub.holdout_image_names(img_names, n_examples=N_EXAMPLES)

    for n in CUB_SIZES:
        expected = max(N_EXAMPLES, int(round(cub.HOLDOUT_FRACTION * n)))
        assert len(holdout[n]) == expected, f"{n} images -> {len(holdout[n])}"
        assert len(holdout[n]) >= N_EXAMPLES
        assert n - len(holdout[n]) >= N_EXAMPLES


def test_every_species_can_supply_a_full_game_from_either_side():
    """
    The property the floor exists for: `sample_game` calls
    `np.random.choice(names, size=n_examples, replace=False)`, which raises if a
    pool is too small. Both sides of the partition must survive it.
    """
    img_names = _img_names(CUB_SIZES, lambda cl: cl)
    holdout = cub.holdout_image_names(img_names, n_examples=N_EXAMPLES)

    for cl, names in img_names.items():
        held = sorted(holdout[cl])
        kept = sorted(set(names) - holdout[cl])
        for pool in (held, kept):
            np.random.choice(pool, size=N_EXAMPLES, replace=False)


def test_holdout_raises_when_a_species_is_too_small():
    """
    Loudly, rather than silently handing training an under-filled pool. No CUB
    species is this small, but a future dataset or a raised `n_examples` could
    make one so.
    """
    try:
        cub.holdout_image_names(_img_names([1], lambda cl: 15), n_examples=N_EXAMPLES)
    except ValueError as e:
        assert "below n_examples" in str(e), e
    else:
        raise AssertionError("15 images at n_examples=10 should not be splittable")


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_holdout_does_not_touch_the_global_rng():
    """
    Neither reads nor advances numpy's global state. If it did, the partition
    would depend on how much sampling had happened before it, and adding a
    single draw anywhere upstream would move the split.
    """
    img_names = _img_names(range(1, 21), lambda cl: 41 + cl)

    np.random.seed(0)
    expected = np.random.random()

    np.random.seed(0)
    cub.holdout_image_names(img_names, n_examples=N_EXAMPLES)
    assert np.random.random() == expected


def test_holdout_is_independent_of_the_global_seed():
    """
    The load-bearing one. An image held out under seed 0 must not be a training
    image under seed 1, or every seed trains on some other seed's test set and
    `test_same` stops measuring generalisation.
    """
    img_names = _img_names(range(1, 21), lambda cl: 41 + cl)

    np.random.seed(0)
    torch.manual_seed(0)
    first = cub.holdout_image_names(img_names, n_examples=N_EXAMPLES)

    np.random.seed(1234)
    torch.manual_seed(1234)
    second = cub.holdout_image_names(img_names, n_examples=N_EXAMPLES)

    assert first == second


def test_holdout_is_stable_across_input_order():
    """
    Keyed on the image name alone, so npz key order -- whatever `os.listdir`
    handed `save_cub_np.py` when the archives were built -- cannot move it.
    Regenerating the data on another filesystem must reproduce the partition.
    """
    img_names = _img_names(range(1, 21), lambda cl: 41 + cl)
    expected = cub.holdout_image_names(img_names, n_examples=N_EXAMPLES)

    shuffler = random.Random(1)
    shuffled = {}
    for cl in reversed(list(img_names)):
        names = list(img_names[cl])
        shuffler.shuffle(names)
        shuffled[cl] = names

    assert cub.holdout_image_names(shuffled, n_examples=N_EXAMPLES) == expected


def test_holdout_is_stable_under_the_set_of_species_passed():
    """
    Per-species and per-name, not drawn from one global stream. A debug run over
    four species must hold out exactly what a full run holds out for those four.
    """
    img_names = _img_names(range(1, 21), lambda cl: 41 + cl)
    full = cub.holdout_image_names(img_names, n_examples=N_EXAMPLES)
    alone = cub.holdout_image_names({7: img_names[7]}, n_examples=N_EXAMPLES)

    assert alone[7] == full[7]


# --------------------------------------------------------------------------
# The partition and the class ranges
# --------------------------------------------------------------------------

def test_train_and_test_same_partition_the_training_images():
    """
    Disjoint, exhaustive, and both sides usable. Overlap is the leak this whole
    change exists to avoid; a gap would mean images paid for and then discarded.
    """
    classes = range(1, 21)
    imgs, _ = _imgs(classes, lambda cl: 41 + cl)
    holdout = cub.holdout_image_names(
        {cl: list(im) for cl, im in imgs.items()}, n_examples=N_EXAMPLES
    )

    train = cub.split_images(imgs, classes, holdout, keep_holdout=False)
    test_same = cub.split_images(imgs, classes, holdout, keep_holdout=True)

    assert set(train) == set(test_same) == set(classes)
    for cl in classes:
        train_names = set(train[cl])
        same_names = set(test_same[cl])
        assert not (train_names & same_names)
        assert train_names | same_names == set(imgs[cl])
        assert len(train_names) >= N_EXAMPLES
        assert len(same_names) >= N_EXAMPLES


def test_test_split_keeps_every_image_of_its_species():
    """
    `test` species are unseen wholesale, so no image-level holdout applies --
    passing no `holdout` must be a plain class filter.
    """
    classes = list(range(1, 6)) + list(range(151, 156))
    imgs, _ = _imgs(classes, lambda cl: 50)

    test = cub.split_images(imgs, range(151, 201))

    assert set(test) == set(range(151, 156))
    for cl in test:
        assert set(test[cl]) == set(imgs[cl])


def test_class_ranges_are_disjoint_and_cover_cub():
    """
    150 training species and 50 novel ones, together the whole of CUB. The 50
    that used to be an unused val split are now in `train`, which is what pays
    for the image-level holdout.
    """
    assert set(cub.TRAIN_CLASSES) & set(cub.TEST_CLASSES) == set()
    assert set(cub.TRAIN_CLASSES) | set(cub.TEST_CLASSES) == set(range(1, 201))
    assert len(cub.TRAIN_CLASSES) == 150
    assert len(cub.TEST_CLASSES) == 50


def test_debug_class_ranges_are_valid_cub_classes():
    """
    CUB is 1-indexed. jayelm's `TRAIN_CLASSES_DEBUG = range(4)` asked for a
    class 0 that does not exist, so debug runs quietly got three species.
    """
    assert set(cub.TRAIN_CLASSES_DEBUG) & set(cub.TEST_CLASSES_DEBUG) == set()
    for classes in (cub.TRAIN_CLASSES_DEBUG, cub.TEST_CLASSES_DEBUG):
        assert set(classes) <= set(range(1, 201))
        assert len(classes) == 4


# --------------------------------------------------------------------------
# The dataset built on the held-out side
# --------------------------------------------------------------------------

def test_test_same_games_never_show_a_training_image():
    """
    Targets *and* distractors. `sample_negatives` draws from `self.imgs`, so
    filtering at construction is the only thing enforcing this -- there is no
    check at sampling time and no metric that would reveal a distractor drawn
    from the training pool.
    """
    classes = range(1, 11)
    imgs, by_colour = _imgs(classes, lambda cl: 41 + cl)
    holdout = cub.holdout_image_names(
        {cl: list(im) for cl, im in imgs.items()}, n_examples=N_EXAMPLES
    )

    held_names = set().union(*holdout.values())
    metadata = {
        name: np.zeros(312, dtype=np.uint8)
        for cl in classes for name in imgs[cl]
    }

    dataset = cub.CUBDataset(
        cub.split_images(imgs, classes, holdout, keep_holdout=True),
        metadata,
        n_examples=N_EXAMPLES,
        transform=lambda img: torch.from_numpy(np.asarray(img)),
        length=200,
        percent_novel=1.0,
    )

    np.random.seed(0)
    seen = set()
    for _ in range(200):
        spk_inp, _, lis_inp, _, _, _ = dataset.sample_game()
        for view in (spk_inp, lis_inp):
            colours = torch.unique(view.reshape(-1, 3), dim=0)
            for colour in colours.tolist():
                name = by_colour[tuple(colour)]
                seen.add(name)
                assert name in held_names, (
                    f"{name} is a training image but appeared in a test_same game"
                )

    # Guard against the assertion above passing vacuously on an empty or
    # single-image sample: 200 games of 20 images over ten species should reach
    # most of the held-out pool.
    assert len(seen) > len(held_names) / 2, f"only {len(seen)} images sampled"


# --------------------------------------------------------------------------
# Epoch sizes
# --------------------------------------------------------------------------

def test_eval_length_is_sized_per_species():
    """
    Equal games per species across both eval splits, which is what makes their
    topsim readable against each other: topsim builds one prototype per concept,
    so it is coverage per *species* that has to match, not total games. Pins
    DEFAULT.toml's birds values against a silent edit.
    """
    config = _cub_config()

    assert config['data']['n_examples'] == N_EXAMPLES
    assert config['data']['eval_games_per_species'] == 16
    assert config['data']['games_per_epoch'] == 5000

    assert cub.n_games("train", 150, config) == 5000
    assert cub.n_games("test", 50, config) == 800
    assert cub.n_games("test_same", 150, config) == 2400


def test_debug_shrinks_every_split_but_never_to_zero():
    """`// 10` on a small debug species count must not produce an empty epoch."""
    config = _cub_config()
    config['debug'] = True

    assert cub.n_games("train", 4, config) == 500
    assert cub.n_games("test", 4, config) == 6
    assert cub.n_games("test", 1, config) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")

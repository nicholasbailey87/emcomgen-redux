#!/usr/bin/env python3
"""
Write a 40-image-per-game copy of a ShapeWorld dataset.

Each stored game holds 80-100 images, but ``data/util.py:split_spk_lis`` only
ever reads ``n_examples`` (20) of them: images ``[0, 10)`` and ``[midp, midp+10)``
go to the speaker, ``[10, 20)`` and ``[midp+10, midp+20)`` to the listener. With
``midp = 20`` that consumes the full stored row, so 40 images is the smallest row
that leaves the concept game (``percent_novel = 1.0``, speaker and listener on
fully disjoint targets *and* distractors) intact. Shrinking to 40 takes the
train store from ~20-25 GB to ~9.8 GB, which fits in memory and removes ~32 GB
of disk reads per epoch.

Augmentation is unaffected in practice: ``ConceptDataset.__getitem__`` permutes
positives and negatives independently, so ``midp = 20`` still yields
``C(20,10) = 184,756`` speaker/listener divisions on each side.

Which 20 to keep
----------------
*Not* the first 20. Appendix ``app:hard`` of the paper states that for
disjunctive concepts the generator samples 1/3 of targets satisfying only the
left disjunct, 1/3 only the right, and 1/3 both (and correspondingly for
conjunctive distractors). If that stratification is stored in order, slicing the
first 20 of 40 would silently drop a whole stratum and change what the task
tests.

Two strategies are available:

``stratified`` (default)
    Group the positives of each game by the ``(color, shape)`` descriptor of
    their object, read from ``{split}_worlds.json``, and draw proportionally
    within each group (largest-remainder allocation, ties broken at random).
    Each ``(color, shape)`` pair satisfies exactly one of {left-only,
    right-only, both} for a given concept, so preserving the descriptor
    proportions preserves the paper's 1/3-1/3-1/3 mixture exactly, whatever
    order the generator wrote them in. Negatives are handled the same way.

``uniform``
    Uniform random draw without replacement from each half. Preserves the
    mixture in expectation rather than exactly. Used automatically as a
    fallback when the world JSON is missing or an image holds more than one
    object.

Both are seeded, so the output is reproducible.

Usage
-----
    # inspect the source without writing anything
    python data/shrink_shapeworld.py --audit --src ../data/shapeworld

    # write the shrunk copy (originals are left untouched)
    python data/shrink_shapeworld.py \
        --src ../data/shapeworld --dst ../data/shapeworld_40

``*_worlds.json(.gz)`` files are copied across unmodified. They are read per
*game* by ``shapeworld.extract_concepts`` (concept -> satisfying assignments),
which feeds the topsim concept distances and is unaffected by image
subsampling. The per-*image* ``shapeworld.extract_shapes`` would disagree with
the shrunk rows, so ``shapeworld.load_split`` only calls it when
``reference_game`` is true -- and reference games use the separate
``shapeworld_ref`` dataset, which this script is not intended for.
"""

import argparse
import gzip
import json
import os
import shutil
import sys
import zipfile

import numpy as np
from numpy.lib import format as npformat

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is required only for hdf5 IO
    h5py = None


SPLITS = ["train", "test", "test_same"]
KEEP_PER_HALF = 20
WORLD_SUFFIXES = ["_worlds.json", "_worlds.json.gz"]


# --------------------------------------------------------------------------
# Source IO
# --------------------------------------------------------------------------


def find_source(src, split):
    """Return ``(path, format)`` for a split, or ``(None, None)`` if absent."""
    npz = os.path.join(src, f"{split}.npz")
    hdf5 = os.path.join(src, f"{split}.hdf5")
    if os.path.exists(npz):
        return npz, "npz"
    if os.path.exists(hdf5):
        return hdf5, "hdf5"
    return None, None


def open_source(path, fmt):
    """
    Open a split for reading.

    hdf5 handles stay lazy, so rows can be streamed a chunk at a time. npz is
    lazy per *key* but materialises a whole array on access, so an npz source
    needs enough RAM to hold the original ``imgs`` (20-25 GB for train).
    """
    if fmt == "npz":
        return np.load(path)
    if h5py is None:
        raise RuntimeError("h5py is required to read hdf5 datasets")
    return h5py.File(path, "r")


def decode_langs(raw):
    """Langs are stored as bytes in hdf5 and (usually) as str in npz."""
    return [lang.decode("utf-8") if isinstance(lang, bytes) else str(lang) for lang in raw]


def peek_shape(path, fmt, key):
    """
    Read an array's shape without materialising it.

    Indexing an npz decompresses the whole array, which for train is 20-25 GB --
    far too much just to report a shape during an audit. The .npy header inside
    the zip carries the shape on its own.
    """
    if fmt == "hdf5":
        with h5py.File(path, "r") as f:
            return f[key].shape
    try:
        with zipfile.ZipFile(path) as z, z.open(f"{key}.npy") as f:
            version = npformat.read_magic(f)
            return npformat._read_array_header(f, version)[0]
    except Exception:
        # Private-API fallback: pay the decompression rather than fail.
        return np.load(path)[key].shape


def load_worlds(src, split):
    """Load ``{split}_worlds.json(.gz)``, or None if neither exists."""
    plain = os.path.join(src, f"{split}_worlds.json")
    gzipped = plain + ".gz"
    if os.path.exists(plain):
        with open(plain, "r") as f:
            return json.load(f)
    if os.path.exists(gzipped):
        with gzip.open(gzipped, "rt") as f:
            return json.load(f)
    return None


# --------------------------------------------------------------------------
# Descriptors + selection
# --------------------------------------------------------------------------


def world_descriptors(worlds):
    """
    Per-game, per-image ``(color, shape)`` descriptors from the world JSON.

    Returns a list of lists of hashable descriptors, or None if any image holds
    more than one object -- in which case a single ``(color, shape)`` pair does
    not identify the stratum and we fall back to a uniform draw.
    """
    descriptors = []
    for game in worlds:
        game_descriptors = []
        for img in game["imgs"]:
            if len(img) != 1:
                return None
            obj = img[0]
            game_descriptors.append((obj["color"], obj["shape"]))
        descriptors.append(game_descriptors)
    return descriptors


def stratified_choice(indices, keys, k, rng):
    """
    Draw ``k`` of ``indices`` keeping the proportions of ``keys`` intact.

    Each key group contributes ``floor(k * len(group) / len(indices))`` members
    chosen at random; the remaining slots go to the groups with the largest
    fractional remainders, ties broken at random so repeated odd-sized groups
    do not all round the same way.
    """
    groups = {}
    for idx, key in zip(indices, keys):
        groups.setdefault(key, []).append(idx)

    n = len(indices)
    exact = {key: k * len(members) / n for key, members in groups.items()}
    allocation = {key: int(np.floor(v)) for key, v in exact.items()}

    remaining = k - sum(allocation.values())
    if remaining:
        # Largest remainder first; the random tiebreak keeps the rounding
        # unbiased across games.
        order = sorted(
            groups,
            key=lambda key: (-(exact[key] - allocation[key]), rng.random()),
        )
        for key in order[:remaining]:
            allocation[key] += 1

    chosen = []
    for key, members in groups.items():
        take = allocation[key]
        if take:
            chosen.extend(rng.choice(members, size=take, replace=False))
    chosen = np.asarray(sorted(chosen), dtype=np.int64)
    assert len(chosen) == k, f"allocated {len(chosen)} of {k}"
    return chosen


def select_indices(n_img, game_descriptors, rng):
    """
    Choose ``KEEP_PER_HALF`` positives and ``KEEP_PER_HALF`` negatives.

    ``game_descriptors`` is None for a uniform draw. Returns the concatenated
    index array, positives first.
    """
    midp = n_img // 2
    pos = np.arange(midp)
    neg = np.arange(midp, n_img)

    if game_descriptors is None:
        keep_pos = np.sort(rng.choice(pos, size=KEEP_PER_HALF, replace=False))
        keep_neg = np.sort(rng.choice(neg, size=KEEP_PER_HALF, replace=False))
    else:
        keep_pos = stratified_choice(
            pos, [game_descriptors[i] for i in pos], KEEP_PER_HALF, rng
        )
        keep_neg = stratified_choice(
            neg, [game_descriptors[i] for i in neg], KEEP_PER_HALF, rng
        )

    return np.concatenate([keep_pos, keep_neg])


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def audit_split(src, split, n_games=200):
    """
    Report whether the stored image order is stratified.

    Compares the descriptor mixture of the first half of each game's positives
    against the second half. A large divergence means slicing the first 20 of
    40 would change what the task tests -- i.e. the stratified draw is doing
    real work rather than just being defensive.
    """
    path, fmt = find_source(src, split)
    if path is None:
        print(f"  {split}: absent")
        return

    imgs_shape = peek_shape(path, fmt, "imgs")
    labels_shape = peek_shape(path, fmt, "labels")
    data = open_source(path, fmt)
    labels = np.asarray(data["labels"][: min(n_games, imgs_shape[0])])
    print(f"  {split}: {path} ({fmt})  imgs={imgs_shape} labels={labels_shape}")

    midp = imgs_shape[1] // 2
    ok_layout = bool(labels[:, :midp].all() and not labels[:, midp:].any())
    print(f"    positives-then-negatives layout: {'ok' if ok_layout else 'VIOLATED'}")

    worlds = load_worlds(src, split)
    if worlds is None:
        print("    world JSON: absent -> uniform draw only")
        return

    descriptors = world_descriptors(worlds[:n_games])
    if descriptors is None:
        print("    world JSON: >1 object per image -> uniform draw only")
        return

    print(f"    world JSON: ok ({len(worlds)} games, single object per image)")

    # Per game, how much of the first half's descriptor mixture is missing from
    # the second half (and vice versa), as a total-variation distance.
    quarter = midp // 2
    tvs = []
    for game_descriptors in descriptors:
        front = game_descriptors[:quarter]
        back = game_descriptors[quarter:midp]
        keys = set(front) | set(back)
        tv = 0.5 * sum(
            abs(front.count(k) / len(front) - back.count(k) / len(back)) for k in keys
        )
        tvs.append(tv)
    tvs = np.asarray(tvs)
    print(
        f"    positives, first-half vs second-half descriptor mixture "
        f"(total variation over {len(tvs)} games): "
        f"mean={tvs.mean():.3f} median={np.median(tvs):.3f} max={tvs.max():.3f}"
    )
    print(
        "    -> ordering looks "
        + ("STRATIFIED (do not slice the first 20)" if tvs.mean() > 0.25 else "shuffled")
    )


# --------------------------------------------------------------------------
# Shrink
# --------------------------------------------------------------------------


def shrink_split(src, dst, split, rng, chunk_size, strategy, out_format=None):
    """Write the 40-image version of one split. Returns True if it was written."""
    path, fmt = find_source(src, split)
    if path is None:
        if not split.endswith("_same"):
            raise RuntimeError(f"Can't find {split}.npz or {split}.hdf5 in {src}")
        print(f"{split}: absent, skipping (optional split)")
        return False

    out_format = out_format or fmt
    data = open_source(path, fmt)

    imgs_src = data["imgs"]
    labels_src = data["labels"]
    n_games, n_img = imgs_src.shape[0], imgs_src.shape[1]
    img_shape = imgs_src.shape[2:]

    assert n_img % 2 == 0, f"{split}: odd n_img {n_img}"
    assert n_img >= 2 * KEEP_PER_HALF, f"{split}: only {n_img} images per game"
    midp = n_img // 2

    langs = decode_langs(data["langs"][:])
    assert len(langs) == n_games, f"{split}: {len(langs)} langs for {n_games} games"

    descriptors = None
    if strategy == "stratified":
        worlds = load_worlds(src, split)
        if worlds is None:
            print(f"{split}: no world JSON, falling back to a uniform draw")
        elif len(worlds) != n_games:
            print(
                f"{split}: world JSON has {len(worlds)} games but the store has "
                f"{n_games}, falling back to a uniform draw"
            )
        else:
            descriptors = world_descriptors(worlds)
            if descriptors is None:
                print(f"{split}: >1 object per image, falling back to a uniform draw")

    n_keep = 2 * KEEP_PER_HALF
    print(
        f"{split}: {n_games} games, {n_img} -> {n_keep} images "
        f"({'stratified' if descriptors is not None else 'uniform'} draw, "
        f"{fmt} -> {out_format})"
    )

    os.makedirs(dst, exist_ok=True)
    writer = _make_writer(dst, split, out_format, n_games, n_keep, img_shape,
                          imgs_src.dtype, labels_src.dtype)

    with writer as w:
        for start in range(0, n_games, chunk_size):
            stop = min(start + chunk_size, n_games)
            imgs_chunk = np.asarray(imgs_src[start:stop])
            labels_chunk = np.asarray(labels_src[start:stop])

            # The layout every downstream consumer assumes (and that
            # ConceptDataset.__getitem__ asserts): positives then negatives.
            assert labels_chunk[:, :midp].all(), f"{split}: positives not all 1"
            assert not labels_chunk[:, midp:].any(), f"{split}: negatives not all 0"

            out_imgs = np.empty((stop - start, n_keep, *img_shape), dtype=imgs_src.dtype)
            out_labels = np.empty((stop - start, n_keep), dtype=labels_src.dtype)
            for row in range(stop - start):
                keep = select_indices(
                    n_img,
                    descriptors[start + row] if descriptors is not None else None,
                    rng,
                )
                out_imgs[row] = imgs_chunk[row, keep]
                out_labels[row] = labels_chunk[row, keep]

            w.write(start, out_imgs, out_labels)
            print(f"  {stop}/{n_games}", end="\r", flush=True)

        w.write_langs(langs)

    print(f"  {n_games}/{n_games} done")

    for suffix in WORLD_SUFFIXES:
        world_src = os.path.join(src, f"{split}{suffix}")
        if os.path.exists(world_src):
            shutil.copy2(world_src, os.path.join(dst, f"{split}{suffix}"))
            print(f"  copied {split}{suffix}")
            break

    return True


class _Hdf5Writer:
    """Streams chunks straight to disk, so peak RAM is one chunk."""

    def __init__(self, path, n_games, n_keep, img_shape, img_dtype, label_dtype):
        self.path = path
        self.tmp = path + ".tmp"
        self.f = h5py.File(self.tmp, "w")
        self.imgs = self.f.create_dataset(
            "imgs", (n_games, n_keep, *img_shape), dtype=img_dtype
        )
        self.labels = self.f.create_dataset("labels", (n_games, n_keep), dtype=label_dtype)

    def write(self, start, imgs, labels):
        stop = start + imgs.shape[0]
        self.imgs[start:stop] = imgs
        self.labels[start:stop] = labels

    def write_langs(self, langs):
        # Variable-length UTF-8, which is what h5py reads back as bytes and
        # `shapeworld.load_split` decodes. Passed as a plain list because h5py
        # has no conversion path from numpy's fixed-width '<U' dtype.
        self.f.create_dataset("langs", data=list(langs), dtype=h5py.string_dtype())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.close()
        if exc[0] is None:
            os.replace(self.tmp, self.path)
        else:
            os.remove(self.tmp)


class _NpzWriter:
    """
    npz has no partial-write API, so the whole output array is held in RAM
    (~9.8 GB for train) until the final save. Prefer hdf5 for the train split.
    """

    def __init__(self, path, n_games, n_keep, img_shape, img_dtype, label_dtype):
        self.path = path
        self.imgs = np.empty((n_games, n_keep, *img_shape), dtype=img_dtype)
        self.labels = np.empty((n_games, n_keep), dtype=label_dtype)
        self.langs = None

    def write(self, start, imgs, labels):
        stop = start + imgs.shape[0]
        self.imgs[start:stop] = imgs
        self.labels[start:stop] = labels

    def write_langs(self, langs):
        # Fixed-width unicode: npz cannot store an object array without pickling,
        # which `np.load` refuses to read back by default.
        self.langs = np.asarray(langs, dtype=np.str_)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is not None:
            return
        tmp = self.path + ".tmp.npz"
        # Uncompressed: compressing ~10 GB of images costs far more time than
        # the disk it saves, and the point of this dataset is that it is read
        # once into memory.
        np.savez(tmp, imgs=self.imgs, labels=self.labels, langs=self.langs)
        os.replace(tmp, self.path)


def _make_writer(dst, split, out_format, n_games, n_keep, img_shape, img_dtype,
                 label_dtype):
    if out_format == "hdf5":
        if h5py is None:
            raise RuntimeError("h5py is required to write hdf5 datasets")
        path = os.path.join(dst, f"{split}.hdf5")
        return _Hdf5Writer(path, n_games, n_keep, img_shape, img_dtype, label_dtype)
    path = os.path.join(dst, f"{split}.npz")
    return _NpzWriter(path, n_games, n_keep, img_shape, img_dtype, label_dtype)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify_split(src, dst, split):
    """Check the written split against the source (verification step 1)."""
    src_path, src_fmt = find_source(src, split)
    dst_path, dst_fmt = find_source(dst, split)
    if dst_path is None:
        print(f"  {split}: absent in both, ok")
        return True

    src_data = open_source(src_path, src_fmt)
    dst_data = open_source(dst_path, dst_fmt)

    imgs = dst_data["imgs"]
    labels = np.asarray(dst_data["labels"])
    langs = dst_data["langs"]

    problems = []
    if imgs.shape[1] != 2 * KEEP_PER_HALF:
        problems.append(f"imgs.shape[1] == {imgs.shape[1]}, expected 40")
    if imgs.shape[0] != src_data["imgs"].shape[0]:
        problems.append(
            f"{imgs.shape[0]} games, source has {src_data['imgs'].shape[0]}"
        )
    if not labels[:, :KEEP_PER_HALF].all():
        problems.append("labels[:, :20] are not all positive")
    if labels[:, KEEP_PER_HALF:].any():
        problems.append("labels[:, 20:] are not all negative")
    if len(langs) != imgs.shape[0]:
        problems.append(f"{len(langs)} langs for {imgs.shape[0]} games")
    if decode_langs(langs[:]) != decode_langs(src_data["langs"][:]):
        problems.append("langs differ from the source")

    if problems:
        print(f"  {split}: FAILED")
        for p in problems:
            print(f"    - {p}")
        return False

    print(
        f"  {split}: ok ({imgs.shape[0]} games, {imgs.shape[1]} images, "
        f"{imgs.dtype}, langs match source)"
    )
    return True


# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Write a 40-image-per-game copy of a ShapeWorld dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src", default="../data/shapeworld",
        help="Source dataset directory (left untouched)."
    )
    parser.add_argument(
        "--dst", default="../data/shapeworld_40",
        help="Destination directory for the 40-image copy."
    )
    parser.add_argument(
        "--splits", nargs="+", default=SPLITS,
        help="Splits to process. '_same' splits are optional."
    )
    parser.add_argument(
        "--strategy", choices=["stratified", "uniform"], default="stratified",
        help="How to draw the 20 kept images from each half."
    )
    parser.add_argument(
        "--format", choices=["hdf5", "npz"], default=None,
        help="Output format. Defaults to matching the source; hdf5 streams to "
             "disk and so needs far less RAM for the train split."
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the draw.")
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Games held in memory at once while streaming."
    )
    parser.add_argument(
        "--audit", action="store_true",
        help="Report the source's layout and image ordering; write nothing."
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Re-run the integrity checks against an existing --dst."
    )
    args = parser.parse_args()

    if args.audit:
        print(f"Auditing {args.src}")
        for split in args.splits:
            audit_split(args.src, split)
        return 0

    if args.verify_only:
        print(f"Verifying {args.dst} against {args.src}")
        ok = all(verify_split(args.src, args.dst, split) for split in args.splits)
        return 0 if ok else 1

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("--src and --dst must differ; the source is not modified")

    rng = np.random.default_rng(args.seed)
    print(f"Shrinking {args.src} -> {args.dst} (seed {args.seed})")
    written = [
        split
        for split in args.splits
        if shrink_split(
            args.src, args.dst, split, rng, args.chunk_size, args.strategy,
            out_format=args.format,
        )
    ]

    print("\nVerifying:")
    ok = all(verify_split(args.src, args.dst, split) for split in written)
    print("\nAll checks passed." if ok else "\nVERIFICATION FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

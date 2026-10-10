"""
Do ShapeWorld's concept games really sample "hard" targets and distractors?

    python diagnostics/concept_strata.py [--data DIR] [--splits ...] [--draws N]

Appendix `app:hard` of the paper (and methodology.tex after it) claims that for
a disjunction 1/3 of the targets satisfy only the left disjunct, 1/3 only the
right and 1/3 both, and that for a conjunction the *distractors* are split the
same way among those failing only the left conjunct, only the right, or both.
This script measures that rather than taking it on trust.

Two levels, because there are two places it could go wrong:

  generator   The full games in `{split}_worlds.json(.gz)`. Note that the copy
              in `shapeworld_40` was carried over from the source unmodified by
              `data/shrink_shapeworld.py`, so it describes the generator's
              original 80-100-image rows, not the 40 kept. That is exactly what
              the paper's claim is about.

  per agent   What one agent sees in one game: 10 targets and 10 distractors.
              Simulated, because the shrink did not record which images it kept
              and the training loop reshuffles every epoch anyway: a
              `stratified_choice` of 20 per half (the shrink's own draw, fresh
              RNG), then a random 10/10 speaker/listener split of each half, as
              `ConceptDataset.__getitem__` + `split_spk_lis` do. 10 does not
              divide by 3, so exact thirds are impossible here by construction;
              the question is how far off, and how often a stratum is empty.

It also checks every image's label against the concept evaluated on its
`(color, shape)`, since a stratum count means nothing if the labels are wrong.

Memory. The train worlds file is 20,000 games of 80-100 objects, which
`json.load` would turn into a few GB of Python dicts. It is streamed instead,
one game at a time, and the statistics are running totals, so peak memory is
one game plus the langs (20,000 short strings).

Needs `h5py` (for the langs) and numpy, nothing from `code/`.
"""

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from shrink_shapeworld import stratified_choice  # noqa: E402

COLORS = {"red", "blue", "green", "yellow", "white", "gray"}
PER_AGENT = 10
CHUNK = 1 << 20  # characters read per refill


def iter_worlds(src, split):
    """
    Yield the games of `{split}_worlds.json(.gz)` one at a time.

    The file is a single top-level JSON array. Rather than parse it whole, keep
    a text buffer and `raw_decode` one element at a time off its front, reading
    another chunk whenever the element at the front is incomplete. Stdlib only,
    so it needs no `ijson` on the cluster.
    """
    plain = os.path.join(src, f"{split}_worlds.json")
    if os.path.exists(plain):
        f = open(plain)
    else:
        f = gzip.open(plain + ".gz", "rt")  # FileNotFoundError if neither
    decoder = json.JSONDecoder()
    with f:
        buf = f.read(CHUNK).lstrip()
        assert buf.startswith("["), "worlds file is not a JSON array"
        buf = buf[1:]
        eof = False
        while True:
            buf = buf.lstrip().lstrip(",").lstrip()
            if buf.startswith("]"):
                return
            try:
                game, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                if eof:
                    raise
                more = f.read(CHUNK)
                eof = not more
                buf += more
                continue
            yield game
            buf = buf[end:]
            if len(buf) < CHUNK and not eof:
                more = f.read(CHUNK)
                eof = not more
                buf += more


def load_langs(src, split):
    with h5py.File(os.path.join(src, f"{split}.hdf5"), "r") as f:
        raw = f["langs"][:]
    return [(x.decode() if isinstance(x, bytes) else str(x)).lower().split() for x in raw]


def parse(tokens):
    """As `code/data/shapeworld.py:_concept_to_lf`: one level, `or` before `and`."""
    for op in ("or", "and"):
        if op in tokens:
            i = tokens.index(op)
            return (op, parse(tokens[:i]), parse(tokens[i + 1:]))
    if tokens[0] == "not":
        assert len(tokens) == 2, tokens
        return ("not", (tokens[1],))
    assert len(tokens) == 1, tokens
    return (tokens[0],)


def holds(lf, obj):
    if lf[0] == "not":
        return not holds(lf[1], obj)
    if lf[0] in ("and", "or"):
        left, right = holds(lf[1], obj), holds(lf[2], obj)
        return (left and right) if lf[0] == "and" else (left or right)
    return lf[0] in obj


def operand_type(lf):
    """`not` ignored, as in `get_metadata`."""
    if lf[0] == "not":
        lf = lf[1]
    return "color" if lf[0] in COLORS else "shape"


def stratum(lf, obj):
    """
    For `or` on a target: 'left', 'right' or 'both' disjuncts satisfied.
    For `and` on a distractor: 'left', 'right' or 'both' conjuncts failed.
    """
    left, right = holds(lf[1], obj), holds(lf[2], obj)
    if lf[0] == "and":
        left, right = not left, not right
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return None  # a mislabelled image; counted separately


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data", default=os.path.expanduser("~/archive/data/emcomgen/data/shapeworld_40")
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test", "test_same"])
    parser.add_argument("--draws", type=int, default=20,
                        help="Simulated agent views per game.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    keys = ("left", "right", "both")

    for split in args.splits:
        try:
            langs = load_langs(args.data, split)
            worlds = iter_worlds(args.data, split)
            first = next(worlds)
        except FileNotFoundError as e:
            print(f"{split}: absent ({e.filename or e})")
            continue

        n_games = 0
        mislabelled = 0
        n_images = 0
        n_imgs_seen = set()
        # Running totals per concept type, so nothing grows with the split.
        # gen: sum over games of the stratum fractions; agent: sum over views of
        # the stratum counts, of the smallest count, and of views with one empty.
        n_gen = defaultdict(int)
        gen = defaultdict(lambda: np.zeros(3))
        n_view = defaultdict(int)
        agent = defaultdict(lambda: np.zeros(3))
        agent_min = defaultdict(float)
        agent_empty = defaultdict(int)

        for world in _chain(first, worlds):
            assert n_games < len(langs), f"more worlds than the {len(langs)} langs"
            lf = parse(langs[n_games])
            n_games += 1
            objs = []
            for img in world["imgs"]:
                assert len(img) == 1, "more than one object per image"
                objs.append({img[0]["color"], img[0]["shape"]})
            n = len(objs)
            n_imgs_seen.add(n)
            midp = n // 2

            for i, obj in enumerate(objs):  # positives first; audited
                n_images += 1
                mislabelled += holds(lf, obj) != (i < midp)

            if lf[0] not in ("and", "or"):
                continue
            kind = f"{lf[0]}_{'_'.join(sorted([operand_type(lf[1]), operand_type(lf[2])]))}"
            # Targets for `or`, distractors for `and`.
            idx = np.arange(midp) if lf[0] == "or" else np.arange(midp, n)
            strata = [stratum(lf, objs[i]) for i in idx]
            n_gen[kind] += 1
            gen[kind] += [strata.count(k) / len(strata) for k in keys]

            # The shrink's draw keyed on (color, shape), then per-agent halves.
            descr = [tuple(sorted(objs[i])) for i in idx]
            kept = stratified_choice(idx, descr, min(20, len(idx)), rng)
            by_index = dict(zip(idx, strata))
            for _ in range(args.draws):
                view = rng.permutation(kept)[:PER_AGENT]
                s = [by_index[i] for i in view]
                counts = [s.count(k) for k in keys]
                n_view[kind] += 1
                agent[kind] += counts
                agent_min[kind] += min(counts)
                agent_empty[kind] += min(counts) == 0

        assert n_games == len(langs), f"{n_games} worlds for {len(langs)} langs"

        print(f"\n{split}: {n_games} games, images per game {sorted(n_imgs_seen)}")
        print(f"  labels disagreeing with the concept: {mislabelled} of {n_images}")
        print(
            "  stratum = disjuncts satisfied (or, over targets) / conjuncts failed "
            "(and, over distractors)"
        )
        print(
            f"  {'type':<22}{'games':>6}  {'generator: left/right/both':>28}  "
            f"{'per agent (of 10): mean':>24}  {'min stratum':>11}  {'any empty':>9}"
        )
        for kind in sorted(gen):
            gm = " / ".join(f"{v:.2f}" for v in gen[kind] / n_gen[kind])
            am = " / ".join(f"{v:.1f}" for v in agent[kind] / n_view[kind])
            print(
                f"  {kind:<22}{n_gen[kind]:>6}  {gm:>28}  {am:>24}  "
                f"{agent_min[kind] / n_view[kind]:>11.2f}  "
                f"{agent_empty[kind] / n_view[kind]:>9.1%}"
            )


def _chain(first, rest):
    yield first
    yield from rest

if __name__ == "__main__":
    main()

"""
Write one example concept game from each of ShapeWorld and CUB to PNGs.

    python diagnostics/export_example_games.py [--out DIR] [--game N] [--seed S]

For looking at, not for training: no augmentation, no silhouetting, no
normalisation. Each dataset gets a folder holding the images of one game (40 for
ShapeWorld, 20 for CUB),
named by agent and polarity, a `sheet.png` contact sheet with one row per
agent/polarity, and `concept.txt`.

The speaker/listener division is `data/util.py:split_spk_lis`'s at
`percent_novel = 1.0`, with `k = n_examples / 2` -- 10 for ShapeWorld
(`n_examples = 20`) and 5 for CUB (`[birds.data] n_examples = 10`):

    speaker targets      positives [0, k)
    listener targets     positives [k, 2k)
    speaker distractors  negatives [0, k)
    listener distractors negatives [k, 2k)

ShapeWorld reads a stored game from `shapeworld_40` in stored order (training
reshuffles each half every epoch, so which images land with which agent varies;
the game's contents do not). CUB stores no games, so one is drawn the way
`cub.py:CUBDataset.sample_game` draws it: a species, 20 of its images, and 20
negatives each from a uniformly chosen other species -- 10 and 10 at
`n_examples = 10`. This draws from all 200
species rather than one split's.

Needs `h5py`, numpy and PIL, nothing from `code/`.
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

DATA = os.path.expanduser("~/archive/data/emcomgen/data")
SHAPEWORLD_K = 10
CUB_K = 5
ROWS = ("speaker_target", "listener_target", "speaker_distractor", "listener_distractor")
THUMB = 128


def to_pil(img):
    """CHW or HWC, uint8 or float in [0, 1]."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def save_game(out, rows, title, k):
    """`rows` maps each of ROWS to a list of (PIL image, caption-or-None)."""
    out.mkdir(parents=True, exist_ok=True)
    pad, label_h = 4, 16
    sheet = Image.new(
        "RGB",
        (k * (THUMB + pad) + pad, 24 + len(ROWS) * (THUMB + label_h + pad)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 4), title, fill="black")
    for r, name in enumerate(ROWS):
        y = 24 + r * (THUMB + label_h + pad)
        draw.text((pad, y), name.replace("_", " "), fill="black")
        for c, (img, _) in enumerate(rows[name]):
            img.save(out / f"{name}_{c:02d}.png")
            thumb = img.copy()
            thumb.thumbnail((THUMB, THUMB))
            sheet.paste(thumb, (pad + c * (THUMB + pad), y + label_h))
    sheet.save(out / "sheet.png")

    with open(out / "concept.txt", "w") as f:
        f.write(title + "\n")
        for name in ROWS:
            for c, (_, caption) in enumerate(rows[name]):
                if caption:
                    f.write(f"{name}_{c:02d}.png\t{caption}\n")
    print(f"  wrote {out}  ({title})")


def shapeworld(src, split, game, rng, out):
    with h5py.File(os.path.join(src, f"{split}.hdf5"), "r") as f:
        n_games = f["imgs"].shape[0]
        if game is None:
            # A conjunction or disjunction, since those are what the hard
            # sampling is about; a primitive like `blue` shows little.
            langs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["langs"][:]]
            compound = [i for i, x in enumerate(langs) if {"and", "or"} & set(x.split())]
            game = int(rng.choice(compound)) if compound else int(rng.integers(n_games))
        imgs = f["imgs"][game]
        labels = f["labels"][game]
        lang = f["langs"][game]
    lang = lang.decode() if isinstance(lang, bytes) else str(lang)
    k = SHAPEWORLD_K
    midp = len(labels) // 2
    assert labels[:midp].all() and not labels[midp:].any(), "layout is not pos-then-neg"

    pos, neg = imgs[:midp], imgs[midp:]
    rows = {
        "speaker_target": pos[:k],
        "listener_target": pos[k:2 * k],
        "speaker_distractor": neg[:k],
        "listener_distractor": neg[k:2 * k],
    }
    rows = {name: [(to_pil(i), None) for i in v] for name, v in rows.items()}
    save_game(out, rows, f"ShapeWorld {split} game {game}: {lang}", k)


def cub(src, rng, out):
    img_dir = Path(src) / "CUB_200_2011" / "images"
    species = sorted(p.name for p in img_dir.iterdir() if (p / "img.npz").exists())
    target = species[rng.integers(len(species))]

    def load(sp):
        with np.load(img_dir / sp / "img.npz") as z:
            return {k: z[k] for k in z.files}

    k = CUB_K
    pos = load(target)
    names = rng.choice(sorted(pos), size=2 * k, replace=False)
    pos_imgs = [(to_pil(pos[n]), f"{target}  {Path(n).name}") for n in names]

    others = [s for s in species if s != target]
    cache = {}
    neg_imgs = []
    for _ in range(2 * k):
        sp = others[rng.integers(len(others))]
        if sp not in cache:
            cache[sp] = load(sp)
        n = rng.choice(sorted(cache[sp]))
        neg_imgs.append((to_pil(cache[sp][n]), f"{sp}  {Path(n).name}"))

    rows = {
        "speaker_target": pos_imgs[:k],
        "listener_target": pos_imgs[k:],
        "speaker_distractor": neg_imgs[:k],
        "listener_distractor": neg_imgs[k:],
    }
    save_game(out, rows, f"CUB: {target}", k)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shapeworld", default=os.path.join(DATA, "shapeworld_40"))
    parser.add_argument("--cub", default=os.path.join(DATA, "cub"))
    parser.add_argument("--out", default="example_games")
    parser.add_argument("--split", default="train", help="ShapeWorld split.")
    parser.add_argument("--game", type=int, default=None,
                        help="ShapeWorld game index; random if omitted.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)

    shapeworld(args.shapeworld, args.split, args.game, rng, out / "shapeworld")
    cub(args.cub, rng, out / "cub")


if __name__ == "__main__":
    main()

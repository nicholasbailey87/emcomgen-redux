"""
Is shape still there after the silhouette, and does it survive the eval gap?

    python diagnostics/silhouette_shape_probe.py [--data DIR] [--games N]

The intervention silhouettes at training time only; eval is never silhouetted
(`docs/data.md`). So a run that learns shape under silhouetting has to do two
things, and they are separable:

  1. read shape off a repainted object at all, and
  2. carry that reading across to un-repainted images at eval.

This trains a fresh `Conv4` — the sender's own vision model, at its own input
size — to classify the concept attribute of ShapeWorld positives under each
fill, and then tests it twice: once on held-out images repainted the same way
(`in_domain`, question 1) and once on the same held-out images left clean
(`clean`, question 2). Games are split before anything is rendered, so no game
contributes to both halves.

Four arms. `clean` is the ceiling -- what this probe reads when nothing has been
taken away -- and the other three are the live `silhouette` at three fills:

  white       (1.0, 1.0, 1.0), the fill in force from 536c59e to e884662
  fill        the current default, ShapeWorld's mean object colour
  half        a flat 0.5, the fill in force from e884662 to 2026-09-01

All three repaint by threshold, because that is what the live code does. The
coverage blend this script was written to indict is gone as of 2026-09-01 and
lives at `f7dc0de`; what it read, and what decided that change:

  arm                in_domain    clean
  clean                  0.999    0.999
  white_threshold        0.794    0.560
  white_coverage         1.000    0.483
  fill_coverage          1.000    0.486     (chance 0.306, job 123354)

The coverage arms were perfectly readable under their own repainting and lost
almost all of it at eval. Note what that leaves open, and what this script now
exists to close: `white_threshold` was measured at full white, and the fill
adopted is the chromatic one, so no arm has ever read the transform as it now
stands. `white` and `fill` here are the two halves of that question.

Colour is the control, on the same games and the same split. A silhouette is
supposed to erase it, so colour under any repainting arm should sit at chance;
where it does not, the arm is leaking. Under a threshold the output is
two-valued, so there is no anti-aliased ramp left to carry colour and the coverage
leak documented in docs/data.md is gone -- but `half` is expected to leak anyway,
and for a different reason: `0.5 * 255 = 128` is exactly ShapeWorld's `gray`, so
a grey object under that arm comes back bit-identical to itself and one colour in
six is not repainted at all.

Reading the table:

  * an arm's in_domain high, clean high      -> shape survives and transfers
  * in_domain high, clean at chance          -> readable, but the eval gap eats
                                                it; the augmentation teaches a
                                                different problem
  * in_domain at chance                      -> the repainting destroyed shape,
                                                and no run under that fill could
                                                have learned shape

This measures the *data*, not any run: nothing here is loaded from a checkpoint,
and every arm gets the same architecture, the same optimiser and the same games.
It answers what is available to be learned, not what a given run did learn — for
that, see `probe_shape.py`, which sweeps a trained sender across its checkpoints.
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

from data.generic import DEFAULT_SILHOUETTE_FILL, silhouette  # noqa: E402
from models.backbone import vision  # noqa: E402

SHAPES = ["circle", "cross", "ellipse", "pentagon",
          "rectangle", "semicircle", "square", "triangle"]
COLOURS = ["red", "green", "blue", "yellow",
           "magenta", "cyan", "gray", "grey", "white"]

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# Every repainting arm goes through the live `silhouette`, so this script says
#     what training does rather than what it did when the script was written.
ARMS = {
    "clean": lambda x: x,
    "white": lambda x: silhouette(x, 1.0),
    "fill": lambda x: silhouette(x, DEFAULT_SILHOUETTE_FILL),
    "half": lambda x: silhouette(x, 0.5),
}


def bare_concept_games(langs, words, max_games, rng):
    """Games whose concept is exactly one of `words`, and their class labels."""
    games = [i for i, s in enumerate(langs) if s.strip() in words]
    rng.shuffle(games)
    games = sorted(games[:max_games])
    classes = sorted({langs[i].strip() for i in games})
    lab = {c: j for j, c in enumerate(classes)}
    y = np.array([lab[langs[i].strip()] for i in games])
    return games, y, classes


def load_positives(f, games):
    """The 20 positives of each game, as (n_games, 20, 3, H, W) uint8."""
    return np.stack([np.asarray(f["imgs"][i])[:20] for i in games])


def render(imgs, arm, chunk=512):
    """Apply an arm's fill to a (n, 3, H, W) uint8 array, on `DEV`."""
    fn = ARMS[arm]
    out = []
    for i in range(0, len(imgs), chunk):
        block = torch.as_tensor(imgs[i:i + chunk], device=DEV)
        out.append(fn(block).cpu())
    return torch.cat(out)


def train_probe(X_tr, y_tr, evals, epochs, batch, lr, seed):
    """
    Fit Conv4 + a linear head on `X_tr`, then score each (name, X, y) in `evals`.

    Inputs are uint8 tensors; scaling by 255 is done per batch, which is what
    `probe_shape.py` does and what the training pipeline's `vis_input` amounts
    to on ShapeWorld.
    """
    torch.manual_seed(seed)
    n_class = int(y_tr.max()) + 1
    net = vision.Conv4().to(DEV)
    head = nn.Linear(net.final_feat_dim, n_class).to(DEV)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=lr)

    y_tr_t = torch.as_tensor(y_tr, dtype=torch.long, device=DEV)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(X_tr), generator=g).split(batch):
            xb = X_tr[idx].to(DEV, non_blocking=True).float() / 255.0
            opt.zero_grad()
            nn.functional.cross_entropy(head(net(xb)), y_tr_t[idx]).backward()
            opt.step()

    net.eval()
    scores = {}
    with torch.no_grad():
        for name, X, y in evals:
            hits = 0
            y_t = torch.as_tensor(y, dtype=torch.long, device=DEV)
            for i in range(0, len(X), batch):
                xb = X[i:i + batch].to(DEV).float() / 255.0
                hits += (head(net(xb)).argmax(1) == y_t[i:i + batch]).sum().item()
            scores[name] = hits / len(X)
    return scores


def main(args):
    rng = np.random.RandomState(args.seed)
    path = os.path.join(os.path.expanduser(args.data), args.split + ".hdf5")
    f = h5py.File(path, "r")
    langs = [l.decode() if isinstance(l, (bytes, np.bytes_)) else str(l)
             for l in f["langs"][:]]
    print("data     ", path, "|", args.split, f"({len(langs)} games)")
    print("device   ", DEV, "| epochs", args.epochs, "| games", args.games)

    for attribute, words in (("shape", SHAPES), ("colour", COLOURS)):
        games, y_game, classes = bare_concept_games(langs, words, args.games, rng)
        if len(classes) < 2:
            print(f"\n[skip] {attribute}: only {len(classes)} bare concepts")
            continue

        # Split by game, before anything is rendered: the 20 positives of one
        #     game are near-duplicates of each other, so splitting by image
        #     would put the same object on both sides.
        order = rng.permutation(len(games))
        cut = int(0.7 * len(games))
        tr_g, te_g = order[:cut], order[cut:]

        imgs = load_positives(f, games)
        y_img = np.repeat(y_game, imgs.shape[1])
        n, k = imgs.shape[:2]
        flat = imgs.reshape(n * k, *imgs.shape[2:])
        tr_i = np.concatenate([np.arange(g * k, (g + 1) * k) for g in tr_g])
        te_i = np.concatenate([np.arange(g * k, (g + 1) * k) for g in te_g])

        _, counts = np.unique(y_img[te_i], return_counts=True)
        chance = counts.max() / len(te_i)
        print(f"\n{attribute}: {len(games)} games over {len(classes)} concepts "
              f"{classes}")
        print(f"  {len(tr_i)} train images, {len(te_i)} test, chance {chance:.3f}")

        clean_te = render(flat[te_i], "clean")
        head_row = f"  {'arm':<18}{'in_domain':>11}{'clean':>9}"
        print(head_row)
        print("  " + "-" * (len(head_row) - 2))
        for arm in ARMS:
            X_tr = render(flat[tr_i], arm)
            X_te = render(flat[te_i], arm)
            evals = [("in_domain", X_te, y_img[te_i]), ("clean", clean_te, y_img[te_i])]
            s = train_probe(X_tr, y_img[tr_i], evals, args.epochs, args.batch,
                            args.lr, args.seed)
            print(f"  {arm:<18}{s['in_domain']:>11.3f}{s['clean']:>9.3f}", flush=True)
        print(f"  {'chance':<18}{chance:>11.3f}{chance:>9.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="~/sharedscratch/data/emcomgen/data/shapeworld_40")
    p.add_argument("--split", default="train")
    p.add_argument("--games", type=int, default=600,
                   help="bare-concept games per attribute, before the 70/30 split")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())

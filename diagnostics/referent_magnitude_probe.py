"""
Is what `referent_layer_norm` deletes signal or nuisance?

    python diagnostics/referent_magnitude_probe.py --run RUN_DIR [--games N]
    python diagnostics/referent_magnitude_probe.py --config PATH --untrained

`BilinearDiscriminator` scores candidate `j` as `LN(r_j) . proj`, and
`referent_layer_norm` is non-affine, so `LN` deletes exactly two numbers per
candidate: the mean and the standard deviation of `r_j` over its feature axis.
The norm went in at `4248fca` on the premise that those are nuisance -- "no
candidate is read loudly for being large" -- and that premise has never been
measured.

It matters because a proposal to reinstate them is only worth building if they
carry something. The version this was written for keeps the direction normalised
but restores the *relative* magnitudes within a game: standardise the 20
candidates' means and sds across the candidate axis, then put them back, so a
candidate that was larger than its neighbours stays larger while the backbone's
absolute scale still divides out. Within a game rather than across the batch,
which keeps the statistic self-contained and leaves no train/eval gap.

This script measures the thing that decides it. For every game it takes the
tensor `Receiver.forward` hands to the discriminator -- `adapter(feature_model(
lis_inp))`, before `input_dropout` -- and asks whether the two deleted numbers
predict `lis_label`.

Five columns, all computed within a game and then averaged over games:

  spread_mu, spread_sd   the within-game coefficient of variation of the two
                         statistics. If these are ~0 there is nothing to
                         reinstate whatever the correlations say, and the rest
                         of the table is moot.
  auc_mu, auc_sd         rank AUC of each statistic against the label, chance
                         0.5. Symmetric about chance: 0.3 is as informative as
                         0.7, in the other direction.
  direction              a linear probe on `LN(r)` alone, held out by game
  direction+magnitude    the same probe with the two within-game standardised
                         statistics appended

`direction+magnitude - direction` is the answer. If it is ~0 the norm is right
and the proposal buys nothing; if it is positive the magnitudes carry something
the direction does not, and the size of the gap is the ceiling on what
reinstating them could be worth.

**Read `delta`, not `auc_sd`.** On synthetic embeddings built so that only the
*direction* carries the label -- a fixed vector added to the positives --
`auc_sd` still reads 1.000, because adding any vector to noise moves the norm of
what it was added to. Magnitude is a shadow of direction in general, and a high
`auc_*` on its own is not evidence that reinstating the magnitudes would buy
anything. The four synthetic cells the columns were checked against:

    arm              spread_sd   auc_sd  direction  dir+mag   delta
    neither              0.087    0.514      0.501    0.497   -0.003
    magnitude only       0.307    1.000      0.489    1.000   +0.511
    direction only       0.219    1.000      0.997    1.000   +0.003
    both                 0.478    1.000      0.997    1.000   +0.003

`delta` is the only column that separates the four. `auc_*` and `spread_*` are
descriptive: they say whether there is variation and which way it points, not
whether it is worth keeping.

**The silhouette control, and why it is not optional.** Silhouetting is applied
to the listener's referents at training time only and eval is never silhouetted
(docs/data.md). A repainted object has different pixel statistics from a clean
one, so embedding magnitude is a candidate carrier for "was this silhouetted" --
a cue that exists in training and not at eval. `--silhouette both` runs the whole
table twice and adds `auc_silhouetted`, the rank AUC of each statistic against
that indicator rather than against the label. A high `auc_silhouetted` with a
chance `auc_*` says the magnitudes are carrying the augmentation and nothing
else, and reinstating them would be reinstating a train-only artefact.

**BatchNorm mode.** `ResNet18SmallInput` opens with `BatchNorm2d` and an
untrained network's running statistics are meaningless, so reading an untrained
backbone under `eval()` gives magnitudes that are an artefact of that rather than
of the architecture -- it has been off by 12x. `--bn auto` therefore reads an
untrained model in `train()` and a checkpoint in `eval()`, which is the mode each
one is honest in. Override it only deliberately.

Nothing here trains the agents. The probe head is the only thing fit, and it is
fit on frozen embeddings.
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

import data.loader          # noqa: E402
import models.builder       # noqa: E402
import parse_config         # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rank_auc(scores, labels):
    """
    Mann-Whitney U over one game, as a fraction of its maximum.

    Ties share their rank, so a constant score gives exactly 0.5 rather than
        whichever way the sort happened to fall.
    """
    labels = labels.astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average the ranks inside each tied block.
    for value in np.unique(scores):
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def embed(receiver, referents, bn_mode, batch):
    """
    The tensor `Receiver.forward` hands to the discriminator, for one split.

    `adapter(feature_model(...))` and nothing after it: `input_dropout` is a
        regulariser and is off here, and the discriminator's own norm is the
        thing under test.
    """
    was_training = receiver.training
    receiver.train(bn_mode == "train")
    out = []
    with torch.no_grad():
        for i in range(0, len(referents), batch):
            block = referents[i:i + batch].to(DEV).float()
            n_game, n_obj = block.shape[0], block.shape[1]
            flat = block.view(n_game * n_obj, *block.shape[2:])
            e = receiver.adapter(receiver.feature_model(flat))
            out.append(e.view(n_game, n_obj, -1).float().cpu())
    receiver.train(was_training)
    return torch.cat(out)


def fit_probe(train_x, train_y, test_x, test_y, seed, steps=400, lr=0.05):
    """
    A logistic regression on frozen features, held out by game.

    Full-batch Adam rather than a solver so the same code path serves both
        feature sets and neither gets a convergence advantage the other does
        not. Reports accuracy at the module's own `> 0` threshold.
    """
    torch.manual_seed(seed)
    head = nn.Linear(train_x.shape[-1], 1).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    tx, ty = train_x.to(DEV), train_y.to(DEV).float()
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(
            head(tx).squeeze(-1), ty
        ).backward()
        opt.step()
    with torch.no_grad():
        pred = head(test_x.to(DEV)).squeeze(-1) > 0
        return (pred.cpu() == test_y.bool()).float().mean().item()


def within_game_standardise(x):
    """Zero mean, unit sd over the candidate axis of a (games, candidates) array."""
    mu = x.mean(1, keepdim=True)
    sd = x.std(1, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd


def table(embedded, labels, seed):
    """Every column of the report, for one silhouette arm."""
    mu = embedded.mean(-1)                       # (games, candidates)
    sd = embedded.std(-1)
    lab = labels.numpy()

    aucs = {
        "auc_mu": np.nanmean([rank_auc(mu[g].numpy(), lab[g]) for g in range(len(mu))]),
        "auc_sd": np.nanmean([rank_auc(sd[g].numpy(), lab[g]) for g in range(len(sd))]),
    }
    spread = {
        "spread_mu": (mu.std(1) / mu.mean(1).abs().clamp_min(1e-6)).mean().item(),
        "spread_sd": (sd.std(1) / sd.mean(1).abs().clamp_min(1e-6)).mean().item(),
    }

    # The direction the discriminator actually sees, and the two numbers the
    #     norm removed on the way. Standardised within the game, which is the
    #     proposal's own reconstruction.
    direction = nn.functional.layer_norm(embedded, (embedded.shape[-1],))
    magnitude = torch.stack(
        (within_game_standardise(mu), within_game_standardise(sd)), dim=-1
    )

    n_train = int(len(embedded) * 0.7)
    flat = lambda t, s: t[s].reshape(-1, t.shape[-1])
    tr, te = slice(0, n_train), slice(n_train, None)
    y_tr, y_te = labels[tr].reshape(-1), labels[te].reshape(-1)

    dir_only = fit_probe(flat(direction, tr), y_tr, flat(direction, te), y_te, seed)
    both = torch.cat((direction, magnitude), dim=-1)
    dir_mag = fit_probe(flat(both, tr), y_tr, flat(both, te), y_te, seed)

    return {**spread, **aucs, "direction": dir_only,
            "direction+magnitude": dir_mag, "delta": dir_mag - dir_only}


def main(args):
    config_path = args.config or os.path.join(args.run, "config.toml")
    config = parse_config.get_config(config_path)
    if args.games:
        config['data']['batch_size'] = min(config['data']['batch_size'], args.games)

    dataloaders = data.loader.load_dataloaders(config)
    pair = models.builder.build_models(dataloaders, config)
    receiver = pair.receiver.to(DEV)

    if args.run and not args.untrained:
        ckpt = args.checkpoint or _find_checkpoint(args.run)
        state = torch.load(ckpt, map_location=DEV)
        state = state.get("model", state)
        missing = pair.load_state_dict(state, strict=False)
        print("checkpoint", ckpt)
        if missing.missing_keys:
            print("  missing keys:", len(missing.missing_keys),
                  "-- a checkpoint written before the listener split will not load")
    else:
        print("checkpoint  (none: untrained)")

    bn = args.bn if args.bn != "auto" else ("train" if args.untrained else "eval")
    print("config     ", config_path)
    print("backbone   ", config['receiver']['feature_model'],
          "| BatchNorm in", bn + "()",
          "| silhouette_p_receiver", config['data']['silhouette_p_receiver'])
    print("device     ", DEV)

    referents, labels, silhouetted = _collect(dataloaders, config, args)
    print(f"games       {len(referents)} x {referents.shape[1]} candidates")
    print()

    rows = {}
    if args.silhouette in ("on", "both"):
        rows["silhouetted"] = table(
            embed(receiver, referents, bn, args.batch), labels, args.seed)
    if args.silhouette in ("off", "both"):
        rows["clean"] = table(
            embed(receiver, _clean(referents, silhouetted), bn, args.batch),
            labels, args.seed)

    cols = ["spread_mu", "spread_sd", "auc_mu", "auc_sd",
            "direction", "direction+magnitude", "delta"]
    print(f"{'arm':<14}" + "".join(f"{c:>21}" for c in cols))
    for name, row in rows.items():
        print(f"{name:<14}" + "".join(f"{row[c]:>21.4f}" for c in cols))
    print()
    print("chance: auc 0.5, direction 0.5. `delta` is what reinstating the")
    print("magnitudes could be worth; a spread_* near zero makes the rest moot.")


def _find_checkpoint(run):
    for name in ("final_model.pt", "checkpoint_last.pt"):
        path = os.path.join(run, name)
        if os.path.exists(path):
            return path
    raise SystemExit(f"no final_model.pt or checkpoint_last.pt in {run}")


def _collect(dataloaders, config, args):
    """
    `--games` games' listener sets, their labels, and which were silhouetted.

    Taken from the *train* loader because that is the only split the
        augmentation runs on, and the silhouette question is about training.
    """
    referents, labels = [], []
    for batch in dataloaders["train"]:
        _, _, lis_inp, lis_label = batch[:4]
        referents.append(lis_inp)
        labels.append(lis_label)
        if sum(len(r) for r in referents) >= args.games:
            break
    referents = torch.cat(referents)[:args.games]
    labels = torch.cat(labels)[:args.games]
    return referents, labels, None


def _clean(referents, _silhouetted):
    """
    Placeholder for the un-silhouetted counterpart.

    The loader silhouettes inside `__getitem__`, so a clean copy of the same
        games needs a second loader built with `silhouette_p_receiver = 0.0`
        and the same seed. `--silhouette both` is wired but this half is not
        implemented; run the two arms as two invocations against two configs
        until it is.
    """
    raise SystemExit(
        "--silhouette both/off needs a second config at silhouette_p_receiver "
        "= 0.0; run this twice and compare, or implement _clean."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="a run directory holding config.toml and a .pt")
    p.add_argument("--config", help="a config to build from instead of --run's")
    p.add_argument("--checkpoint", help="override which .pt in --run to load")
    p.add_argument("--untrained", action="store_true",
                   help="skip the checkpoint: is magnitude informative at init?")
    p.add_argument("--games", type=int, default=600)
    p.add_argument("--batch", type=int, default=16,
                   help="games per forward, not images")
    p.add_argument("--bn", choices=["auto", "train", "eval"], default="auto")
    p.add_argument("--silhouette", choices=["on", "off", "both"], default="on")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not args.run and not args.config:
        p.error("one of --run or --config is required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)

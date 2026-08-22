"""
Run the whole pair, with the vision models taken out of the argument.

`comparer_probe.py` asks whether the listener's comparer can use a message. It
can -- on rung 12 it reaches 100% in under two hundred steps at the config's own
learning rate, on dense codes and on discrete symbol sequences alike, and it is
more robust to unseparated referents than the bilinear baseline. That leaves the
one thing a probe with a fixed protocol cannot test: the comparer in a loop with
a speaker that is still learning, where each half only becomes useful because
the other one already is.

So this script keeps the speaker, the prototyper, the language model, the
Gumbel channel and the comparer exactly as `models.builder` builds them, and
replaces only the two vision models with a frozen embedding: every referent is
one of `--concepts` fixed random vectors plus Gaussian noise. That is a vision
model that has already succeeded. It removes "the ViT has not learned to
separate species yet" as an explanation without removing anything else, and it
is what makes the loop cheap enough to run on a laptop.

The measurement is the divergence itself. `metrics.csv` says a working run takes
off in a fixed order -- the speaker's prototyper leaves uniform pooling, the
polarity tag separates, the logit scale traverses and the channel opens, and
only then does accuracy move -- so this prints those same columns, under the
same names, and the question is which of them moves.

The reference: rung 10 on CUB, which works. Its prototyper left uniform pooling
at epoch 2 (`pool_effective_examples` 4.99 -> 4.43), `polarity_separation` went
0.10 -> 0.49 in the same epoch and reached 13.2 by epoch 29, `logit_scale` rose
0.91 -> 2.24 and `realised_survival` 0.22 -> 0.85. Rung 12's prototyper never
departs 4.99 in thirty epochs and its logit scale falls.

Steps here are not epochs -- a rung 10 epoch is 3100 games -- so expect the
order of events rather than the timings.

What it has already settled. Run against rung 12 at the config's own 1e-4, this
is what identified the listener's readout as the cause and not the comparer, the
channel or the speaker: with the readout standardised at a fixed gain it reached
accuracy 0.606 and `polarity_separation` 3.64 at 2500 steps, and with a plain
linear readout -- the same module, one change -- 0.863 and 8.17, taking off at
step ~1600 by the same route rung 10 takes. That is what removed the batch norm
from `TransformerCrossAttentionComparer`. See `diagnostics/README.md`.

One thing it is not for: choosing a learning rate. At 1e-3 *both* comparers
saturate their logits within a hundred steps and freeze, rung 10 included --
accuracy 0.59 against 1.000 at 1e-4. The frozen vision and the linearly
separable concepts make this a poor model of where the real run's optimum is.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import models.builder   # noqa: E402
import parse_config     # noqa: E402
import train            # noqa: E402  -- for `clip_gradients`, the real one

CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "ablation", "configs"
)
FEATS = {"cub": (3, 224, 224), "shapeworld": (3, 64, 64)}


class FrozenConceptVision(nn.Module):
    """
    A vision model that has already won.

    Takes concept ids where the real model takes images -- the pair reshapes
        `(batch, n_obj, *rest)` to `(batch * n_obj, *rest)` and hands that
        straight to `feat_model`, so `rest = (1,)` carrying an id is all this
        needs to accept. Returns a fixed random vector per concept plus noise,
        with `requires_grad=False` throughout, because a vision model that
        learns is a suspect and the point of this probe is to have only one.
    """

    def __init__(self, n_concepts, final_feat_dim, noise):
        super().__init__()
        self.final_feat_dim = final_feat_dim
        self.noise = noise
        self.register_buffer("prototypes", torch.randn(n_concepts, final_feat_dim))

    def forward(self, ids):
        embedded = self.prototypes[ids.reshape(-1).long()]
        return embedded + self.noise * torch.randn_like(embedded)

    def reset_parameters(self):
        """Nothing to reset; kept so `Pair.reset_parameters` still works."""


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=os.path.join(CONFIG_DIR, "12_birds_receiver_cross_attention.toml"),
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--lr", type=float, default=None,
        help="base learning rate; defaults to the config's own optimiser.lr. "
             "Every group is scaled by the same factor, so the elevated rates "
             "for the logit scale and the polarity tag keep their ratio to the "
             "base -- changing the base and the ratio at once would leave two "
             "explanations for whatever happens",
    )
    parser.add_argument("--concepts", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build(config_path, concepts, noise, lr=None):
    """The real pair and the real optimiser, with the vision models swapped."""
    config = parse_config.get_config(config_path)
    config["cuda"] = False

    name = "cub" if "cub" in config["data"]["dataset"] else "shapeworld"

    class _Dataset:
        n_feats = FEATS[name]

    _Dataset.name = name

    class _Loader:
        dataset = _Dataset()

    built = models.builder.build_models({"train": _Loader()}, config)
    pair, optimiser = built["pair"], built["optimiser"]

    # After the build, so that every width the pair derived from
    #     `final_feat_dim` is the width the real rung uses.
    pair.sender.feat_model = FrozenConceptVision(
        concepts, pair.sender.feat_model.final_feat_dim, noise
    )
    pair.receiver.feature_model = FrozenConceptVision(
        concepts, pair.receiver.feature_model.final_feat_dim, noise
    )

    # The optimiser was built over the real vision parameters, which no longer
    #     exist. Dropping those groups leaves every other group -- and the
    #     elevated rates for the logit scale and the polarity tag, which are
    #     what the take-off depends on -- exactly as configured.
    scale = 1.0 if lr is None else lr / config["optimiser"]["lr"]
    live = {id(p) for p in pair.parameters()}
    for group in optimiser.param_groups:
        group["params"] = [p for p in group["params"] if id(p) in live]
        group["lr"] *= scale
    optimiser.param_groups = [g for g in optimiser.param_groups if g["params"]]

    return config, pair, optimiser


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    config, pair, optimiser = build(
        args.config, args.concepts, args.noise, args.lr
    )
    pair.train()

    batch = config["data"]["batch_size"]
    n_obj = config["data"]["n_examples"]
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"{os.path.basename(args.config)}")
    print(f"  speaker  : {type(pair.sender.language_model).__name__} / "
          f"{type(pair.sender.prototyper).__name__}")
    print(f"  listener : {type(pair.receiver.comparer).__name__}")
    rates = sorted({group["lr"] for group in optimiser.param_groups})
    print(f"  batch {batch} x {n_obj} slots, {args.concepts} concepts, "
          f"noise {args.noise}, {args.steps} steps")
    print(f"  rates    : {', '.join(f'{rate:.2g}' for rate in rates)}")
    print(f"  vision   : frozen prototypes, both agents\n")

    # The listener's own column, where it has one. `decision_kurtosis` reads
    #     the shape of its scores: negative means bimodal, which is what
    #     discriminating looks like; sustained positive alongside a flat `acc`
    #     means a listener with nothing to say.
    comparer = pair.receiver.comparer
    has_kurtosis = hasattr(comparer, "decision_kurtosis")

    header = (
        f"{'step':>6} {'loss':>7} {'acc':>7} {'pool_eff':>9} {'pool_norm':>10} "
        f"{'polarity':>9} {'lgt_scale':>10} {'survival':>9} {'spread':>7}"
        + (f" {'kurtosis':>9}" if has_kurtosis else "")
    )
    print(header)
    print("-" * len(header))

    def make_batch():
        """
        The speaker's view has positives first -- `Sender.get_prototypes`
            asserts it -- and the listener's is shuffled, as `split_spk_lis`
            arranges for the real game. Both halves are clusters, so the
            partition is free and only the polarity needs the channel.
        """
        positive = torch.randint(args.concepts, (batch,))
        negative = (positive + torch.randint(1, args.concepts, (batch,))) % args.concepts

        ids = negative[:, None].expand(batch, n_obj).clone()
        ids[:, : n_obj // 2] = positive[:, None]
        y = torch.zeros(batch, n_obj)
        y[:, : n_obj // 2] = 1.0

        perm = torch.argsort(torch.rand(batch, n_obj), dim=1)
        return (
            ids[..., None].float(), y,                                  # speaker
            torch.gather(ids, 1, perm)[..., None].float(),              # listener
            torch.gather(y, 1, perm),
        )

    for step in range(args.steps + 1):
        spk_inp, spk_y, lis_inp, lis_y = make_batch()

        lang, _ = pair.sender(spk_inp, spk_y)
        scores = pair.receiver(lis_inp, lang)
        loss = criterion(scores, lis_y)

        optimiser.zero_grad()
        loss.backward()
        train.clip_gradients(pair, config["optimiser"]["clip_grad_norm"])
        optimiser.step()

        if step % args.every:
            continue

        with torch.no_grad():
            acc = ((scores > 0).float() == lis_y).float().mean().item()

        language_model = pair.sender.language_model
        prototyper = pair.sender.prototyper
        line = (
            f"{step:6d} {loss.item():7.4f} {acc:7.4f} "
            f"{prototyper.pool_effective_examples:9.4f} "
            f"{prototyper.pool_score_norm:10.4f} "
            f"{language_model.polarity_separation:9.4f} "
            f"{language_model.logit_scale.item():10.4f} "
            f"{language_model.realised_survival:9.4f} "
            f"{language_model.logit_spread:7.4f}"
        )
        if has_kurtosis:
            line += f" {comparer.decision_kurtosis:+9.2f}"
        print(line)


if __name__ == "__main__":
    main()

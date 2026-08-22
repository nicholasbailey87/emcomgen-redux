"""
Ask the listener's comparer one question at a time, with nothing else running.

A stuck run has too many suspects. The speaker may have nothing to say, the
Gumbel channel may be shut, the vision models may not have learned a
representation, the loss may have found a way to go quiet, and every one of
those looks the same in `metrics.csv`: accuracy at 0.5 and a loss near ln 2.
This script removes all of them but one. There is no dataset, no vision model,
no speaker and no channel -- the comparer is built exactly as `train.py` builds
it, from a real rung config, and then handed a synthetic game directly.

The game. Each row is `n_examples` slots: half are one concept, half another,
and both concepts are drawn from a fixed pool of random prototype vectors with
Gaussian noise on top. The message is that concept's fixed code, noise-free.
Slots are shuffled, so position carries nothing.

The three things it can tell you, and they are different questions:

  --message informative --distractors clustered   (the default)
      Can the comparer use a message at all? Both halves are clusters, so
      clustering the referents recovers the partition but not which half is
      positive. Exactly one bit is missing and only the message carries it. If
      this does not reach high accuracy, the fault is in the comparer and no
      amount of channel opening or loss reshaping will help.

  --message scrambled --distractors clustered
      The control, and the one that reproduces a failing run. Identical in
      every respect except that the message names an unrelated concept. The
      comparer cannot win, so whatever it does here is what "nothing to say"
      looks like in the metrics columns.

  --message scrambled --distractors varied
      The concept game's own shortcut. The distractors are all different
      concepts, so the positives are the only repeated ones and
      `referent_self_attention` can find them without reading anything. A run
      that scores well this way has learned to cluster, not to talk.

What to read. Accuracy and loss, but mainly `excess_kurt` -- the excess
kurtosis of the scores, which reads their *shape* where accuracy and spread read
their size. Negative means bimodal, which is what a discriminating listener
produces and which floors at -2; sustained positive alongside accuracy at 0.5 is
a listener with nothing to say, dumping its magnitude into a few outliers while
the bulk sits at sigmoid 0.5. See `diagnostics/README.md` for the readings this
has produced.

Runs on CPU in about a minute. It builds the whole pair because that is what
`models.builder` exposes, then uses only the comparer.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import models.builder   # noqa: E402
import models.receiver  # noqa: E402
import parse_config     # noqa: E402

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..", "experiments", "ablation", "configs",
    "12_birds_receiver_cross_attention.toml",
)

# Only used to construct the vision models, which this probe never calls.
FEATS = {"cub": (3, 224, 224), "shapeworld": (3, 64, 64)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--message", choices=("informative", "scrambled"), default="informative",
        help="whether the message names the positive concept or an unrelated one",
    )
    parser.add_argument(
        "--message-form", choices=("dense", "tokens"), default="dense",
        help="dense gives each concept a random real-valued code, which is the "
             "easiest signal there is; tokens gives it a one-hot symbol "
             "sequence through the receiver's own embedding, which is the "
             "shape and the entropy a real message actually has",
    )
    parser.add_argument(
        "--distractors", choices=("clustered", "varied"), default="clustered",
        help="clustered leaves one bit for the message; varied leaves the "
             "positives as the only repeated concept, which is the shortcut",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=None,
                        help="defaults to the config's own optimiser.lr")
    parser.add_argument("--concepts", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.5,
                        help="Gaussian noise added to each prototype")
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_receiver(config_path):
    """The real modules, built the way `train.py` builds them."""
    config = parse_config.get_config(config_path)
    config["cuda"] = False

    name = "cub" if "cub" in config["data"]["dataset"] else "shapeworld"

    class _Dataset:
        n_feats = FEATS[name]

    _Dataset.name = name

    class _Loader:
        dataset = _Dataset()

    pair = models.builder.build_models({"train": _Loader()}, config)["pair"]
    return config, pair.receiver.train()


def widths(comparer, config):
    """What shapes this comparer wants, asked of the module rather than assumed."""
    if isinstance(comparer, models.receiver.TransformerCrossAttentionComparer):
        return (
            comparer.referent_adapter.in_features,
            comparer.token_embedding_size,
            comparer.message_length,
        )
    return (
        comparer.referent_embedding_size,
        comparer.token_embedding_size,
        config["receiver_comparer"]["message_length"],
    )


def main():
    args = parse_args()
    config, receiver = build_receiver(args.config)
    comparer = receiver.comparer

    d_ref, d_msg, msg_len = widths(comparer, config)
    batch = config["data"]["batch_size"]
    n_obj = config["data"]["n_examples"]
    lr = args.lr if args.lr is not None else config["optimiser"]["lr"]
    has_kurtosis = hasattr(comparer, "decision_kurtosis")

    # +4 for PAD, SOS, EOS and UNK, matching `models.builder`.
    vocabulary = config["sender_language_model"]["vocabulary"] + 4
    vocabulary_note = (
        f" over {vocabulary} symbols" if args.message_form == "tokens" else ""
    )

    print(f"{os.path.basename(args.config)}")
    print(f"  comparer    : {type(comparer).__name__}")
    print(f"  referents   : (batch {batch}, n_obj {n_obj}, d_ref {d_ref})")
    print(f"  message     : (batch {batch}, len {msg_len}, d_msg {d_msg})"
          f"  [{args.message}, {args.message_form}{vocabulary_note}]")
    print(f"  distractors : {args.distractors}")
    print(f"  concepts {args.concepts}, noise {args.noise}, lr {lr}, "
          f"{args.steps} steps\n")

    torch.manual_seed(args.seed)

    # One prototype and one code per concept, both fixed for the whole run.
    # This is the easiest protocol there is: already converged, noise-free, and
    # the same every time the concept comes up.
    prototypes = torch.randn(args.concepts, d_ref)

    if args.message_form == "tokens":
        # What the speaker actually emits: `message_length` one-hot rows over
        #     the vocabulary, which `Receiver.forward` turns into embeddings
        #     with `messages @ token_embedding.weight`. Sampled once per
        #     concept and then fixed, so this is still a converged protocol --
        #     the only thing taken away from the dense case is the width and
        #     the entropy of the signal carrying it.
        #
        # Distinctness is asserted rather than assumed. With 24 symbols over 10
        #     positions a collision is vanishingly unlikely, but a collision
        #     would make the task unsolvable rather than hard, and this script
        #     exists to distinguish those two.
        tokens = torch.randint(vocabulary, (args.concepts, msg_len))
        assert len({tuple(row.tolist()) for row in tokens}) == args.concepts, (
            "two concepts drew the same message"
        )
        codes = torch.nn.functional.one_hot(tokens, vocabulary).float()
    else:
        codes = torch.randn(args.concepts, msg_len, d_msg)

    def make_batch():
        positive = torch.randint(args.concepts, (batch,))
        offset = torch.randint(1, args.concepts, (batch, n_obj))
        negative = (positive[:, None] + offset) % args.concepts

        which = negative.clone()
        if args.distractors == "clustered":
            # Every distractor is the same concept as every other, so the two
            # halves are equally good clusters and neither is distinguishable
            # as "the odd one out".
            which = negative[:, :1].expand(batch, n_obj).clone()
        which[:, : n_obj // 2] = positive[:, None]

        y = torch.zeros(batch, n_obj)
        y[:, : n_obj // 2] = 1.0

        # Shuffle so slot position says nothing about the label.
        perm = torch.argsort(torch.rand(batch, n_obj), dim=1)
        which = torch.gather(which, 1, perm)
        y = torch.gather(y, 1, perm)

        referents = prototypes[which] + args.noise * torch.randn(batch, n_obj, d_ref)
        said = (
            torch.randint(args.concepts, (batch,))
            if args.message == "scrambled"
            else positive
        )
        return referents, codes[said], y

    # The token embedding learns in the real run and is the first thing the
    #     message meets, so it learns here too. Excluded in the dense case,
    #     where nothing routes through it.
    learners = [comparer]
    if args.message_form == "tokens":
        learners.append(receiver.token_embedding)

    parameters = [p for module in learners for p in module.parameters()]
    optimiser = torch.optim.AdamW(parameters, lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_acc = 0.0
    for step in range(args.steps + 1):
        referents, message, y = make_batch()
        if args.message_form == "tokens":
            message = message @ receiver.token_embedding.weight
        scores = comparer(referents, message)
        loss = criterion(scores, y)

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimiser.step()

        with torch.no_grad():
            acc = ((scores > 0).float() == y).float().mean().item()
        best_acc = max(best_acc, acc)

        if step % args.every:
            continue

        with torch.no_grad():
            s = scores.detach().float()
            standardised = (s - s.mean()) / s.std()
            kurtosis = (standardised ** 4).mean().item() - 3.0

        line = (
            f"  step {step:5d}  loss {loss.item():.4f}  acc {acc:.4f}  "
            f"score_sd {s.std().item():.3f}  excess_kurt {kurtosis:+7.2f}"
        )
        if has_kurtosis:
            line += (
                f"  [module: spread {comparer.decision_spread:.3f} "
                f"kurt {comparer.decision_kurtosis:+.2f}]"
            )
        print(line)

    print(f"\n  best accuracy {best_acc:.4f} over {args.steps} steps")


if __name__ == "__main__":
    main()

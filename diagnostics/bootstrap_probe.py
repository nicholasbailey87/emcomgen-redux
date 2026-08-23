"""
Run the whole pair, with the vision models taken out of the argument.

Rung numbers in this docstring are the *old* ladder, which was renumbered when it
grew to sixteen rungs -- see `experiments/README.md`. Old 10 is new 12 and old 12
is new 16, both with the contrast stage added.

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


class LearnableConceptVision(nn.Module):
    """
    A vision model that has *not* won yet, on the same game.

    `FrozenConceptVision` hands each agent a feature space in which the concepts
        are already separated. That is what makes the probe cheap, and it is
        also the one thing it cannot ask about: both comparers solve the game
        under it -- the two-stack reaches 1.000 by step 600 and the four-stage
        one by step 800 -- while only the bilinear one learns in the real run.
        The difference the probe removes is the difference that is left.

    So here the concepts live in a fixed `input_dim` space with a nuisance
        subspace on top, and each agent owns a learnable linear map from that
        space to its own features. Two properties make it the right hard case:

    1. At initialisation the map is a random projection, which by
       Johnson-Lindenstrauss preserves the input's class separation and adds
       nothing. The features therefore open at the *input's* signal-to-noise,
       not at a separation someone arranged.
    2. The nuisance lives in a `nuisance_dim` subspace, so a linear map can
       null it. The task is learnable, and learning it is exactly the job the
       ViT does: amplify between-class over within-class variation.

    Both agents see the same concepts through their own map and their own draw
        of the nuisance, as the real pair sees different images of one species.
    """

    def __init__(self, concepts, final_feat_dim, input_dim, nuisance_dim,
                 nuisance, generator):
        super().__init__()
        self.final_feat_dim = final_feat_dim
        self.nuisance = nuisance

        # Shared across both agents, so the two are looking at one world.
        self.register_buffer(
            "concept_prototypes",
            torch.randn(concepts, input_dim, generator=generator),
        )
        basis = torch.randn(nuisance_dim, input_dim, generator=generator)
        self.register_buffer("nuisance_basis", basis / basis.norm(dim=1, keepdim=True))

        # The agent's own encoder, and the only thing here that learns.
        self.encoder = nn.Linear(input_dim, final_feat_dim, bias=False)

    def forward(self, ids):
        clean = self.concept_prototypes[ids.reshape(-1).long()]
        coefficients = torch.randn(
            clean.size(0), self.nuisance_basis.size(0), device=clean.device
        )
        return self.encoder(clean + self.nuisance * (coefficients @ self.nuisance_basis))

    def reset_parameters(self):
        self.encoder.reset_parameters()


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=os.path.join(CONFIG_DIR, "16_birds_receiver_cross_attention_lm.toml"),
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
    parser.add_argument(
        "--cross-beta", type=float, default=1.0,
        help="multiply the LISTENER's referent-stack cross-attention branch by "
             "this, which is the same as giving that branch its own beta. 2.45 "
             "undamps it to beta 1.0 at three blocks, leaving the other two "
             "branches and the message stack at DeepNorm. Cross-attention "
             "comparer only",
    )
    parser.add_argument(
        "--vision", choices=("frozen", "learnable"), default="frozen",
        help="`frozen` gives both agents a feature space in which the concepts "
             "are already separated -- a vision model that has won. "
             "`learnable` gives each agent a linear encoder over a shared "
             "input space with a nuisance subspace on top, so the separation "
             "has to be learned while the pair bootstraps",
    )
    parser.add_argument(
        "--input-dim", type=int, default=128,
        help="learnable vision only: width of the shared input space",
    )
    parser.add_argument(
        "--nuisance-dim", type=int, default=32,
        help="learnable vision only: dimension of the within-class subspace, "
             "which is what a working encoder learns to null",
    )
    parser.add_argument(
        "--nuisance", type=float, default=1.0,
        help="learnable vision only: scale of the within-class variation "
             "against the unit-scale concept prototypes. 0 is a separable "
             "space that still has to be found; large is an encoder with a "
             "long way to go",
    )
    parser.add_argument(
        "--contrast", action="store_true",
        help="turn on the speaker's contrast stage, whatever the config says: "
             "one self-attention over all the referents, added back as a "
             "residual before they are pooled. Watch `gate` leave zero and "
             "`within` against `share` -- a large share with a small within is "
             "a branch emitting one vector per polarity, which the prototyper "
             "could already do",
    )
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build(config_path, concepts, noise, lr=None, vision="frozen",
          input_dim=128, nuisance_dim=32, nuisance=1.0, contrast=False):
    """The real pair and the real optimiser, with the vision models swapped."""
    config = parse_config.get_config(config_path)
    config["cuda"] = False

    # An override rather than a second config: the point is the same rung with
    #     and without the stage, and no rung sets it yet.
    if contrast:
        config["sender"]["contrast"] = True

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
    if vision == "frozen":
        def make(width):
            return FrozenConceptVision(concepts, width, noise)
    else:
        # One generator for both agents, so the concepts and the nuisance
        #     subspace are a property of the world rather than of the agent.
        #     `torch.manual_seed` in `main` still fixes the whole run.
        world = torch.Generator().manual_seed(torch.initial_seed() % (2 ** 31))

        def make(width):
            return LearnableConceptVision(
                concepts, width, input_dim, nuisance_dim, nuisance, world
            )

    pair.sender.feat_model = make(pair.sender.feat_model.final_feat_dim)
    pair.receiver.feature_model = make(pair.receiver.feature_model.final_feat_dim)

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

    # The encoders are new parameters in no group, so without this they would
    #     sit at their random projection for the whole run and `learnable`
    #     would be a harder `frozen` rather than a different question. Base
    #     rate: they stand in for the vision models, which take the base rate.
    encoders = [p for n, p in pair.named_parameters() if ".encoder." in n]
    if encoders:
        optimiser.add_param_group(
            {"params": encoders,
             "lr": config["optimiser"]["lr"] * scale,
             "weight_decay": 0.0}
        )

    return config, pair, optimiser



def scale_message_crossing(pair, factor):
    """
    Give the referent stack's cross-attention branch its own beta.

    `DecoderBlock._residual` multiplies all three branches by one `self.beta`,
        so DeepNorm damps the crossing that carries the message by exactly as
        much as the candidates' self-attention and the feedforward -- and
        `alpha = beta = 1.0` in the config undamps all three, in both stacks,
        including the message stack's crossing, which reads the candidates and
        runs the other way. That is four changes to test one idea.

    Scaling this branch's output by `factor` is identical to running it at
        `factor * beta` and leaves everything else at DeepNorm's values. The
        message stack is untouched: the question is how much message reaches
        the scored stream, not how much the message reads.
    """
    if factor == 1.0:
        return 0
    blocks = pair.receiver.discriminator.referent_decoder.blocks
    for block in blocks:
        block.cross_attention.register_forward_hook(
            lambda module, args, output, k=factor: output * k
        )
    return len(blocks)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    config, pair, optimiser = build(
        args.config, args.concepts, args.noise, args.lr,
        args.vision, args.input_dim, args.nuisance_dim, args.nuisance,
        args.contrast,
    )
    pair.train()

    batch = config["data"]["batch_size"]
    n_obj = config["data"]["n_examples"]
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"{os.path.basename(args.config)}")
    print(f"  speaker  : {type(pair.sender.language_model).__name__} / "
          f"{type(pair.sender.prototyper).__name__}")
    print(f"  listener : {type(pair.receiver.language_model).__name__} / "
          f"{type(pair.receiver.discriminator).__name__}")
    rates = sorted({group["lr"] for group in optimiser.param_groups})
    print(f"  batch {batch} x {n_obj} slots, {args.concepts} concepts, "
          f"noise {args.noise}, {args.steps} steps")
    print(f"  rates    : {', '.join(f'{rate:.2g}' for rate in rates)}")
    if args.vision == "frozen":
        print(f"  vision   : frozen prototypes, both agents -- already separated\n")
    else:
        print(f"  vision   : learnable linear encoder, both agents; "
              f"{args.input_dim}-d input, {args.nuisance_dim}-d nuisance "
              f"subspace at scale {args.nuisance}\n")

    # The listener's own columns, where it has them. `decision_kurtosis` reads
    #     the shape of its scores: negative means bimodal, which is what
    #     discriminating looks like; sustained positive alongside a flat `acc`
    #     means a listener with nothing to say. `mix_alpha` is how much of the
    #     score is the attention path, and it is the column this probe exists
    #     to watch: the attention arm on its own reaches 0.469 under nuisance 8
    #     where the bilinear one reaches 0.938, so what has to be seen is
    #     whether attention gets taken up once the mix carries the bootstrap.
    discriminator = pair.receiver.discriminator
    has_kurtosis = hasattr(discriminator, "decision_kurtosis")
    has_mix = hasattr(discriminator, "mix_alpha")

    # The speaker's contrast stage, when it is on. `gate` is the scalar between
    #     the branch and the identity, `share` how much of the referent going
    #     into the prototyper is contrast, and `within` how much of the branch
    #     is example-level rather than one vector for a game or a polarity. The
    #     first says whether it opened, the second how loud it is and the third
    #     whether it is doing anything the prototyper could not already do.
    contrast = pair.sender.contrast
    if contrast is not None:
        print(f"  contrast : on, {contrast.heads} heads at "
              f"{contrast.d_model} wide over "
              f"{n_obj} referents; gate opens at 0\n")

    if args.cross_beta != 1.0:
        scaled = scale_message_crossing(pair, args.cross_beta)
        print(f"  crossing : referent-stack cross-attention x{args.cross_beta} "
              f"over {scaled} blocks -- effective beta "
              f"{discriminator.beta * args.cross_beta:.3f} on that branch, "
              f"{discriminator.beta:.3f} on the other two\n")

    header = (
        f"{'step':>6} {'loss':>7} {'acc':>7} {'pool_eff':>9} {'pool_norm':>10} "
        f"{'polarity':>9} {'lgt_scale':>10} {'survival':>9} {'spread':>7}"
        + (f" {'kurtosis':>9}" if has_kurtosis else "")
        + (f" {'mix_a':>7} {'agree':>7}" if has_mix else "")
        + (f" {'gate':>7} {'share':>7} {'within':>7}"
           if contrast is not None else "")
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
            line += f" {discriminator.decision_kurtosis:+9.2f}"
        if has_mix:
            line += (
                f" {discriminator.mix_alpha:7.3f}"
                f" {discriminator.path_agreement:+7.3f}"
            )
        if contrast is not None:
            line += (
                f" {contrast.contrast_gate.item():+7.3f}"
                f" {contrast.contrast_share:7.3f}"
                f" {contrast.contrast_within_share:7.3f}"
            )
        print(line)


if __name__ == "__main__":
    main()

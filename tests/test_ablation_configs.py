"""
The ablation rungs, built the way `train.py` builds them.

Nothing checked this before, and the cost of that was the speaker rung -- 7 in
the old numbering, 9 in this one. Its speaker came out at 533,943 parameters
against the GRU baseline's 6,813,499 -- twelve times smaller -- because `SenderTransformerLM` pins its width to the vision model's
and ShapeWorld's ViT had been sized against Conv4's 113,088. The config was
valid, every unit test was green, and the run produced numbers. They just were
not measuring architecture.

So these tests assert two things a unit test cannot: that each rung constructs
and forwards at all, and that the arms it compares are the sizes they claim to
be. The parameter counts below are the point of the rebalancing rather than
incidental facts about it -- if one moves, either the change was intended and
the number should be updated deliberately, or an arm has quietly stopped being
comparable to the one beside it.

Counts are exact rather than banded. A band wide enough to be robust to a real
architectural change is too wide to catch the thing this file exists to catch.
"""

import glob
import os

import pytest
import torch

import _bootstrap  # noqa: F401

import broccoli.transformer
import models.builder
import parse_config

from _bootstrap import CONFIG_DIRS, EXPERIMENTS_DIR, all_rungs, rung

SHAPEWORLD_FEATS = (3, 64, 64)
BIRDS_FEATS = (3, 224, 224)

# Even rungs are CUB, odd rungs ShapeWorld.
RUNGS = all_rungs()


def _feats(config_file):
    return BIRDS_FEATS if int(config_file[:2]) % 2 == 0 else SHAPEWORLD_FEATS


def _name(config_file):
    return "cub" if int(config_file[:2]) % 2 == 0 else "shapeworld"


def _build(config_file, contrast=False):
    """A real rung, through `models.builder`, with a stub dataloader."""
    config = parse_config.get_config(rung(config_file))
    config["cuda"] = False
    # Forced either way rather than inherited, so a test that builds both arms
    #     of a rung gets both arms whatever the rung itself says. Rungs 7 and
    #     above set this to true.
    config["sender"]["contrast"] = contrast

    class _Dataset:
        n_feats = _feats(config_file)
        name = _name(config_file)

    class _Loader:
        dataset = _Dataset()

    return config, models.builder.build_models({"train": _Loader()}, config)


def _pair(config_file, contrast=False):
    config, built = _build(config_file, contrast=contrast)
    return config, built["pair"]


def _count(module):
    return sum(p.numel() for p in module.parameters())


def test_every_rung_found_can_be_opened():
    """
    The plumbing backstop, and deliberately not a count.

    `all_rungs` looks in `experiments/ablation_shapeworld/configs/` and
        `experiments/ablation_birds/configs/`, and in the two experiment
        folders above them, because the first of each pair is the live SLURM
        queue rather than the ladder's home: `scripts/run_experiment.sh` builds
        its job array from `configs/*.toml`, so running a subset means moving
        the rest up a level. The two experiments are the ladder's ShapeWorld and
        birds arms, split so they can be queued independently; the rungs kept
        their numbers, so this scan still returns the whole ladder. `973a68b` did exactly that and left `CONFIG_DIR`
        pointing at the queue, which raised `FileNotFoundError` inside
        `get_config` for the fourteen rungs that had moved and hid 159 tests for
        a day.

    That failure is quiet -- it reads as a wall of unrelated failures rather
        than as a missing path -- so this is the test that names it. What it
        asserts is that every name the scan returned resolves to a file that
        parses, which is a fact about the harness.

    **How many rungs there are, and which, is not tested anywhere.** That is a
        property of the experiment, and the experiment is allowed to change: a
        rung parked, a variant added, the queue cut to the three configs a
        question needs. A test that pinned the shape would have to be edited
        every time the experiment moved, which makes it a chore rather than a
        guard -- and a guard nobody trusts gets edited to pass.
    """
    assert RUNGS, f"no configs found in {', '.join(CONFIG_DIRS)}"

    for config_file in RUNGS:
        assert parse_config.get_config(rung(config_file))


# Experiment folders whose configs are known not to parse, with the key that
#     retired them. They are finished experiments and their configs are the
#     record of runs that happened, not instructions for runs that will; a
#     config naming a retired key cannot be re-run as written, and rewriting one
#     to parse would be inventing a setting for a run that never used it.
#
#     `silhouette_titration_norms` sets `[receiver_discriminator]
#     normalise_score`, which `parse_config.validate_config` retired when the
#     flag was split into `scale_score` and `bias_score`. There is deliberately
#     no translation: the old `false` also removed the `1/sqrt(d)` calibration,
#     and no current setting does that, so the arm is unreachable rather than
#     renamed. See DEFAULT.toml beside those two keys.
STALE_EXPERIMENTS = ("silhouette_titration_norms",)


def test_every_experiment_config_parses():
    """
    Every queued config in the repository, not just the ladder's.

    `all_rungs` scans `ablation_shapeworld` and `ablation_birds` alone, which is
        right for the tests above -- they *build* each rung, and building the
        whole of `experiments/` would be minutes of model construction to
        re-assert a property the ladder already covers. But it leaves the other
        experiment folders with no check at all, and there are eighty-odd config
        files in `experiments/lr_sweep_*/` whose only failure mode is a config
        that does not parse.

    Parse only, therefore, and no `build_models`: this catches a retired key, a
        malformed table, a rate that is not a positive number, a `[data]
        dataset` naming neither dataset -- everything `validate_config` knows
        about -- in well under a second. It is the check that the seventy sweep
        arms were generated correctly, and it would have caught the
        `estimator` removal against the nine `lr_sweep_1_cnn` arms that still
        carried the key.

    Failures are collected rather than raised one at a time, because a
        generated set of configs tends to be wrong all together or not at all,
        and one name out of eighty is not a useful error message.

    **`STALE_EXPERIMENTS` is the one exclusion, and it is a statement rather
        than a convenience.** Those folders hold finished experiments whose
        configs name keys that have since been retired, so they no longer parse
        and cannot be re-run as written. That is a real fact about them and
        this test would otherwise be red for a reason nobody intends to fix;
        naming them here says which ones and why, where scoping the glob to the
        live folders would have hidden it. Remove a name from that tuple by
        making its configs parse, not by narrowing the scan.
    """
    pattern = os.path.join(EXPERIMENTS_DIR, "*", "configs", "*.toml")
    configs = sorted(
        path for path in glob.glob(pattern)
        if os.path.basename(os.path.dirname(os.path.dirname(path)))
        not in STALE_EXPERIMENTS
    )

    assert configs, f"no configs found under {pattern}"

    failures = []
    for path in configs:
        try:
            parse_config.get_config(path)
        except Exception as error:
            failures.append(
                f"{os.path.relpath(path, EXPERIMENTS_DIR)}: "
                f"{type(error).__name__}: {error}"
            )

    assert not failures, "\n".join([""] + failures)


@pytest.mark.parametrize("config_file", RUNGS)
def test_every_rung_constructs(config_file):
    _pair(config_file)


@pytest.mark.parametrize("config_file", RUNGS)
def test_every_rung_speaks_a_message_of_the_configured_length(config_file):
    """
    End to end through the speaker, which is where a width mismatch surfaces:
    `SenderTransformerLM` raises on construction if the referent and token widths
    disagree, but a latent array that leaked downstream would only show up here.
    """
    config, pair = _pair(config_file)
    pair.eval()

    batch, n_obj = 2, config["data"]["n_examples"]
    samples = torch.randn(batch, n_obj, *_feats(config_file))
    targets = torch.zeros(batch, n_obj)
    targets[:, : n_obj // 2] = 1.0

    with torch.no_grad():
        messages, _ = pair.sender(samples, targets)

    assert messages.shape == (
        batch,
        config["sender_language_model"]["message_length"],
        config["sender_language_model"]["vocabulary"] + 4,
    )


# The sizes the ladder is built around: sender vision, speaker language model,
# and both listener slots, for each dataset's baseline and for the top of the
# ladder, which carries every change. See the rung headers for where each number
# comes from.
#
# **Every speaker language model here lost one parameter on 2026-08-30 and got it
# back on 2026-08-31**, as `log_logit_scale` stopped and then resumed being
# learned. The counts below are the second figure, so they match what the rung
# headers and docs quote again. Both arms moved by the same one parameter either
# way, so the *ratio* between them never changed. See docs/channel.md.
#
# **Every number here was re-measured when the ladder was rebuilt.** The counts
# this file used to carry for rungs 11 to 14 had gone stale -- they predated the
# removal of the absolute position tables and the `ff_inner_size` 554 -> 576
# alignment -- and had been failing rather than catching anything. Do not read
# the old figures out of git history as a reference; they describe a tree that
# no longer exists.
@pytest.mark.parametrize(
    "config_file,module,expected",
    [
        # The speaker's counts are taken with its adapter in the path,
        # which sizes that agent from its language model's `d_model` rather than
        # from its backbone. The numbers that moved when it arrived, moved for
        # that reason and not because a module was redesigned:
        #
        #   * the speakers' `init_h` reads `2 * referent_width`, so it doubles
        #     at the GRU rungs as the referents go 512 -> 1024. That restores
        #     jayelm's own width: their speaker sits on `Conv4`, whose
        #     `final_feat_dim` is 1024, so `Linear(2048 -> 1024)` is the paper's
        #     projection and 512 was `ResNet18SmallInput`'s number.
        #
        # **The listener's counts all moved again when its adapters were hoisted
        # into `Receiver`.** Its single adapter is gone; each slot declares
        # the widths it wants and `Receiver` delivers each input through a
        # `model_util.LinearInterface` sized straight from `final_feat_dim`,
        # which is the same class the speaker's adapter is. Three
        # consequences show up in the rows below:
        #
        #   * `receiver.language_model` on the transformer arm lost its
        #     `referent_adapter` -- 4,784,566 -> 4,702,646 -- which is what makes
        #     the parity number against the GRU +0.3% rather than +2.1%. See
        #     test_receiver_slots.py.
        #   * `receiver.discriminator` on the attention arm lost its two
        #     adapters and two norms, and its composed bilinear path moved from
        #     `[receiver_language_model] d_model` down to the slot's own
        #     `d_model`. 3,891,782 -> 2,384,198.
        #   * that same module is now *the same size on rungs 13 and 15*, where
        #     it used to differ by a `memory_adapter` reading a 1024-wide GRU
        #     state against a 256-wide one. The memory still has to be brought to
        #     `d_model`; it is an interface one stage upstream, so the 13 -> 15
        #     step is clean on this module and the difference has moved into
        #     `receiver.interfaces`.
        #
        # The interfaces are pinned per rung below rather than left implicit,
        # because they are where the listener's remaining width arithmetic
        # lives and they are the group `[optimiser.module_lr] receiver_adapter`
        # still names. The speaker's adapter is not pinned: it is an
        # architectural constant whose size is a function of two widths that are
        # already pinned, so a claim about it would restate them.
        # ShapeWorld: the CNN/GRU baseline.
        #
        # **Every ShapeWorld vision count below moved on 2026-09-06**, when the
        # backbone became `ResNet56` -- He et al.'s CIFAR network at 852,368
        # parameters -- and the ShapeWorld ViT2 was narrowed to 876,599 to match
        # it. It was `ResNet18SmallInput` at 11,168,832 against a 320-wide ViT at
        # 10,317,986, and `Conv4` at 113,088 against the same ViT before that:
        # 91x. Neither arrangement was a comparison of architectures at a fixed
        # size, which is what the ladder claims to be making. The birds counts
        # are untouched -- `[birds.sender_feature_model]` pins CUB's ViT at the
        # 320/10/5/576 SwiGLU stack it always had -- so the two datasets are no
        # longer matched to each other, deliberately. See
        # `test_the_shapeworld_backbones_are_matched`.
        ("01_shapeworld_baseline.toml", "sender.feat_model", 852_368),
        ("01_shapeworld_baseline.toml", "receiver.feature_model", 852_368),
        ("01_shapeworld_baseline.toml", "sender.language_model", 6_813_499),
        # The listener is two modules: `receiver.language_model` encodes the
        # message and `receiver.discriminator` scores the candidates from it.
        #
        # **These two are a capacity-matching argument, and that is new.** The
        # baseline's GRU encoder is 4,687,872 -- jayelm's 1 layer unidirectional
        # at 1024 -- against 4,702,646 for `ReceiverCrossAttentionLM` at rung
        # 15's 6 blocks, which is +0.3%. Both numbers are pinned here so that a
        # config change to either arm breaks this test rather than quietly
        # reopening the gap.
        #
        # It used to be 28,262,400 against 2,466,139: `DEFAULT.toml` carried 2
        # layers bidirectional for parity at a *shared* width of 256, and nothing
        # in the ladder set both widths, so every rung up to 14 got that key at
        # 1024 wide with a 500-wide token embedding. Parity is now sought at
        # jayelm's width by deepening the transformer arm instead.
        ("01_shapeworld_baseline.toml", "receiver.language_model", 4_687_872),
        # 1024 * 1024 for the `bilinear` weight, plus `log_score_scale` and
        # `score_bias`. The comment here used to say "exactly 512 * 1024" and
        # name 524,289, which had not matched the pinned value since the
        # discriminator started comparing at the language model's width; the
        # arithmetic is 1024 * 1024 + 2. Both operand norms have moved to
        # `Receiver` and neither held a parameter, so this count is unchanged by
        # the hoist. See test_score_scale.py.
        ("01_shapeworld_baseline.toml", "receiver.discriminator", 1_048_578),
        # The interfaces. `discriminator_referents` is `final_feat_dim` -> 1024
        # with no bias and `discriminator_message` 1024 -> 1024 with one; the GRU
        # declares no referent width, so there is no third. This is the whole of
        # what replaced the listener's single adapter.
        #
        # ShapeWorld reads 1,115,136 -- 64 * 1024 for the referents plus
        # 1024 * 1024 + 1024 for the message -- where CUB reads 1,573,888 at
        # `ResNet18`'s 512. The two used to agree at 512; they differ now because
        # `ResNet56` emits 64, which is what a CIFAR network's last stage is
        # wide. That narrowing is also why this number is worth pinning per
        # dataset rather than once.
        ("01_shapeworld_baseline.toml", "receiver.interfaces", 1_115_136),
        # **Every `sender.language_model` count here moved by one parameter
        # twice on 2026-09-05, and is back where it started.**
        # `[sender_language_model] normalise_logits` was defaulted to `false`
        # that morning as one of four settings that moved together, and back to
        # `true` that evening -- see DEFAULT.toml beside the key for why the
        # confound was the reason to return it. The parameter is
        # `log_logit_scale`, a lone scalar that is *absent* from the module when
        # the key is off rather than frozen, which is the arrangement that lets
        # `split_out_parameter` and `SCALAR_GROUPS` see the truth. Four numbers,
        # all -1 and then +1, on both datasets and both speaker arms.
        #
        # No rung sets the key, so the capacity match between arms was never
        # disturbed -- they moved together both times. A rung that pins the key
        # off will read one lower here and that is the setting talking, not a
        # size drift.
        #
        # ShapeWorld: the top of the ladder. The speaker's language model is the
        # causal arm at seven blocks -- see rung 9's `layers` for why seven, and
        # for the two depths before it.
        ("15_shapeworld_receiver_cross_attention_lm.toml", "sender.feat_model", 876_599),
        ("15_shapeworld_receiver_cross_attention_lm.toml", "receiver.feature_model", 876_599),
        ("15_shapeworld_receiver_cross_attention_lm.toml", "sender.language_model", 6_758_354),
        ("15_shapeworld_receiver_cross_attention_lm.toml", "receiver.language_model", 4_702_646),
        ("15_shapeworld_receiver_cross_attention_lm.toml", "receiver.discriminator", 2_384_198),
        # Three interfaces here, and all of them narrow: `final_feat_dim` -> 256
        # twice for the two slots' referents and 256 -> 256 for the message.
        # Against rung 13 the difference is the message interface, which reads a
        # 256-wide encoded message rather than a 1024-wide GRU state.
        #
        # ShapeWorld reads 131,328 -- 128 * 256 twice plus 256 * 256 + 256 --
        # against CUB's 229,632 at the 320-wide birds ViT. The referent side is
        # where the two datasets' ViT widths show up.
        ("15_shapeworld_receiver_cross_attention_lm.toml", "receiver.interfaces", 131_328),
        # CUB: the CNN/GRU baseline.
        ("02_birds_baseline.toml", "sender.feat_model", 11_176_512),
        ("02_birds_baseline.toml", "sender.language_model", 6_822_649),
        ("02_birds_baseline.toml", "receiver.language_model", 4_687_872),
        ("02_birds_baseline.toml", "receiver.discriminator", 1_048_578),
        # CUB: the top of the ladder. Only the two vision-dependent counts differ
        # from ShapeWorld's -- the ViT's patch tokeniser scales with image size,
        # and the speaker's language model carries a longer message.
        ("16_birds_receiver_cross_attention_lm.toml", "sender.feat_model", 11_332_626),
        ("16_birds_receiver_cross_attention_lm.toml", "sender.language_model", 6_764_120),
        ("16_birds_receiver_cross_attention_lm.toml", "receiver.language_model", 4_702_646),
        ("16_birds_receiver_cross_attention_lm.toml", "receiver.discriminator", 2_384_198),
        # Rung 13's discriminator, pinned because it used to be the number that
        # made the 13 -> 15 step unclean and now is not: it is *equal* to rung
        # 15's. The gap was a `memory_adapter` bringing the GRU's 1024-wide
        # output down to 256 where rung 15 read a 256-wide message directly, and
        # that adapter is `Receiver`'s message interface now -- so the two rungs
        # differ in `receiver.interfaces`, pinned separately below, and not in
        # the module under test. It was 3,891,782 before the hoist, and
        # 3,580,487 before the listener GRU's output went 2048 -> 1024.
        #
        # These two are unchanged across `7b10d47`, and the arithmetic is worth
        # stating because it is a coincidence: the module gained one parameter
        # in `log_score_scale` and lost one in `decision.bias`, which the
        # readout's per-game centring annihilated. Its composed bilinear path
        # has neither of `ScoreVolume`'s scalars, being built with both
        # composition gates off.
        #
        # Unchanged again by the commit that added `score_bias`, and again by
        # coincidence: this module already had an offset in `mix_bias`, which
        # `score_bias` replaces one for one. What changed is where it lives and
        # that it now has a config key and a metrics column.
        #
        # The bilinear discriminator's 524,288 became 524,289 across `7b10d47`
        # -- it gained the volume with no bias to lose -- and 524,290 with
        # `score_bias`, which is the parameter it never had. Before that it had
        # no bias anywhere, `bilinear` being built `bias=False`, so nothing in
        # rungs 1-12 could place the score against `train.py`'s fixed
        # `lis_scores > 0`. Two scalars is the whole cost of the listener's
        # readout.
        ("13_shapeworld_attention_discriminator.toml", "receiver.discriminator", 2_384_198),
        ("14_birds_attention_discriminator.toml", "receiver.discriminator", 2_384_198),
        # Where the 13 -> 15 difference went: `final_feat_dim` -> 256 for the
        # referents and 1024 -> 256 for the GRU's state. The language model
        # declares no referent width on this rung, so there are two interfaces
        # here and three there, and the message interface is the expensive one.
        #
        # ShapeWorld reads 295,168 -- 128 * 256 plus 1024 * 256 + 256 -- against
        # rung 15's 131,328. CUB reads 344,320 against rung 16's 229,632, which
        # is the same arithmetic at 320.
        ("13_shapeworld_attention_discriminator.toml", "receiver.interfaces", 295_168),
        ("14_birds_attention_discriminator.toml", "receiver.interfaces", 344_320),
        # The two intermediate vision swaps, so a rung that stopped inheriting
        # the shared ViT specification shows up here rather than in a run.
        ("03_shapeworld_sender_vit.toml", "sender.feat_model", 876_599),
        ("04_birds_sender_vit.toml", "sender.feat_model", 11_332_626),
        # And the prototyper, which is one scoring direction and a bias per
        # polarity, where rung 3's is nothing at all. 2,050 rather than the 642
        # it was: it sizes off the referents, which the adapter now delivers at
        # the speaker's GRU width of 1024 rather than at the ViT's 320.
        ("05_shapeworld_attention_prototyper.toml", "sender.prototyper", 2_050),
        ("06_birds_attention_prototyper.toml", "sender.prototyper", 2_050),
    ],
)
def test_the_arms_are_the_sizes_they_claim(config_file, module, expected):
    _, pair = _pair(config_file)

    submodule = pair
    for part in module.split("."):
        submodule = getattr(submodule, part)

    assert _count(submodule) == expected


def test_the_shapeworld_backbones_are_matched():
    """
    The claim `ResNet56` exists to make, asserted directly rather than left to
    be inferred from the two rows in the table above.

    Both numbers are stated in DEFAULT.toml as the reason for the sizes chosen
    there, so both are pinned here: 852,368 for the CNN, which is `6n + 2` at
    n = 9 and is within 1% of the 0.85M the architecture is known by, and
    876,599 for the ViT, which is what 128 / 6 / 4 / 256 with GELU comes to. The
    ratio is the point -- a ViT rung against a CNN rung measures architecture
    only if the two are the same size -- and 3% is the band the feedforward
    width can be tuned to at this depth.

    The `Conv4` and `ResNet18` numbers are here as the contrast: this pair used
    to be 0.11M against 10.3M.
    """
    _, baseline = _pair("01_shapeworld_baseline.toml")
    _, transformer = _pair("03_shapeworld_sender_vit.toml")

    cnn = _count(baseline.sender.feat_model)
    vit = _count(transformer.sender.feat_model)

    assert cnn == 852_368
    assert abs(cnn - 850_000) / 850_000 < 0.01, f"{cnn:,} is not 0.85M"
    assert vit == 876_599

    assert abs(vit / cnn - 1.0) < 0.03, f"{vit / cnn:.3f}x"

    # Both agents, and both on the same backbone as their partner. A rung that
    # moved one and not the other would be measuring asymmetry.
    assert _count(baseline.receiver.feature_model) == cnn
    _, receiver_vit = _pair("11_shapeworld_receiver_vit.toml")
    assert _count(receiver_vit.receiver.feature_model) == vit


@pytest.mark.parametrize(
    "config_file",
    ["01_shapeworld_baseline.toml", "02_birds_baseline.toml"],
)
def test_nothing_that_should_be_undecayed_is_decayed(config_file):
    """
    `[optimiser] weight_decay` is 0.1 rather than 0.0 since 2026-09-06, so which
    parameters `get_optimiser` hands a non-zero coefficient stopped being a
    question with only one answer.

    Two things have to hold and neither is obvious from the config. First, no
    bias, normalisation gain, embedding table or lone scalar may be decayed --
    `gradboard` excludes them by name and, independently, gives every parameter
    with fewer than two axes a coefficient of 0.0, and `builder.build_models`
    adds `"bn"` to the keyword list because this repository's BatchNorm gains
    are named `BN1` and `BN2` and match none of the stock keywords.

    Second, a module moved to a rate of its own must *keep* its decay.
    `_regroup` used to add every new group at `weight_decay = 0.0`, which was
    inert while the base was 0.0 and would have switched the decay off under
    exactly the two modules the learning-rate sweeps move.

    The rate is moved through `[optimiser.implementation_lr]` rather than
    `[optimiser.module_lr]`, because that is the table a sweep result lands in
    and it is the one that wins: a `module_lr.sender_vision` set here would be
    silently outranked by DEFAULT.toml's rate for whichever backbone the rung
    runs, and this would then be asserting the default rather than its own
    override. Keyed off `builder.implementation_of` so it follows the rung
    rather than naming `ResNet56` and `ResNet18` here as well.
    """
    config = parse_config.get_config(rung(config_file))
    config["cuda"] = False
    config["optimiser"]["weight_decay"] = 0.1
    # As a sweep does: both backbones off the base rate, so both are split into
    # groups of their own on the way through `split_out_module`.
    for group in ("sender_vision", "receiver_vision"):
        implementation = models.builder.implementation_of(config, group)
        config["optimiser"]["implementation_lr"][group] = {implementation: 1e-5}

    class _Dataset:
        n_feats = _feats(config_file)
        name = _name(config_file)

    class _Loader:
        dataset = _Dataset()

    built = models.builder.build_models({"train": _Loader()}, config)
    pair, optimiser = built["pair"], built["optimiser"]

    names = {id(p): n for n, p in pair.named_parameters()}
    decayed = {
        names[id(p)]
        for group in optimiser.param_groups
        if group["weight_decay"] != 0.0
        for p in group["params"]
    }

    exposed = [
        n for n in decayed
        if any(k in n.lower() for k in ("bias", "norm", "embedding", "beta", "bn"))
    ]
    assert not exposed, f"decayed by name: {sorted(exposed)}"

    shapes = {id(p): p.dim() for _, p in pair.named_parameters()}
    flat = [n for n in decayed if shapes[id(dict(pair.named_parameters())[n])] < 2]
    assert not flat, f"decayed 0-d or 1-d: {sorted(flat)}"

    for module, group_name in (
        (pair.sender.feat_model, "sender_vision"),
        (pair.receiver.feature_model, "receiver_vision"),
    ):
        moved = {id(p) for p in module.parameters()}
        still_decayed = sum(
            1
            for group in optimiser.param_groups
            if group["weight_decay"] != 0.0
            for p in group["params"]
            if id(p) in moved
        )
        assert still_decayed, (
            f"{group_name} was moved to its own rate and lost its weight decay"
        )
        assert all(
            group["lr"] == 1e-5
            for group in optimiser.param_groups
            if any(id(p) in moved for p in group["params"])
        ), f"{group_name} did not all move to the configured rate"


@pytest.mark.parametrize(
    "baseline,transformer,tolerance",
    [
        # Measured at 0.992x on ShapeWorld and 0.991x on CUB, at seven blocks
        # and `ff_inner_size = 512`. The tolerance is 0.05 rather than something
        # tighter because `layers` is an integer: the neighbouring depths are
        # 0.86x and 1.13x, so nothing between them is reachable on depth alone
        # and a tighter band would only be pinning the arithmetic of one depth
        # against one feedforward width.
        #
        # It was 1.029x at four blocks and 576, then 1.015x at six once the
        # speaker's blocks lost their cross-attention sublayer, and six became
        # seven when the speaker's adapter widened the *baseline* -- `init_h` reads
        # `2 * referent_width`, which the adapter took from the backbone's 512
        # to the GRU's own 1024. This rung's stack runs at its own `d_model` and
        # did not move with it. See rung 9's `layers`.
        ("01_shapeworld_baseline.toml", "09_shapeworld_sender_transformer_lm.toml", 0.05),
        ("02_birds_baseline.toml", "10_birds_sender_transformer_lm.toml", 0.05),
        # The same speaker at the top of the ladder, which nothing above rung 9
        # is supposed to touch. If these two diverge from the pair above, a
        # listener rung has reached into the speaker.
        ("01_shapeworld_baseline.toml", "15_shapeworld_receiver_cross_attention_lm.toml", 0.05),
        ("02_birds_baseline.toml", "16_birds_receiver_cross_attention_lm.toml", 0.05),
    ],
)
def test_the_speakers_language_models_are_matched(baseline, transformer, tolerance):
    """
    The claim the whole rebalancing exists to support, stated as a ratio rather
    than as two absolute numbers so it survives a deliberate resize of both.

    Only the language models. The vision models are within about 11% on either
    dataset rather than matched, because the ViT's patch tokeniser scales with
    image size while a ResNet's stem does not -- see rung 6.
    """
    _, base = _pair(baseline)
    _, arm = _pair(transformer)

    ratio = _count(arm.sender.language_model) / _count(base.sender.language_model)

    assert abs(ratio - 1.0) < tolerance, f"{ratio:.3f}x"


# Both agents: the rotary modules are the speaker's decoder self-attention at
# rung 9 and, on top of that, the listener's two stacks at rung 15, so neither
# rung covers the other.
@pytest.mark.parametrize(
    "config_file",
    [
        "09_shapeworld_sender_transformer_lm.toml",
        "15_shapeworld_receiver_cross_attention_lm.toml",
    ],
)
def test_every_rope_attention_takes_all_its_heads(config_file):
    """
    `positional_heads` is pinned at 1.0 and is no longer a config key.

    Below 1.0 broccoli splits the head axis -- `math.ceil(fraction * n_heads)`
    heads take axial RoPE and the rest are carried through a second value
    projection and concatenated back -- so the size of the partition moved
    whenever `heads` moved. In a study that varies width that is a hidden
    confound, and 0.5 was the default.

    Scoped to the modules where the setting can act. The bare `MHAttention`
    cross-attentions -- the speaker's prototype read, its latent read, and the
    listener's message read -- carry `rotary_embedding=None` and sit at
    broccoli's own 0.25 default, which is inert and deliberately left there.
    """
    _, pair = _pair(config_file)

    checked = 0
    for module in pair.modules():
        if not isinstance(module, broccoli.transformer.MHAttention):
            continue
        if module.rotary_embedding is None:
            continue
        assert module.positional_heads == module.n_heads, (
            f"rotates {module.positional_heads} of {module.n_heads} heads"
        )
        checked += 1

    assert checked, "no rotary attention in this pair; the test proved nothing"


# --------------------------------------------------------------------------
# The speaker's contrast stage, forced on and off independently of what a rung
#     says. Rungs 7 and above set `[sender] contrast` themselves; these build
#     both arms of each rung below, one per dataset for each of the two sender
#     backbones the ladder uses.
# --------------------------------------------------------------------------

CONTRAST_RUNGS = (
    "01_shapeworld_baseline.toml",
    "02_birds_baseline.toml",
    "11_shapeworld_receiver_vit.toml",
    "12_birds_receiver_vit.toml",
)


@pytest.mark.parametrize("config_file", CONTRAST_RUNGS)
def test_a_rung_with_contrast_still_speaks(config_file):
    """
    The stage returns the backbone's own width, so everything downstream should
    be unable to tell it ran. This is the same end-to-end pass as
    `test_every_rung_speaks_a_message_of_the_configured_length`, with the flag
    on.
    """
    config, pair = _pair(config_file, contrast=True)
    pair.eval()

    batch, n_obj = 2, config["data"]["n_examples"]
    samples = torch.randn(batch, n_obj, *_feats(config_file))
    targets = torch.zeros(batch, n_obj)
    targets[:, : n_obj // 2] = 1.0

    with torch.no_grad():
        messages, _ = pair.sender(samples, targets)

    assert messages.shape == (
        batch,
        config["sender_language_model"]["message_length"],
        config["sender_language_model"]["vocabulary"] + 4,
    )


@pytest.mark.parametrize("config_file", CONTRAST_RUNGS)
def test_contrast_opens_at_the_parent_rung(config_file):
    """
    Bit-identical messages with the flag on and off, from the same seed. This is
    what makes the contrast arm an ablation of one thing, and it holds only
    because `contrast_gate` opens at zero *and* because the stage is built after
    the speaker's other modules, so it does not shift their draws from the RNG.

    Greedy at eval, so there is no channel noise to average over.
    """
    batch = 2
    messages = {}

    for contrast in (False, True):
        torch.manual_seed(0)
        config, pair = _pair(config_file, contrast=contrast)
        pair.eval()

        n_obj = config["data"]["n_examples"]
        generator = torch.Generator().manual_seed(1)
        samples = torch.randn(
            batch, n_obj, *_feats(config_file), generator=generator
        )
        targets = torch.zeros(batch, n_obj)
        targets[:, : n_obj // 2] = 1.0

        with torch.no_grad():
            messages[contrast], _ = pair.sender(samples, targets)

    assert torch.equal(messages[False], messages[True])


@pytest.mark.parametrize(
    "config_file,expected",
    [
        # What the stage sees is no longer the backbone's width but the
        #     adapter's output, which is the speaker's own language model
        #     `d_model`: 1024 at the GRU rungs, 320 wherever
        #     `SenderTransformerLM` sets it there. The rest is the stage's own
        #     width, so the numbers are
        #     `2 * feat * 320 + 4 * 320^2 + 320 + 2 * 320 + feat + 1` -- the two
        #     projections, the attention's four, its `out_norm` gain, the label
        #     tag and the gate.
        #
        #     Both datasets read the same number now where they always did, but
        #     for a different reason: it used to be that `ResNet18` and `Conv4`
        #     both happened to hand over 512, and it is now that both speakers
        #     run their GRU at the same `d_model`. A backbone swap no longer
        #     moves this count at all, which is the adapter working.
        ("01_shapeworld_baseline.toml", 1_066_945),
        ("02_birds_baseline.toml", 1_066_945),
        ("11_shapeworld_receiver_vit.toml", 615_681),
        ("12_birds_receiver_vit.toml", 615_681),
    ],
)
def test_contrast_costs_what_it_says(config_file, expected):
    """
    Exact, for the reason every other count in this file is exact: the stage is
    one attention and two projections, and a second block or a feedforward
    creeping in would otherwise show up only as a slower run.
    """
    _, plain = _pair(config_file)
    _, contrasted = _pair(config_file, contrast=True)

    assert plain.sender.contrast is None
    assert _count(contrasted.sender.contrast) == expected
    assert (
        _count(contrasted.sender) - _count(plain.sender) == expected
    ), "the stage changed something outside itself"


@pytest.mark.parametrize("config_file", CONTRAST_RUNGS)
def test_the_gate_gets_its_own_learning_rate(config_file):
    """
    The gate is a lone scalar opening at zero, and at the base rate it cannot
    travel further than `lr * steps` -- sixteen epochs of sign-consistent
    gradient to reach 0.1 on birds. `contrast_gate_lr` is what makes the arm
    answerable inside a run, so a group that quietly stopped being created would
    look like "the contrast stage does nothing".
    """
    config, built = _build(config_file, contrast=True)

    gate = built["pair"].sender.contrast.contrast_gate
    expected_lr = config["optimiser"]["contrast_gate_lr"]

    group = [
        g for g in built["optimiser"].param_groups
        if any(p is gate for p in g["params"])
    ]

    assert len(group) == 1
    assert group[0]["lr"] == expected_lr
    assert group[0]["lr"] != config["optimiser"]["lr"]

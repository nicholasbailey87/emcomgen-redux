"""
The sixteen ablation rungs, built the way `train.py` builds them.

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

import os

import pytest
import torch

import _bootstrap  # noqa: F401

import broccoli.transformer
import models.builder
import parse_config

from _bootstrap import all_rungs, rung

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


def test_there_are_sixteen_rungs():
    """
    A rung added or renamed without the counts below being revisited -- and the
        backstop for the other way this breaks.

    `all_rungs` looks in `experiments/ablation/configs/` and in
        `experiments/ablation/` above it, because the first is the live SLURM
        queue rather than the ladder's home: `scripts/run_experiment.sh` builds
        its job array from `configs/*.toml`, so running a subset means moving
        the rest up a level. `973a68b` did exactly that and left `CONFIG_DIR`
        pointing at the queue, which raised `FileNotFoundError` inside
        `get_config` for the fourteen rungs that had moved and hid 159 tests for
        a day.

    That failure is quiet -- it reads as a wall of unrelated failures rather
        than as a missing path -- so this is the test that names it. If a rung
        is in neither directory, the count is wrong here first.
    """
    assert len(RUNGS) == 16, f"found {len(RUNGS)}: {RUNGS}"


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
# **Every speaker language model here lost one parameter on 2026-08-30**, when
# `log_logit_scale` stopped being learned: the channel scale is a constant solved
# from `token_max_probability` at construction and is not in `state_dict` at all.
# So each `sender.language_model` count below is one lower than the figure the
# rung headers and docs quote, and the rung headers are the ones to trust for the
# *ratio* between the arms, which is unchanged because both arms lost the same
# one parameter. See docs/channel.md.
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
        # ShapeWorld: the CNN/GRU baseline.
        ("01_shapeworld_baseline.toml", "sender.feat_model", 11_168_832),
        ("01_shapeworld_baseline.toml", "sender.language_model", 5_764_922),
        # The listener is two modules: `receiver.language_model` encodes the
        # message and `receiver.discriminator` scores the candidates from it.
        #
        # **These two are a capacity-matching argument, and that is new.** The
        # baseline's GRU encoder is 4,687,872 -- jayelm's 1 layer unidirectional
        # at 1024 -- against 4,784,566 for `ReceiverCrossAttentionLM` at rung
        # 15's 6 blocks, which is +2.1%. Both numbers are pinned here so that a
        # config change to either arm breaks this test rather than quietly
        # reopening the gap.
        #
        # It used to be 28,262,400 against 2,466,139: `DEFAULT.toml` carried 2
        # layers bidirectional for parity at a *shared* width of 256, and nothing
        # in the ladder set both widths, so every rung up to 14 got that key at
        # 1024 wide with a 500-wide token embedding. Parity is now sought at
        # jayelm's width by deepening the transformer arm instead.
        ("01_shapeworld_baseline.toml", "receiver.language_model", 4_687_872),
        # Halved with the GRU's `output_size`, 2048 -> 1024. Exactly 512 * 1024,
        # the `bilinear` weight and nothing else: it was 524,289 while
        # `log_score_scale` carried the volume, and that scalar is gone -- the
        # weight carries it now. See test_score_scale.py.
        ("01_shapeworld_baseline.toml", "receiver.discriminator", 524_290),
        # ShapeWorld: the top of the ladder. The speaker's language model is the
        # causal arm at six blocks -- see rung 9's `layers` for why six, and why
        # it was four until the blocks stopped cross-attending.
        ("15_shapeworld_receiver_cross_attention_lm.toml", "sender.feat_model", 10_317_986),
        ("15_shapeworld_receiver_cross_attention_lm.toml", "sender.language_model", 5_854_088),
        ("15_shapeworld_receiver_cross_attention_lm.toml", "receiver.language_model", 4_784_566),
        ("15_shapeworld_receiver_cross_attention_lm.toml", "receiver.discriminator", 2_548_294),
        # CUB: the CNN/GRU baseline.
        ("02_birds_baseline.toml", "sender.feat_model", 11_176_512),
        ("02_birds_baseline.toml", "sender.language_model", 5_774_072),
        ("02_birds_baseline.toml", "receiver.language_model", 4_687_872),
        ("02_birds_baseline.toml", "receiver.discriminator", 524_290),
        # CUB: the top of the ladder. Only the two vision-dependent counts differ
        # from ShapeWorld's -- the ViT's patch tokeniser scales with image size,
        # and the speaker's language model carries a longer message.
        ("16_birds_receiver_cross_attention_lm.toml", "sender.feat_model", 11_332_626),
        ("16_birds_receiver_cross_attention_lm.toml", "sender.language_model", 5_859_854),
        ("16_birds_receiver_cross_attention_lm.toml", "receiver.language_model", 4_784_566),
        ("16_birds_receiver_cross_attention_lm.toml", "receiver.discriminator", 2_548_294),
        # Rung 13's discriminator, still pinned because it is still the number
        # that makes the 13 -> 15 step unclean, though far less so than it was:
        # 2,990,662 against rung 15's 2,548,294. The gap is a `memory_adapter`
        # bringing the GRU's output down to 256 rather than reading a 256-wide
        # message directly, and it shrank from 3,580,487 when that output went
        # 2048 -> 1024 with the listener GRU's restoration.
        #
        # These two are unchanged across `7b10d47`, and the arithmetic is worth
        # stating because it is a coincidence: the module gained one parameter
        # in `log_score_scale` and lost one in `decision.bias`, which the
        # readout's per-game centring annihilated. Its composed bilinear path
        # has neither of `ScoreVolume`'s scalars, being built with
        # `score_scale=False`.
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
        ("13_shapeworld_attention_discriminator.toml", "receiver.discriminator", 2_990_662),
        ("14_birds_attention_discriminator.toml", "receiver.discriminator", 2_990_662),
        # The two intermediate vision swaps, so a rung that stopped inheriting
        # the shared ViT specification shows up here rather than in a run.
        ("03_shapeworld_sender_vit.toml", "sender.feat_model", 10_317_986),
        ("04_birds_sender_vit.toml", "sender.feat_model", 11_332_626),
        # And the prototyper, which is 642 parameters -- one scoring direction
        # and a bias per polarity, at the ViT's 320 -- where rung 3's is nothing
        # at all.
        ("05_shapeworld_attention_prototyper.toml", "sender.prototyper", 642),
        ("06_birds_attention_prototyper.toml", "sender.prototyper", 642),
    ],
)
def test_the_arms_are_the_sizes_they_claim(config_file, module, expected):
    _, pair = _pair(config_file)

    submodule = pair
    for part in module.split("."):
        submodule = getattr(submodule, part)

    assert _count(submodule) == expected


@pytest.mark.parametrize(
    "baseline,transformer,tolerance",
    [
        # Measured at 1.015x on both datasets, at six blocks and
        # `ff_inner_size = 512`. The tolerance is 0.05 rather than something
        # tighter because `layers` is an integer: at 512 the neighbouring depths
        # are 0.88x and 1.08x, so nothing between them is reachable on depth
        # alone and a tighter band would only be pinning the arithmetic of one
        # depth against one feedforward width.
        #
        # It was 1.029x at four blocks and 576, before the speaker's blocks lost
        # their cross-attention sublayer.
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
        # ResNet18 and Conv4 both hand over 512; the ViT2 rungs hand over their
        #     own `d_model`, 320. The rest is the stage's own width, so the two
        #     numbers are `2 * feat * 320 + 4 * 320^2 + 320 + 2 * 320 + feat + 1`
        #     -- the two projections, the attention's four, its `out_norm` gain,
        #     the label tag and the gate.
        ("01_shapeworld_baseline.toml", 738_753),
        ("02_birds_baseline.toml", 738_753),
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

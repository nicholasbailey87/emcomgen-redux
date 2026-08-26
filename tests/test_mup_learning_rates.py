"""
The muP learning-rate rule, as `models/builder.py` applies it to the ladder.

The rule is

    lr(module) = [optimiser] lr * [optimiser] mup_reference_width / d_model

with the reference width 1024 -- jayelm's `--speaker_hidden_size` and
`--listener_hidden_size`, and therefore the width `lr = 1e-4` was tuned at.

What these tests exist to catch is not the arithmetic, which is one line, but
the *partition*. Six overrides already move named scalars into groups of their
own, and muP now moves whole modules; if a parameter fell into two groups it
would be stepped twice per optimiser step, and if it fell into none it would
never be stepped at all. Neither shows up as an error, and both would look like
an architecture result. `is_mup_exempt` is what keeps the two schemes disjoint,
by dimension and by name rather than by ordering, and it is asserted here
directly as well as through its consequences.

The other failure mode is `PASS`. It deep-copies the parameter groups once at
construction and zips them `strict=True` thereafter, so a group added after it
was built raises -- which is why `build_models` does all of its regrouping
before `train.py` reaches the scheduler. Every rung is constructed through a
real `PASS` here for that reason.
"""

import os

import pytest
import torch

import _bootstrap  # noqa: F401

import gradboard.cycles
from gradboard.scheduler import PASS

import models.builder as builder
import parse_config

from _bootstrap import CONFIG_DIR

SHAPEWORLD_FEATS = (3, 64, 64)
BIRDS_FEATS = (3, 224, 224)

RUNGS = sorted(f for f in os.listdir(CONFIG_DIR) if f.endswith(".toml"))

# The rungs each width first appears on, so a failure names the change that
#     introduced it rather than the whole ladder.
WIDTH_320 = "09_shapeworld_sender_transformer_lm.toml"
WIDTH_256 = "15_shapeworld_receiver_cross_attention_lm.toml"

# `(suffix, config key)` for every group `SPLIT_LEARNING_RATES` creates. muP
#     must leave all of these alone.
SCALAR_OVERRIDES = (
    ("log_logit_scale", "logit_scale_lr"),
    ("log_score_scale", "score_scale_lr"),
    ("polarity_embedding", "polarity_embedding_lr"),
    ("mix_logit", "mix_logit_lr"),
    ("contrast_gate", "contrast_gate_lr"),
)


def _build(config_file):
    """A real rung through `models.builder`, with a stub dataloader."""
    config = parse_config.get_config(os.path.join(CONFIG_DIR, config_file))
    config["cuda"] = False

    class _Dataset:
        n_feats = (
            BIRDS_FEATS if int(config_file[:2]) % 2 == 0 else SHAPEWORLD_FEATS
        )

    class _Loader:
        dataset = _Dataset()

    return config, builder.build_models({"train": _Loader()}, config)


def _lr_by_id(optimiser):
    return {
        id(p): group["lr"]
        for group in optimiser.param_groups
        for p in group["params"]
    }


def _trainable(pair):
    return [(n, p) for n, p in pair.named_parameters() if p.requires_grad]


# --------------------------------------------------------------------------
# The rule itself.
# --------------------------------------------------------------------------

def test_the_reference_width_is_jayelms():
    """
    1024 is `--speaker_hidden_size` / `--listener_hidden_size`, and the whole
        rule is a departure measured from it. A different number here would
        rescale every module at once and silently.
    """
    config = parse_config.get_config()

    assert config["optimiser"]["mup_reference_width"] == 1024
    assert config["optimiser"]["lr"] == 1e-4


def test_a_missing_reference_width_is_rejected():
    """
    `SafeDict` warns on a missing key and hands back None, so without this check
        every factor would be `None / d_model` and the failure would arrive as a
        TypeError from inside `build_models`.
    """
    config = parse_config.get_config()
    del config["optimiser"]["mup_reference_width"]

    with pytest.raises(parse_config.InvalidConfig):
        parse_config.validate_config(config)

    for bad in (0, -1024, "1024"):
        config["optimiser"]["mup_reference_width"] = bad
        with pytest.raises(parse_config.InvalidConfig):
            parse_config.validate_config(config)


@pytest.mark.parametrize(
    "config_file,module_name,width,lr",
    [
        # The GRU pair, at the reference width, unchanged at base.
        ("01_shapeworld_baseline.toml", "sender_language_model", 1024, 1e-4),
        ("01_shapeworld_baseline.toml", "receiver_language_model", 1024, 1e-4),
        # `ViT2`, and everything the speaker sizes from `final_feat_dim`.
        ("03_shapeworld_sender_vit.toml", "sender_vision", 320, 3.2e-4),
        ("05_shapeworld_attention_prototyper.toml", "sender_prototyper", 320, 3.2e-4),
        ("07_shapeworld_sender_contrast.toml", "sender_contrast", 320, 3.2e-4),
        # Rung 9 is the drop this whole change is aimed at: 1024 -> 320.
        (WIDTH_320, "sender_language_model", 320, 3.2e-4),
        ("11_shapeworld_receiver_vit.toml", "receiver_vision", 320, 3.2e-4),
        # The listener's own transformers, at 256.
        ("13_shapeworld_attention_discriminator.toml", "receiver_discriminator", 256, 4.0e-4),
        (WIDTH_256, "receiver_language_model", 256, 4.0e-4),
    ],
)
def test_each_width_gets_the_rate_the_rule_predicts(
    config_file, module_name, width, lr
):
    config, built = _build(config_file)
    select = dict((n, s) for n, s, _ in builder.MUP_MODULES)[module_name]
    module = select(built["pair"])

    assert builder.mup_width(module) == width

    lr_of = _lr_by_id(built["optimiser"])
    rates = {
        lr_of[id(p)] for n, p in module.named_parameters()
        if p.requires_grad and not builder.is_mup_exempt(n, p)
    }

    assert len(rates) == 1
    assert rates.pop() == pytest.approx(lr)
    assert config["optimiser"]["resolved_mup_lrs"][module_name] == pytest.approx(lr)


@pytest.mark.parametrize(
    "config_file,module_name",
    [
        # `ResNet18.final_feat_dim` is a hardcoded 512 and muP's rules are
        #     stated for transformers. This is what jayelm tuned 1e-4 against.
        ("01_shapeworld_baseline.toml", "sender_vision"),
        ("01_shapeworld_baseline.toml", "receiver_vision"),
        # No matrices between widths at all.
        ("01_shapeworld_baseline.toml", "sender_prototyper"),
        # `BilinearDiscriminator` reads nothing from its own table: its one
        #     tensor's fan-in is the language model's `output_size`, which the
        #     restored listener GRU puts at 1024 -- the reference width -- so
        #     the factor would be 1.0 anyway.
        ("01_shapeworld_baseline.toml", "receiver_discriminator"),
    ],
)
def test_the_modules_without_a_width_stay_at_base(config_file, module_name):
    config, built = _build(config_file)
    select = dict((n, s) for n, s, _ in builder.MUP_MODULES)[module_name]
    module = select(built["pair"])

    assert builder.mup_width(module) is None

    base_lr = config["optimiser"]["lr"]
    lr_of = _lr_by_id(built["optimiser"])

    # `SPLIT_LEARNING_RATES` still reaches inside these -- every discriminator
    #     carries `log_score_scale` at 2e-3, and an `AttentionDiscriminator`
    #     carries `mix_logit` too -- and muP being out of scope is not a claim
    #     that nothing else moves. What is asserted is that muP itself created
    #     no group here.
    overridden = tuple(suffix for suffix, _ in SCALAR_OVERRIDES)
    unclaimed = [
        (n, p) for n, p in module.named_parameters()
        if p.requires_grad and not n.endswith(overridden)
    ]

    # `AveragePrototyper` has no parameters at all, which is why it is out of
    #     scope and is a legitimately empty list here.
    assert all(lr_of[id(p)] == base_lr for _, p in unclaimed)
    assert config["optimiser"]["resolved_mup_lrs"][module_name] == base_lr


def test_the_bilinear_discriminators_fan_in_really_is_the_reference_width():
    """
    The argument for exempting it is not that it has no width but that its width
        is already 1024. That holds only while the listener GRU is jayelm's; a
        config that made it bidirectional again would put the fan-in at 2048 and
        this exemption would start costing a factor of two.
    """
    config, built = _build("01_shapeworld_baseline.toml")
    listener = built["pair"].receiver

    assert listener.language_model.output_size == 1024
    assert listener.discriminator.message_width == 1024
    assert config["optimiser"]["mup_reference_width"] == 1024


# --------------------------------------------------------------------------
# The exemptions, asserted directly.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["anything", "attn.q_proj.weight"])
def test_anything_below_two_dimensions_is_exempt(name):
    for shape in [(), (1,), (320,)]:
        assert builder.is_mup_exempt(name, torch.zeros(shape))

    assert not builder.is_mup_exempt(name, torch.zeros(320, 320))


@pytest.mark.parametrize(
    "name",
    [
        "polarity_embedding",
        "label_embedding",
        "token_embedding.weight",
        "query",
        "transformer.blocks.0.attn.q_proj.weight",  # "q_proj" is not "query"
    ],
)
def test_the_name_rule_matches_the_tensors_it_is_meant_to(name):
    exempt = builder.is_mup_exempt(name, torch.zeros(2, 320))

    assert exempt is ("embedding" in name or "query" in name)


# --------------------------------------------------------------------------
# The partition, over every rung.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config_file", RUNGS)
def test_every_trainable_parameter_is_in_exactly_one_group(config_file):
    """
    Two groups would step a parameter twice; none would never step it. Neither
        raises, and both would read as an architecture result.

    Frozen parameters are excluded on both sides: `get_optimiser` skips them,
        and `ViT2` has ten -- its blocks' rotary `freqs`.
    """
    _, built = _build(config_file)

    grouped = [
        id(p) for g in built["optimiser"].param_groups for p in g["params"]
    ]
    trainable = {id(p) for _, p in _trainable(built["pair"])}

    assert len(grouped) == len(set(grouped))
    assert set(grouped) == trainable


@pytest.mark.parametrize("config_file", RUNGS)
def test_the_scalar_overrides_survive_the_muP_groups(config_file):
    """
    The disjointness that `is_mup_exempt` buys, read off the built pair: every
        scalar in `SPLIT_LEARNING_RATES` is 0-d and `polarity_embedding` matches
        "embedding", so muP cannot claim any of them whichever loop runs first.
    """
    config, built = _build(config_file)
    lr_of = _lr_by_id(built["optimiser"])
    seen = 0

    for suffix, key in SCALAR_OVERRIDES:
        hits = [p for n, p in _trainable(built["pair"]) if n.endswith(suffix)]

        if not hits:
            continue

        seen += 1
        assert {lr_of[id(p)] for p in hits} == {config["optimiser"][key]}

    # Every rung has both volumes -- `log_logit_scale` on the speaker and
    #     `log_score_scale` on the listener -- so a rung matching fewer than two
    #     suffixes would mean the table had gone stale rather than that the rung
    #     was austere. It was briefly `>= 1`, while the listener's volume lived
    #     in `bilinear.weight` and `decision.weight` at the base rate; that is
    #     the round `7b10d47` reversed.
    assert seen >= 2


@pytest.mark.parametrize("config_file", RUNGS)
def test_embeddings_and_queries_keep_the_base_rate(config_file):
    """
    muP gives input embeddings a Theta(1) rate, and every one of these is
        Theta(1)-initialised as well, so a width factor would be scaling against
        an init that never shrank. `polarity_embedding` is the exception and is
        elevated for its own reason -- see `polarity_embedding_lr`.
    """
    config, built = _build(config_file)
    base_lr = config["optimiser"]["lr"]
    lr_of = _lr_by_id(built["optimiser"])

    matched = [
        (n, p) for n, p in _trainable(built["pair"])
        if any(f in n.lower() for f in builder.MUP_EMBEDDING_LIKE)
        and not n.endswith("polarity_embedding")
    ]

    assert matched, "the name rule matched nothing -- has a tensor been renamed?"
    assert all(lr_of[id(p)] == base_lr for _, p in matched)


@pytest.mark.parametrize("config_file", RUNGS)
def test_pass_builds_over_the_groups_and_still_reports_base(config_file):
    """
    `PASS` deep-copies the groups at construction and zips them `strict=True`,
        so this is the test that would fail if `build_models` ever regrouped
        after the scheduler existed. `PASS.lr` reads group 0, which stays the
        one `get_optimiser` made.
    """
    config, built = _build(config_file)

    scheduler = PASS(
        gradboard.cycles.CycleSequence(
            [gradboard.cycles.Cycle(gradboard.cycles.ascent, 1_000, 1, 32)]
        ),
        built["pair"],
        built["optimiser"],
        scaler=None,
        range_test=False,
        cool_point_multiplier=config["scheduler"]["cool_point_multiplier"],
    )

    assert scheduler.lr == config["optimiser"]["lr"]
    assert len(scheduler.original_param_groups) == len(
        built["optimiser"].param_groups
    )


# --------------------------------------------------------------------------
# What gets recorded.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config_file", RUNGS)
def test_the_resolved_rates_are_written_back_for_save_args(config_file):
    """
    `train.py` calls `save_args` *after* `build_models` precisely so this
        mapping reaches `args.json`. Several of these widths are derived rather
        than declared, so a config key is not evidence that the module was built
        that way -- see docs/training.md.
    """
    config, built = _build(config_file)
    resolved = config["optimiser"]["resolved_mup_lrs"]

    built_modules = {
        name for name, select, _ in builder.MUP_MODULES
        if select(built["pair"]) is not None
    }

    assert set(resolved) == built_modules
    assert all(isinstance(v, float) and v > 0 for v in resolved.values())

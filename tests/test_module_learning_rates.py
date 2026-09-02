"""
The per-module clip groups and the learning rates that ride on them, as
`models/builder.py` applies them to the ladder.

One table now decides both. `MODULE_GROUPS` names the modules that are clipped
together and rateable in `[optimiser.module_lr]`; `SCALAR_GROUPS` names the
scaling scalars that are clipped alone and rated through the `*_lr` keys. Before
this there were three lists -- `train.py`'s `CLIP_GROUPS`, `builder`'s
`MUP_MODULES`, and the scalar overrides -- none derivable from another, and the
first omitted `sender.contrast` so that its `other` catch-all silently *was* the
contrast stage on every rung that had one.

What these tests exist to catch is not the arithmetic, which is now a dictionary
lookup, but the *partition*. If a parameter fell into two groups it would be
clipped twice and, where the rates differ, stepped twice; if it fell into none it
would never be stepped at all. Neither raises, and both would look like an
architecture result. `claimed_separately` is what keeps the module groups
disjoint from the scalar ones, by a single statement used on both sides rather
than by ordering, and it is asserted here directly as well as through its
consequences.

The other failure mode is `PASS`. It deep-copies the parameter groups once at
construction and zips them `strict=True` thereafter, so a group added after it
was built raises -- which is why `build_models` does all of its regrouping before
`train.py` reaches the scheduler. Every rung is constructed through a real `PASS`
here for that reason.

The rates themselves are read out of each rung's own config rather than written
here as literals. A rung that changes its `[optimiser.module_lr]` or its base
`lr` is making a decision, not breaking a rule, and a test that had to be edited
alongside it would be asserting the diff rather than the invariant. What is
asserted is that every parameter of a group sits at exactly the rate the config
names for it -- one rate per group, no strays.
"""

import math

import pytest
import torch

import _bootstrap  # noqa: F401

from gradboard.scheduler import PASS

import models.builder as builder
import parse_config
import train

from _bootstrap import all_rungs, rung

SHAPEWORLD_FEATS = (3, 64, 64)
BIRDS_FEATS = (3, 224, 224)

RUNGS = all_rungs()

# `(suffix, config key)` for every group `SPLIT_LEARNING_RATES` creates. Six
#     keys against four scalar clip groups, and the mismatch is deliberate:
#     `score_bias` and `polarity_embedding` take a rate of their own but clip
#     with the module whose output they modify. See `SCALAR_GROUPS`.
#
#     The speaker's channel scale heads this list again. It was briefly a
#     constant -- between 2026-08-30 and 2026-08-31 -- and is a parameter once
#     more, with both a rate and a clip group of its own.
SCALAR_OVERRIDES = (
    ("log_logit_scale", "logit_scale_lr"),
    ("log_score_scale", "score_scale_lr"),
    ("score_bias", "score_bias_lr"),
    ("polarity_embedding", "polarity_embedding_lr"),
    ("mix_logit", "mix_logit_lr"),
    ("contrast_gate", "contrast_gate_lr"),
)


def _build(config_file):
    """A real rung through `models.builder`, with a stub dataloader."""
    config = parse_config.get_config(rung(config_file))
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


def _selector(module_name):
    return dict(builder.MODULE_GROUPS)[module_name]


# --------------------------------------------------------------------------
# The table itself.
# --------------------------------------------------------------------------

def test_the_group_names_are_the_module_lr_keys():
    """
    `[optimiser.module_lr]` in DEFAULT.toml states every module group
        explicitly, so the surface is discoverable from the config rather than
        only from the code. A group missing here would silently run at base.
    """
    config = parse_config.get_config()

    assert set(config["optimiser"]["module_lr"]) == {
        name for name, _ in builder.MODULE_GROUPS
    }


def test_the_default_rates_are_one_factor_of_two():
    """
    DEFAULT's table is not flat and is not eight independent numbers: every rate
        is the base `lr` times one factor, carrying one claim. The whole listener
        runs at half the whole speaker, and nothing is split within an agent. See
        DEFAULT.toml for the argument and for what it costs the baselines.

    Asserted as the grid rather than as eight literals, because the magnitudes
        are chosen and the structure is the claim. A change that keeps the grid
        is a retune; one that breaks it is a different position, and this is
        where a reader is told which happened. It applies to all sixteen rungs
        at once either way.
    """
    config = parse_config.get_config()
    base_lr = config["optimiser"]["lr"]
    rates = config["optimiser"]["module_lr"]

    assert base_lr == 5e-5

    # The grid is pinned at the listener's language model, the module jayelm
    #     tuned `lr` on. One factor, and no split within an agent: the `vision`
    #     half of the grid was withdrawn on 2026-08-29, so a backbone taking
    #     anything other than its own agent's rate is now a broken grid rather
    #     than a retune.
    speaker = 2.0

    expected = {
        "sender_vision": base_lr * speaker,
        "sender_adapter": base_lr * speaker,
        "sender_prototyper": base_lr * speaker,
        "sender_contrast": base_lr * speaker,
        "sender_language_model": base_lr * speaker,
        "receiver_vision": base_lr,
        "receiver_adapter": base_lr,
        "receiver_token_embedding": base_lr,
        "receiver_language_model": base_lr,
        "receiver_discriminator": base_lr,
    }

    assert rates == pytest.approx(expected)


def test_a_module_lr_key_naming_no_group_is_rejected():
    """
    The failure this exists to prevent is the quiet one: a typo sitting in the
        config looking like a setting while the module it was meant for runs at
        base rate. This is the same guard `split_out_parameter` gives the
        scalars, moved to parse time because a module group cannot fail that
        way -- it selects an attribute, not a name.
    """
    config = parse_config.get_config()
    config["optimiser"]["module_lr"]["sender_langauge_model"] = 1e-4

    with pytest.raises(parse_config.InvalidConfig):
        parse_config.validate_config(config)


@pytest.mark.parametrize("bad", [0, -1e-4, "1e-4", True])
def test_a_module_lr_that_is_not_a_positive_number_is_rejected(bad):
    config = parse_config.get_config()
    config["optimiser"]["module_lr"]["sender_vision"] = bad

    with pytest.raises(parse_config.InvalidConfig):
        parse_config.validate_config(config)


# --------------------------------------------------------------------------
# The rates, read off the built pair.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config_file", RUNGS)
def test_each_module_group_gets_the_rate_its_config_names(config_file):
    """
    One rate per group, and it is the one `[optimiser.module_lr]` names -- read
        out of the rung's own config rather than written here, so that a rung
        retuning itself is a decision rather than a broken test.

    The scaling scalars are excluded on both sides. `claimed_separately` keeps
        them out of the module's optimiser group as well as out of its clip
        group, so a module at 3.2e-4 does not drag its gate along with it and
        `contrast_gate_lr = lr` still means "no override".
    """
    config, built = _build(config_file)
    pair = built["pair"]
    lr_of = _lr_by_id(built["optimiser"])
    base_lr = config["optimiser"]["lr"]

    # `score_bias` and `polarity_embedding` clip with their module but take
    #     their rate from their own key, so they are strays here by design.
    rated_separately = tuple(suffix for suffix, _ in SCALAR_OVERRIDES)

    for name, select in builder.MODULE_GROUPS:
        module = select(pair)

        if module is None:
            continue

        expected = config["optimiser"]["module_lr"].get(name, base_lr)

        rates = {
            lr_of[id(p)] for n, p in module.named_parameters()
            if p.requires_grad and not n.endswith(rated_separately)
        }

        # `AveragePrototyper` has no parameters at all, which is a legitimately
        #     empty set rather than a group that went missing.
        assert rates in ({expected}, set()), f"{name} at {rates}"


@pytest.mark.parametrize("config_file", RUNGS)
def test_the_resolved_rates_are_written_back_for_save_args(config_file):
    """
    `train.py` calls `save_args` *after* `build_models` precisely so this
        mapping reaches `args.json`. It records which groups existed as well as
        what each ran at -- `sender_contrast` is absent when the stage is off,
        and that is a fact about the run worth having in the artefact.
    """
    config, built = _build(config_file)
    resolved = config["optimiser"]["resolved_module_lrs"]

    built_groups = {
        name for name, select in builder.MODULE_GROUPS
        if select(built["pair"]) is not None
    }

    assert set(resolved) == built_groups
    assert all(isinstance(v, float) and v > 0 for v in resolved.values())

    base_lr = config["optimiser"]["lr"]
    for name, lr in resolved.items():
        assert lr == config["optimiser"]["module_lr"].get(name, base_lr)


# --------------------------------------------------------------------------
# The exclusion, asserted directly.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("volume.log_score_scale", True),
        ("mix_logit", True),
        ("contrast.contrast_gate", True),
        # The two that deliberately stay with their module.
        ("score_bias", False),
        ("polarity_embedding", False),
        # Nothing else.
        ("blocks.0.attention.q_proj.weight", False),
        ("token_embedding.weight", False),
    ],
)
def test_only_the_scaling_scalars_are_claimed_separately(name, expected):
    assert builder.claimed_separately(name, torch.zeros(())) is expected


def test_every_scalar_group_name_is_a_scalar_group():
    """
    `claimed_separately` matches on `SCALAR_SUFFIXES`, which *is* the group
        names -- there is no second list to fall out of step with the first.
    """
    assert builder.SCALAR_SUFFIXES == tuple(
        name for name, _ in builder.SCALAR_GROUPS
    )
    assert builder.GROUP_NAMES[-1] == "other"


# --------------------------------------------------------------------------
# The partition, over every rung.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config_file", RUNGS)
def test_the_groups_partition_the_pair(config_file):
    """
    Every trainable parameter in exactly one group, and nothing in `other`.

    `other` is the alarm rather than the fix: it catches whatever a future
        architecture adds so that no parameter goes unclipped, and its being
        non-empty on a rung that exists today means a module was built without
        being added to `MODULE_GROUPS`. That is exactly what it was doing before
        this change -- `sender.contrast`'s ten tensors were clipped there, under
        a name that said nothing about them.

    Frozen parameters are in the clip groups but not the optimiser's:
        `clip_grad_norm_` skips anything with no gradient, and `get_optimiser`
        skips frozen tensors entirely. `ViT2` has ten -- its blocks' rotary
        `freqs`.
    """
    _, built = _build(config_file)
    pair = built["pair"]

    grouped = []
    for name, params in builder.group_parameters(pair):
        if name == "other":
            assert not params, (
                "outside MODULE_GROUPS: "
                + ", ".join(
                    n for n, p in pair.named_parameters()
                    if any(p is q for q in params)
                )
            )
            continue
        grouped.extend(id(p) for p in params)

    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for _, p in pair.named_parameters()}


@pytest.mark.parametrize("config_file", RUNGS)
def test_every_trainable_parameter_is_in_exactly_one_optimiser_group(config_file):
    """
    Two groups would step a parameter twice; none would never step it. Neither
        raises, and both would read as an architecture result.
    """
    _, built = _build(config_file)

    grouped = [
        id(p) for g in built["optimiser"].param_groups for p in g["params"]
    ]
    trainable = {id(p) for _, p in _trainable(built["pair"])}

    assert len(grouped) == len(set(grouped))
    assert set(grouped) == trainable


@pytest.mark.parametrize("config_file", RUNGS)
def test_the_scalar_overrides_survive_the_module_groups(config_file):
    """
    The disjointness `claimed_separately` buys, read off the built pair. The
        four scaling scalars are held out of their module's group, and the two
        that are not -- `score_bias` and `polarity_embedding` -- are moved back
        out by `split_out_parameter` afterwards, so all six end at the rate
        their own key names whatever their module's rate is.

    `log_logit_scale` is one of the four again, and the one whose module rate
        differs most from its own on every rung: it sits inside
        `sender_language_model` at 1e-4 and takes 2e-3, a factor of twenty.
    """
    config, built = _build(config_file)
    lr_of = _lr_by_id(built["optimiser"])
    seen = 0

    for suffix, key in SCALAR_OVERRIDES:
        hits = [p for n, p in _trainable(built["pair"]) if n.endswith(suffix)]

        if not hits:
            continue

        seen += 1
        assert {lr_of[id(p)] for p in hits} == {config["optimiser"][key]}, (
            f"{suffix} is not at {key}"
        )

    # Every rung has the speaker's channel scale, the listener's volume and its
    #     offset, so a rung matching fewer than three suffixes would mean the
    #     table had gone stale rather than that the rung was austere.
    assert seen >= 3


@pytest.mark.parametrize("config_file", RUNGS)
def test_a_scalar_group_that_applies_matches_a_parameter(config_file):
    """
    `group_parameters` raises rather than falling back to the module when an
        applicable scalar matches nothing, because the fallback is silent: the
        scalar would be clipped inside its module's norm and stepped at its
        module's rate, and the run would look fine.
    """
    _, built = _build(config_file)
    pair = built["pair"]
    by_name = dict(builder.group_parameters(pair))

    for name, applies_to in builder.SCALAR_GROUPS:
        if applies_to(pair):
            assert len(by_name[name]) == 1, f"{name} matched {by_name[name]}"
            assert by_name[name][0].dim() == 0, f"{name} is not a scalar"
        else:
            assert by_name[name] == []


@pytest.mark.parametrize("config_file", RUNGS)
def test_pass_builds_over_the_groups_and_still_reports_base(config_file):
    """
    `PASS` deep-copies the groups at construction and zips them `strict=True`,
        so this is the test that would fail if `build_models` ever regrouped
        after the scheduler existed. Group 0 stays the one `get_optimiser` made.

    Read off `original_param_groups` rather than off `PASS.lr`, which reports
        the *live* rate. Under a warm-up the live rate at step 0 is zero -- as it
        should be -- and this test is about the groups surviving construction,
        not about the schedule. It asserted the live rate until the warm-up
        started working, at which point every rung carrying one began failing
        here. See `tests/test_lr_schedule.py`.
    """
    config, built = _build(config_file)

    scheduler = PASS(
        train.build_lr_schedule(config, 1_000, 32),
        built["pair"],
        built["optimiser"],
        scaler=None,
        range_test=False,
        # As `train.py` builds it: the floor lives in the descending cycle, not
        #     in `PASS`. See `train.build_lr_schedule`.
        cool_point_multiplier=0.0,
    )

    assert scheduler.original_param_groups[0]["lr"] == config["optimiser"]["lr"]
    assert len(scheduler.original_param_groups) == len(
        built["optimiser"].param_groups
    )


# --------------------------------------------------------------------------
# What gets reported.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config_file", RUNGS)
def test_every_group_reports_a_norm_on_every_rung(config_file):
    """
    The metrics header has to keep its shape across a resume against a config
        that toggles a stage, so `clip_gradients` reports a key per group on
        every rung and NaN-fills the ones that do not exist -- the same rule the
        contrast columns follow.

    Gradients are not needed for this: what is being asserted is the shape of
        the report, and a group with nothing to clip is reported as NaN for the
        same reason a group that does not exist is.
    """
    _, built = _build(config_file)
    norms = train.clip_gradients(built["pair"], 1.0)

    assert tuple(norms) == builder.GROUP_NAMES

    # No gradients anywhere, so every group is NaN. The value is asserted in
    #     `tests/test_backbones.py`, which builds a pair that has backpropped.
    assert all(math.isnan(v) for v in norms.values())

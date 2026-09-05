"""
The three channel and readout flags -- `[sender_language_model]
normalise_logits`, `[receiver_discriminator] scale_score` and `bias_score` -- as
`models/builder.py` builds them and `train.py` reports them.

All three default true and true is today's behaviour, so this file is almost
entirely about the *off* paths. The one exception is the first test, and it is
the load-bearing one: the guarantee the whole design rests on is that a config
which does not mention any of the keys builds a pair bit-identical to the one
the code built before the keys existed. That is why every edit behind these
flags is shaped `if flag: <the existing expression, character for character>;
else: <new>` -- the true branch is never rewritten, which is what makes the
default identical rather than merely equivalent.

**`scale_score` and `bias_score` replace a single `normalise_score`**, and this
file is where the reason is testable. That key gated four things: the two
operand norms, the `/sqrt(d)` calibration and both readout scalars. The norms
left in the interface hoist, and this round moves the calibration into the
design -- so what a config can still choose is the volume and the offset, and it
chooses them separately because they answer separate questions. The calibration
being unconditional is asserted directly, on both discriminators, with both keys
off.

Two facts make the identity testable rather than merely argued.
`nn.Parameter(torch.zeros(()))` and `nn.LayerNorm(elementwise_affine=False)`
consume no RNG, so making them conditional cannot shift which draw any *other*
parameter gets at a given seed; and the conditional objects hold no parameters
of their own except the two scalars, which the flag removes deliberately. The
argument is what says the test should pass. The test is what says it does.

**Why the flags exist at all.** No ShapeWorld run has learned shape since 17
August. `60f9094` moved the backbone to `ResNet18SmallInput` on the 18th and
`4248fca` gave the bilinear comparer its operand norms, its `/sqrt(d)`
calibration and `log_score_scale` on the 19th; nothing since either has learned
shape, and the two are perfectly confounded. `experiments/silhouette_titration_
norms/` is the sweep these flags were added for, against
`silhouette_titration_resnet18/` as its control. See DEFAULT.toml beside each
key.

Runnable without pytest:  python tests/test_score_norms.py
"""

import copy
import itertools
import math

import pytest
import torch

import _bootstrap  # noqa: F401

import models.builder as builder
import models.receiver as receiver
import models.sender as models_sender
import parse_config
import train

SHAPEWORLD_FEATS = (3, 64, 64)

SEED = 20260902

# Every reachable rung: three independent booleans, so eight, and they are
#     generated rather than listed so that adding a flag cannot leave a corner
#     silently untested. `(True, True, True)` is today. The four listener
#     corners are the point of the split -- a volume with no threshold, a
#     threshold with no volume, both, neither -- and only two of them were
#     expressible under the `normalise_score` these keys replace.
#
# `(False, False, False)` is the corner the cluster smoke test runs, because it
#     exercises every off-path at once.
COMBINATIONS = tuple(itertools.product((True, False), repeat=3))


class _Loader:
    class dataset:
        n_feats = SHAPEWORLD_FEATS


def _config(normalise_logits=None, scale_score=None, bias_score=None):
    """
    `DEFAULT.toml`, with each flag either stated or *deleted*.

    Deleting rather than leaving the default in place is the point of the first
        test: the modules read these through `kwargs.get(..., True)`, so absence
        and an explicit `true` must reach the same code, and only a config with
        the key genuinely missing tests that.
    """
    config = parse_config.get_config()
    config["cuda"] = False

    for table, key, value in (
        ("sender_language_model", "normalise_logits", normalise_logits),
        ("receiver_discriminator", "scale_score", scale_score),
        ("receiver_discriminator", "bias_score", bias_score),
    ):
        if value is None:
            del config[table][key]
        else:
            config[table][key] = value

    return config


def _build(config):
    """A real pair through `models.builder`, at a fixed seed."""
    torch.manual_seed(SEED)
    return builder.build_models({"train": _Loader()}, copy.deepcopy(config))


# The speaker requires an even example count with the positives first; the
#     listener scores a candidate set of its own. Fixed here rather than read
#     from the config because nothing in this file varies with either.
BATCH = 2
EXAMPLES = 4
CANDIDATES = 4


def _inputs(offset=1):
    torch.manual_seed(SEED + offset)

    samples = torch.randn(BATCH, EXAMPLES, *SHAPEWORLD_FEATS)
    targets = torch.zeros(BATCH, EXAMPLES)
    targets[:, : EXAMPLES // 2] = 1.0
    referents = torch.randn(BATCH, CANDIDATES, *SHAPEWORLD_FEATS)

    return samples, targets, referents


def _forward(pair):
    """
    One forward pass of the whole pair on fixed input, in `train()` mode.

    `train()` rather than `eval()` deliberately, and not only because the
        speaker's sampler is greedy outside training: measuring a backbone under
        `eval()` with untrained `BatchNorm` running statistics gives an artefact
        of its own. See docs/measurement.md.
    """
    samples, targets, referents = _inputs()

    pair.train()

    with torch.no_grad():
        torch.manual_seed(SEED + 7)
        messages, _ = pair.sender(samples, targets)
        return pair.receiver(referents, messages)


# --------------------------------------------------------------------------
# The guarantee.
# --------------------------------------------------------------------------

def test_the_default_path_is_bit_identical():
    """
    A config with neither key builds exactly the pair a config stating both as
        `true` builds: the same `state_dict` keys, every tensor equal to the
        bit, and the same scores out of one forward pass on fixed input.

    Equality of *keys* is half the assertion and not a formality. The flags work
        by not constructing things, so a true branch that had been rewritten
        into an equivalent-but-different arrangement would most likely show up
        here first, as a parameter that moved, gained a prefix, or stopped
        existing.
    """
    absent = _build(_config(None, None, None))["pair"]
    stated = _build(_config(True, True, True))["pair"]

    absent_state = absent.state_dict()
    stated_state = stated.state_dict()

    assert list(absent_state) == list(stated_state)

    for key, tensor in absent_state.items():
        assert torch.equal(tensor, stated_state[key]), key

    assert torch.equal(_forward(absent), _forward(stated))


def test_the_scalars_are_present_by_default_and_absent_when_off():
    """
    The three parameters the flags remove, named in the `state_dict` rather than
        inferred from behaviour -- this is what "checkpoints do not cross the
        flags" means, and it is why nothing written under one setting loads
        under another whatever `resume` says.

    One parameter per flag, and this is the assertion the split exists for:
        `log_score_scale` tracks `scale_score` and `score_bias` tracks
        `bias_score`, independently, so the two mixed corners are real
        configurations rather than an unreachable half of a truth table.

    Absent rather than frozen, so `split_out_parameter`'s suffix match and
        `SCALAR_GROUPS` see the truth instead of a parameter that exists and
        never moves.
    """
    for normalise_logits, scale_score, bias_score in COMBINATIONS:
        pair = _build(
            _config(normalise_logits, scale_score, bias_score)
        )["pair"]
        keys = set(pair.state_dict())

        def has(suffix):
            return any(key.endswith(suffix) for key in keys)

        assert has("log_logit_scale") == normalise_logits
        assert has("log_score_scale") == scale_score
        assert has("score_bias") == bias_score


# --------------------------------------------------------------------------
# All four rungs build, step and clip.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "normalise_logits,scale_score,bias_score", COMBINATIONS
)
def test_build_models_survives_every_flag_combination(
    normalise_logits, scale_score, bias_score
):
    """
    Every rung constructs, produces a loss, backpropagates, clips through
        `train.clip_gradients` and takes an optimiser step.

    `build_models` is where the gates are read, and both `group_parameters` and
        `split_out_parameter` raise deliberately when an applicable group
        matches nothing -- which is the failure this whole set of gates exists
        to prevent, and it fires at build time, before a step runs. So a missing
        gate does not reach this far; what this catches is the off-path
        arithmetic being wrong in a way construction alone would not show.
    """
    built = _build(_config(normalise_logits, scale_score, bias_score))
    pair, optimiser = built["pair"], built["optimiser"]

    pair.train()

    samples, targets, referents = _inputs(offset=2)

    messages, _ = pair.sender(samples, targets)
    scores = pair.receiver(referents, messages)

    assert torch.isfinite(scores).all()

    scores.square().mean().backward()

    norms = train.clip_gradients(pair, 1.0)

    # The header keeps its shape: one entry per group name whatever the flags,
    #     NaN where the group does not exist on this rung.
    assert list(norms) == list(builder.GROUP_NAMES)

    optimiser.step()
    pair.sender.language_model.project_channel()


@pytest.mark.parametrize(
    "normalise_logits,scale_score,bias_score", COMBINATIONS
)
def test_the_scalar_groups_gate_on_the_flags(
    normalise_logits, scale_score, bias_score
):
    """
    `group_parameters` partitions without raising in all eight, and the two
        gated groups hold a parameter exactly when their flag is on.

    The partition is what the raise protects: a scalar that fell into its
        module's group instead would be clipped inside that module's norm, which
        it would inflate on the way, and stepped at the module's rate rather
        than its own. Neither is visible from a metrics file.

    `score_bias` has no group of its own -- it is an offset and belongs to its
        module's clip norm -- so `bias_score` is asserted here only through the
        partition staying total and disjoint. Its own gate is
        `SPLIT_LEARNING_RATES`', and `_build` above would have raised at
        construction if that gate and the parameter disagreed.
    """
    pair = _build(_config(normalise_logits, scale_score, bias_score))["pair"]

    groups = dict(builder.group_parameters(pair))

    assert set(groups) == set(builder.GROUP_NAMES)
    assert bool(groups["log_logit_scale"]) == normalise_logits
    assert bool(groups["log_score_scale"]) == scale_score

    # Total and disjoint, which is the invariant the whole table serves.
    claimed = [id(p) for params in groups.values() for p in params]
    assert len(claimed) == len(set(claimed))
    assert set(claimed) == {id(p) for p in pair.parameters()}
    assert not groups["other"]


# --------------------------------------------------------------------------
# What the off path computes.
# --------------------------------------------------------------------------

def test_the_calibration_survives_both_keys_being_off():
    """
    With `scale_score` and `bias_score` both off, `BilinearDiscriminator`
        computes `r_j . W m / sqrt(d)` -- no volume, no offset, but **still the
        calibration**. Checked against the einsum written out here by hand.

    This is the assertion that says the `/sqrt(d)` is design rather than a rung.
        Under the `normalise_score` these two keys replace, the same corner
        returned the bare form; that form was reachable and is no longer, and it
        is not missed. Its purpose had been to revert to `ce7d6a5`, and to
        jayelm's `CopyListener.compare`, and the interface hoist already ended
        that: his `compare` is a dot product on raw backbone output, where this
        one runs on operands `Receiver` has normalised whatever any key says.
        The bare form on normalised operands reverted to nothing, opened at a
        width-dependent spread -- sd 18.4 and BCE 6.87 at d = 1024, against
        `ln 2` = 0.693 -- and no run has used it.

    The module holds no norms under any setting, which is also asserted: the two
        it used to build are `Receiver`'s interface norms now.
    """
    torch.manual_seed(SEED)
    discriminator = receiver.BilinearDiscriminator(
        referent_embedding_size=8,
        message_width=6,
        scale_score=False,
        bias_score=False,
    )

    assert not discriminator.learns_score_scale
    assert not discriminator.learns_score_bias
    assert not hasattr(discriminator, "referent_layer_norm")
    assert not hasattr(discriminator, "message_layer_norm")

    torch.manual_seed(SEED + 1)
    referents = torch.randn(5, 4, 8)
    message_repr = torch.randn(5, 3, 6)

    bare = torch.einsum(
        "ijh,ih->ij",
        (referents, discriminator.bilinear(message_repr[:, -1, :])),
    )
    expected = bare / math.sqrt(8)

    assert torch.equal(discriminator(referents, message_repr), expected)

    # And it is not accidentally the identity at this width: the calibration is
    #     a real division and the bare form is a different function.
    assert not torch.allclose(bare, expected)


def test_the_two_readout_keys_are_independent():
    """
    Each key builds its own scalar and neither reaches the other's, so the two
        mixed corners compute what they say: a volume alone leaves the origin
        where the calibration puts it, and an offset alone shifts an
        uncalibrated-loudness score.

    Under the single `normalise_score` these replace, neither corner existed.
        That is the whole of what the split buys, so it is asserted directly
        rather than only through the `state_dict`.
    """
    torch.manual_seed(SEED + 1)
    referents = torch.randn(5, 4, 8)
    message_repr = torch.randn(5, 3, 6)

    def built(scale_score, bias_score):
        torch.manual_seed(SEED)
        return receiver.BilinearDiscriminator(
            referent_embedding_size=8,
            message_width=6,
            scale_score=scale_score,
            bias_score=bias_score,
        )

    neither = built(False, False)
    calibrated = neither(referents, message_repr)

    volume_only = built(True, False)
    offset_only = built(False, True)

    assert volume_only.learns_score_scale
    assert not volume_only.learns_score_bias
    assert not offset_only.learns_score_scale
    assert offset_only.learns_score_bias

    # Both scalars open at their identity -- `log_score_scale` at 0, so a
    #     multiplier of 1.0, and `score_bias` at 0 -- so all three agree until
    #     one of them is moved. That is what makes the opening of a run
    #     independent of these keys.
    assert torch.allclose(volume_only(referents, message_repr), calibrated)
    assert torch.allclose(offset_only(referents, message_repr), calibrated)

    with torch.no_grad():
        volume_only.log_score_scale.fill_(math.log(3.0))
        offset_only.score_bias.fill_(0.5)

    assert torch.allclose(
        volume_only(referents, message_repr), 3.0 * calibrated
    )
    assert torch.allclose(
        offset_only(referents, message_repr), calibrated + 0.5
    )


def test_the_attention_readout_is_a_passthrough_when_off():
    """
    With both readout keys off, `AttentionDiscriminator` returns
        `(1 - a) * bilinear + a * attention` -- no scale, no bias -- over a
        composed bilinear path that is still calibrated.

    The composed path carries neither scalar under *any* setting of the keys,
        because a volume on a branch is degenerate with `mix_logit` and a
        constant on it with the outer offset. What the keys reach is the outer
        readout, downstream of the mix, which is where the listener's one volume
        and one offset live.

    Its input and memory norms survive the flag, as they always have -- but
        they are `Receiver`'s interface norms now rather than this module's, so
        what is asserted here is that the module holds none of its own and that
        the arithmetic below is the whole of what it does. The reason they are
        flag-immune is unchanged: a post-norm stack normalises its own stream
        and never its memory, so removing them would break the stack rather than
        change how loudly it speaks.
    """
    settings = dict(parse_config.get_config()["receiver_discriminator"])
    settings["scale_score"] = False
    settings["bias_score"] = False

    torch.manual_seed(SEED)
    discriminator = receiver.AttentionDiscriminator(
        referent_embedding_size=settings["d_model"],
        message_width=settings["d_model"],
        **settings,
    )

    assert not discriminator.learns_score_scale
    assert not discriminator.learns_score_bias
    assert not discriminator.bilinear.learns_score_scale
    assert not discriminator.bilinear.learns_score_bias

    # No norms and no projections of its own, under either setting: they are
    #     `Receiver`'s interfaces, and this module declares the widths it wants
    #     them to deliver at.
    for attribute in (
        "referent_layer_norm",
        "referent_adapter",
        "memory_layer_norm",
        "memory_adapter",
    ):
        assert not hasattr(discriminator, attribute), attribute

    # One width for the whole slot, which is what lets the composed bilinear
    #     path read the same tensors the stack does.
    assert discriminator.referent_input_size == settings["d_model"]
    assert discriminator.message_input_size == settings["d_model"]

    discriminator.eval()

    torch.manual_seed(SEED + 1)
    referents = torch.randn(2, 4, settings["d_model"])
    message_repr = torch.randn(2, 3, settings["d_model"])

    with torch.no_grad():
        scores = discriminator(referents, message_repr)

        attention = discriminator.decision(
            discriminator.referent_decoder(referents, message_repr)
        ).squeeze(-1)
        bilinear = discriminator.bilinear(referents, message_repr)

        weight = discriminator.mix_weight
        expected = (1.0 - weight) * bilinear + weight * attention

    assert torch.equal(scores, expected)


def test_an_unnormalised_channel_skips_the_norm_and_the_gain():
    """
    With `normalise_logits` off the speaker's raw logits reach the sampler: no
        `layer_norm_logits`, and no `logit_scale` to multiply them by.

    Read off `sample_symbols`' second return, which is the tap the survival
        diagnostics are measured from -- masked, and normalised only when the
        flag says so. That is also what makes `logit_margin`,
        `logit_prior_share` and the two survival columns stop being comparable
        across this flag while staying computable: they are stated in units of
        the logits' own standard deviation, and off the norm there is no such
        unit. See DEFAULT.toml beside the key.
    """
    settings = dict(parse_config.get_config()["sender_language_model"])
    settings["normalise_logits"] = False

    torch.manual_seed(SEED)
    language_model = models_sender.SenderGRULM(
        settings["d_model"], **settings
    )
    language_model.train()

    assert not language_model.normalises_logits
    assert not hasattr(language_model, "log_logit_scale")

    torch.manual_seed(SEED + 1)
    logits = torch.randn(2, 3, settings["vocabulary"] + 4)

    _, tapped = language_model.sample_symbols(logits)

    expected = models_sender.mask_reserved_tokens(logits)

    assert torch.equal(tapped, expected)

    # Both projection and reset are no-ops rather than errors, so `train.py`'s
    #     `optimiser_step` and `diagnostics/bootstrap_probe.py` need no branch.
    language_model.project_channel()
    language_model.reset_channel_scale()


def test_a_config_naming_the_retired_key_is_rejected():
    """
    `normalise_score` no longer exists, and a config that still names it must
        *fail* rather than run the default arm under a filename saying it did
        not.

    This is the failure mode the whole gate is built to avoid, and it is not the
        one the check above catches: a missing key is loud because the module
        reading it gets nothing, where a retired key is silent because nothing
        reads it at all. It merges under `DEFAULT.toml` like any other key, so
        without a rejection here `experiments/silhouette_titration_norms/` --
        every cell of which states it -- would validate, build, and run the
        arrangement it was written to turn off.

    The message has to name the replacements, because there is no mechanical
        translation: the old `false` also removed the `1/sqrt(d)`, and nothing
        does that now.
    """
    config = parse_config.get_config()
    config["cuda"] = False
    config["receiver_discriminator"]["normalise_score"] = False

    with pytest.raises(parse_config.InvalidConfig) as raised:
        parse_config.validate_config(config)

    message = str(raised.value)

    assert "normalise_score" in message
    assert "scale_score" in message
    assert "bias_score" in message

    # `true` is rejected too. It is not a harmless restatement of the default:
    #     the key it restates is gone, so it says nothing about what runs.
    config["receiver_discriminator"]["normalise_score"] = True

    with pytest.raises(parse_config.InvalidConfig):
        parse_config.validate_config(config)


def test_the_channel_columns_are_nan_rather_than_absent():
    """
    `train.py` writes every scalar column on every rung, NaN where the parameter
        does not exist, so the metrics header keeps its shape across the flags
        exactly as it does across a resume against a config that toggles a
        stage.

    Asserted through `clip_gradients`, which is where the gradient-norm half of
        that promise is made, and through the modules the metric half reads.
    """
    pair = _build(_config(False, False, False))["pair"]

    for name, params in builder.group_parameters(pair):
        assert name in builder.GROUP_NAMES

    # `train_clip_log_logit_scale` and `train_clip_log_score_scale` keep their
    #     places in the header and read NaN, which is what `GROUP_NAMES` being
    #     untouched by this change buys.
    assert "log_logit_scale" in builder.GROUP_NAMES
    assert "log_score_scale" in builder.GROUP_NAMES

    norms = train.clip_gradients(pair, 1.0)

    assert math.isnan(norms["log_logit_scale"])
    assert math.isnan(norms["log_score_scale"])

    # And the three the metrics block reads directly. `train_score_scale` and
    #     `train_score_bias` are read off separate attributes now, so each
    #     column goes NaN on its own key rather than on a shared one.
    assert not pair.sender.language_model.normalises_logits
    assert not pair.receiver.discriminator.learns_score_scale
    assert not pair.receiver.discriminator.learns_score_bias


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

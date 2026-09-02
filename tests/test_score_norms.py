"""
The two normalisation flags, `[sender_language_model] normalise_logits` and
`[receiver_discriminator] normalise_score`, as `models/builder.py` builds them
and `train.py` reports them.

Both default true and true is today's behaviour, so this file is almost
entirely about the *off* path. The one exception is the first test, and it is
the load-bearing one: the guarantee the whole design rests on is that a config
which does not mention either key builds a pair bit-identical to the one the
code built before the keys existed. That is why every edit behind these flags is
shaped `if flag: <the existing expression, character for character>; else:
<new>` -- the true branch is never rewritten, which is what makes the default
identical rather than merely equivalent.

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

# The four reachable rungs. `(True, True)` is today; `(True, False)` is the
#     listener revert to `ce7d6a5`'s arithmetic; `(False, True)` is the speaker
#     alone, which is not a revert; `(False, False)` is both, and the corner the
#     cluster smoke test runs because it exercises both off-paths at once.
COMBINATIONS = (
    (True, True),
    (True, False),
    (False, True),
    (False, False),
)


class _Loader:
    class dataset:
        n_feats = SHAPEWORLD_FEATS


def _config(normalise_logits=None, normalise_score=None):
    """
    `DEFAULT.toml`, with each flag either stated or *deleted*.

    Deleting rather than leaving the default in place is the point of the first
        test: the modules read these through `kwargs.get(..., True)`, so absence
        and an explicit `true` must reach the same code, and only a config with
        the key genuinely missing tests that.
    """
    config = parse_config.get_config()
    config["cuda"] = False

    if normalise_logits is None:
        del config["sender_language_model"]["normalise_logits"]
    else:
        config["sender_language_model"]["normalise_logits"] = normalise_logits

    if normalise_score is None:
        del config["receiver_discriminator"]["normalise_score"]
    else:
        config["receiver_discriminator"]["normalise_score"] = normalise_score

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
    absent = _build(_config(None, None))["pair"]
    stated = _build(_config(True, True))["pair"]

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

    Absent rather than frozen, so `split_out_parameter`'s suffix match and
        `SCALAR_GROUPS` see the truth instead of a parameter that exists and
        never moves.
    """
    for normalise_logits, normalise_score in COMBINATIONS:
        pair = _build(_config(normalise_logits, normalise_score))["pair"]
        keys = set(pair.state_dict())

        def has(suffix):
            return any(key.endswith(suffix) for key in keys)

        assert has("log_logit_scale") == normalise_logits
        assert has("log_score_scale") == normalise_score
        assert has("score_bias") == normalise_score


# --------------------------------------------------------------------------
# All four rungs build, step and clip.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("normalise_logits,normalise_score", COMBINATIONS)
def test_build_models_survives_every_flag_combination(
    normalise_logits, normalise_score
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
    built = _build(_config(normalise_logits, normalise_score))
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


@pytest.mark.parametrize("normalise_logits,normalise_score", COMBINATIONS)
def test_the_scalar_groups_gate_on_the_flags(normalise_logits, normalise_score):
    """
    `group_parameters` partitions without raising in all four, and the two gated
        groups hold a parameter exactly when their flag is on.

    The partition is what the raise protects: a scalar that fell into its
        module's group instead would be clipped inside that module's norm, which
        it would inflate on the way, and stepped at the module's rate rather
        than its own. Neither is visible from a metrics file.
    """
    pair = _build(_config(normalise_logits, normalise_score))["pair"]

    groups = dict(builder.group_parameters(pair))

    assert set(groups) == set(builder.GROUP_NAMES)
    assert bool(groups["log_logit_scale"]) == normalise_logits
    assert bool(groups["log_score_scale"]) == normalise_score

    # Total and disjoint, which is the invariant the whole table serves.
    claimed = [id(p) for params in groups.values() for p in params]
    assert len(claimed) == len(set(claimed))
    assert set(claimed) == {id(p) for p in pair.parameters()}
    assert not groups["other"]


# --------------------------------------------------------------------------
# What the off path computes.
# --------------------------------------------------------------------------

def test_an_unnormalised_score_is_the_bare_bilinear_form():
    """
    With `normalise_score` off, `BilinearDiscriminator` computes `r_j . W m` and
        nothing else: no layer norm on either operand, no `/sqrt(d)`, no volume,
        no offset. Checked against the einsum written out here by hand.

    That form is what `ce7d6a5` scored with and what jayelm's
        `CopyListener.compare` has always computed, which is the point -- the
        listener arm of this flag is a revert rather than a new arrangement.
    """
    torch.manual_seed(SEED)
    discriminator = receiver.BilinearDiscriminator(
        referent_embedding_size=8,
        message_width=6,
        normalise_score=False,
    )

    assert not discriminator.learns_score_scale
    assert not hasattr(discriminator, "referent_layer_norm")
    assert not hasattr(discriminator, "message_layer_norm")

    torch.manual_seed(SEED + 1)
    referents = torch.randn(5, 4, 8)
    message_repr = torch.randn(5, 3, 6)

    expected = torch.einsum(
        "ijh,ih->ij",
        (referents, discriminator.bilinear(message_repr[:, -1, :])),
    )

    assert torch.equal(discriminator(referents, message_repr), expected)

    # And the normalised arm still divides by `sqrt(d)`, so the two are not
    #     accidentally the same function.
    torch.manual_seed(SEED)
    normalised = receiver.BilinearDiscriminator(
        referent_embedding_size=8,
        message_width=6,
        normalise_score=True,
    )

    assert not torch.allclose(
        normalised(referents, message_repr), expected
    )


def test_the_attention_readout_is_a_passthrough_when_off():
    """
    With `normalise_score` off, `AttentionDiscriminator` returns
        `(1 - a) * bilinear + a * attention` -- no scale, no bias -- and the
        bilinear path it composes is unnormalised too.

    Its own `referent_layer_norm` and `memory_layer_norm` survive the flag and
        are asserted to. They are the decoder stack's input and memory norms,
        not score norms: a post-norm stack normalises its own stream and never
        its memory, which is why they exist, and removing them would break the
        stack rather than change how loudly it speaks.
    """
    settings = dict(parse_config.get_config()["receiver_discriminator"])
    settings["normalise_score"] = False

    torch.manual_seed(SEED)
    discriminator = receiver.AttentionDiscriminator(
        referent_embedding_size=settings["d_model"],
        message_width=settings["d_model"],
        **settings,
    )

    assert not discriminator.learns_score_scale
    assert not discriminator.bilinear.learns_score_scale
    assert not discriminator.bilinear.normalises_score

    # The stack's own two norms are untouched by the flag.
    assert isinstance(discriminator.referent_layer_norm, torch.nn.LayerNorm)
    assert isinstance(discriminator.memory_layer_norm, torch.nn.LayerNorm)

    discriminator.eval()

    torch.manual_seed(SEED + 1)
    referents = torch.randn(2, 4, settings["d_model"])
    message_repr = torch.randn(2, 3, settings["d_model"])

    with torch.no_grad():
        scores = discriminator(referents, message_repr)

        adapted = discriminator.referent_layer_norm(
            discriminator.referent_adapter(referents)
        )
        memory = discriminator.memory_layer_norm(
            discriminator.memory_adapter(message_repr)
        )
        attention = discriminator.decision(
            discriminator.referent_decoder(adapted, memory)
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


def test_the_channel_columns_are_nan_rather_than_absent():
    """
    `train.py` writes every scalar column on every rung, NaN where the parameter
        does not exist, so the metrics header keeps its shape across the flags
        exactly as it does across a resume against a config that toggles a
        stage.

    Asserted through `clip_gradients`, which is where the gradient-norm half of
        that promise is made, and through the modules the metric half reads.
    """
    pair = _build(_config(False, False))["pair"]

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

    # And the two the metrics block reads directly.
    assert not pair.sender.language_model.normalises_logits
    assert not pair.receiver.discriminator.learns_score_scale


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

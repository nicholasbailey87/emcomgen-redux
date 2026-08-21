"""
The fourteen ablation rungs, built the way `train.py` builds them.

Nothing checked this before, and the cost of that was rung 7. Its speaker came
out at 533,943 parameters against the GRU baseline's 6,813,499 -- twelve times
smaller -- because `SenderTransformerLM` pins its width to the vision model's
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
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import broccoli.transformer  # noqa: E402
import models.builder  # noqa: E402
import parse_config  # noqa: E402

CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "ablation", "configs"
)

SHAPEWORLD_FEATS = (3, 64, 64)
BIRDS_FEATS = (3, 224, 224)

# Even rungs are CUB, odd rungs ShapeWorld.
RUNGS = sorted(f for f in os.listdir(CONFIG_DIR) if f.endswith(".toml"))


def _feats(config_file):
    return BIRDS_FEATS if int(config_file[:2]) % 2 == 0 else SHAPEWORLD_FEATS


def _name(config_file):
    return "cub" if int(config_file[:2]) % 2 == 0 else "shapeworld"


def _pair(config_file):
    """A real rung, through `models.builder`, with a stub dataloader."""
    config = parse_config.get_config(os.path.join(CONFIG_DIR, config_file))
    config["cuda"] = False

    class _Dataset:
        n_feats = _feats(config_file)
        name = _name(config_file)

    class _Loader:
        dataset = _Dataset()

    built = models.builder.build_models({"train": _Loader()}, config)
    return config, built["pair"]


def _count(module):
    return sum(p.numel() for p in module.parameters())


def test_there_are_fourteen_rungs():
    """A rung added or renamed without the counts below being revisited."""
    assert len(RUNGS) == 14


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


# The sizes the ladder is built around. Sender vision, speaker language model,
# and listener comparer for each dataset's baseline and its fully-Transformer
# counterpart. See the rung headers for where each number comes from.
@pytest.mark.parametrize(
    "config_file,module,expected",
    [
        # ShapeWorld: the CNN/GRU baseline.
        ("01_shapeworld_baseline.toml", "sender.feat_model", 11_168_832),
        ("01_shapeworld_baseline.toml", "sender.language_model", 5_764_923),
        # Reads 512-d referents from the ResNet, where rung 11's reads 320-d
        # from the ViT, and `BilinearGRUComparer`'s projection is sized from
        # that width -- so these two are not the like-for-like pair. Rung 11's
        # counterpart is rung 9's comparer, at 5,015,553, which the
        # cross-attention comparer meets at 1.05x.
        #
        # The odd digit is `log_score_scale`, one 0-d parameter, and it is now
        # `BilinearGRUComparer`'s alone: the cross-attention comparer gave up
        # both that and `decision.bias` when its readout became a batch norm at
        # a fixed gain, so its count is two lower than it was. The earlier
        # movements were 640, when `referent_layer_norm` gave up a `gamma` and a
        # `beta` at `d_model = 320`, and another 320 when `referent_adapter`
        # gave up its bias -- without which `LN(W(cx)) = LN(W(x))` does not hold
        # and the score is not exactly free of the vision model's scale.
        #
        # The cross-attention rows below are stale by considerably more than
        # those two, and were before the readout changed: measured against the
        # current tree both rung 11 and rung 12 build a comparer of 5,357,348,
        # where this table claims 5,272,606. That gap is 84,744 and none of it
        # is the readout. These counts are the capacity-matching argument the
        # whole ladder rests on, so they want re-deriving deliberately rather
        # than being edited to match whatever builds today.
        ("01_shapeworld_baseline.toml", "receiver.comparer", 5_212_161),
        # ShapeWorld: the Transformer arm it is compared against. The speaker's
        # language model is the autoregressive decoder, four blocks at message
        # length -- see rung 7 for why four rather than five.
        ("11_shapeworld_receiver_cross_attention.toml", "sender.feat_model", 10_084_940),
        ("11_shapeworld_receiver_cross_attention.toml", "sender.language_model", 5_848_303),
        ("11_shapeworld_receiver_cross_attention.toml", "receiver.comparer", 5_272_606),
        # CUB: the CNN/GRU baseline.
        ("02_birds_baseline.toml", "sender.feat_model", 11_176_512),
        ("02_birds_baseline.toml", "sender.language_model", 5_774_073),
        ("02_birds_baseline.toml", "receiver.comparer", 5_212_161),
        # CUB: the Transformer arm it is compared against.
        ("12_birds_receiver_cross_attention.toml", "sender.feat_model", 12_338_428),
        ("12_birds_receiver_cross_attention.toml", "sender.language_model", 5_854_069),
        ("12_birds_receiver_cross_attention.toml", "receiver.comparer", 5_272_606),
        # The parallel arm, rungs 13 and 14. Five encoder blocks over the latent
        # array rather than four decoder blocks over the message, which is a
        # different speaker at a different size against the same baseline --
        # 0.965x where the decoder is 1.015x. Pinned here because the pair is
        # what makes a difference between rungs 11 and 13 readable: if either
        # moves without the other, the two are no longer answering the same
        # question.
        ("13_shapeworld_sender_transformer_lm_latent.toml", "sender.language_model", 5_558_454),
        ("14_birds_sender_transformer_lm_latent.toml", "sender.language_model", 5_563_260),
        # And the listener, which is the half that has to be *identical* to 11
        # and 12 for the contrast to be about emission at all. 13 and 14 are
        # those rungs with the speaker's `bidirectional` flipped and nothing
        # else, so these two assertions are the ones that would catch the pair
        # drifting apart.
        ("13_shapeworld_sender_transformer_lm_latent.toml", "receiver.comparer", 5_272_606),
        ("14_birds_sender_transformer_lm_latent.toml", "receiver.comparer", 5_272_606),
        ("13_shapeworld_sender_transformer_lm_latent.toml", "sender.feat_model", 10_084_940),
        ("14_birds_sender_transformer_lm_latent.toml", "sender.feat_model", 12_338_428),
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
        ("01_shapeworld_baseline.toml", "11_shapeworld_receiver_cross_attention.toml", 0.05),
        ("02_birds_baseline.toml", "12_birds_receiver_cross_attention.toml", 0.05),
        # Both Transformer arms are matched to the baseline, not to each other:
        # the decoder lands at 1.015x and the parallel arm at 0.965x, and no
        # integer depth puts them at the same place. See rung 13's `layers`.
        ("01_shapeworld_baseline.toml", "13_shapeworld_sender_transformer_lm_latent.toml", 0.05),
        ("02_birds_baseline.toml", "14_birds_sender_transformer_lm_latent.toml", 0.05),
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


# Both arms: the rotary module is the latent self-attention on one and the
# decoder's causal self-attention on the other, so neither covers the other.
@pytest.mark.parametrize(
    "config_file",
    [
        "11_shapeworld_receiver_cross_attention.toml",
        "13_shapeworld_sender_transformer_lm_latent.toml",
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

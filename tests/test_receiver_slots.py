"""
Tests for the listener's two-slot split, in code/models/receiver.py.

Runnable without pytest:  python tests/test_receiver_slots.py

`config['receiver']['comparer']` used to name one module that did two jobs:
encoding the message, and comparing it against the candidates. The two comparers
split almost exactly in half along that line -- `BilinearGRUComparer` was a
789,504-parameter GRU and a 196,608-parameter bilinear form, and
`TransformerCrossAttentionComparer` was two 2.3M decoder stacks -- so a rung that
swapped one for the other moved both halves at once, and "does attention help
compositionality" could not be attributed to either.

The slots are four language models over one discriminator,
`BilinearDiscriminator`: `ReceiverGRULM`, the two arms of
`ReceiverTransformerLM`, and `ReceiverCrossAttentionLM`. There used to be a
second discriminator, an attention stack in which the candidates read each
other; it was removed on 2026-10-01, and its tests with it.

The first section below is the safety net for the refactor: at `dropout = 0` and
one unidirectional layer, `ReceiverGRULM + BilinearDiscriminator` must reproduce
the pre-split module bit for bit. It is the only pairing that can be pinned that
exactly -- the others did not exist -- so everything else here is a property
test.

It has to be at `dropout = 0` because the mask moved, twice. `Receiver` draws
one mask per referent interface, downstream of that interface's norm, where the
modules it replaced masked once upstream of their own norm. A LayerNorm either
side of a dropout is a genuinely different operation, and two independent masks
are a different one again, so there is no seed at which they agree.

The parity section also runs at `feature_size = d_model`, which the listener no
longer requires and the legacy module did: `Receiver` brings the backbone's
output to each slot's declared width through an interface of its own, so the two
widths are independent now. Holding them equal is what lets the legacy module's
`bilinear` be loaded into the new one at all.
"""

import math

import pytest
import torch
import torch.nn as nn

import _bootstrap
from _bootstrap import build_listener, config_section, rung

from models import receiver as R
import parse_config


# The two readout scalars, asked for rather than inherited. DEFAULT.toml turned
#     them off on 2026-09-11 with `loss = "hinge"`, which made a volume in front
#     of a fixed margin degenerate with the margin. The four tests below are
#     about those parameters existing, so they build the arrangement that has
#     them; see test_score_scale.py's `READOUT_ON`, which is this constant.
READOUT_ON = {"scale_score": True, "bias_score": True}

REFERENT_DIM = 320
BATCH, N_OBJ, SEQ = 6, 10, 7

# Deliberately unlike every other width in these tests, so
#     `test_no_discriminator_reads_the_token_embedding` cannot pass by
#     coincidence.
TOKEN_DIM = 37

CROSS_RUNG = "17_shapeworld_receiver_cross_attention_lm.toml"


def _inputs(listener, seed=0):
    generator = torch.Generator().manual_seed(seed)
    referents = torch.randn(
        BATCH, N_OBJ, listener.feature_size, generator=generator
    )
    messages = torch.randn(
        BATCH,
        getattr(listener, "message_length", SEQ),
        listener.token_embedding_size,
        generator=generator,
    )
    return referents, messages


# --------------------------------------------------------------------------
# The safety net: the historical pairing, pinned bit for bit.
# --------------------------------------------------------------------------

class LegacyBilinearGRUComparer(nn.Module):
    """
    `BilinearGRUComparer` as it stood before the split, arithmetic verbatim,
        kept here and nowhere else.

    A copy rather than an import because the original is gone: this exists to
        pin the refactor, not to be maintained. If a deliberate change to the
        bilinear path makes the test below fail, the fix is to record the
        change here and say why -- not to delete the test, which is the only
        thing standing between the new plumbing and a silent regression.
    """

    def __init__(self, referent_dim, token_dim, d_model):
        super().__init__()
        self.gru = nn.GRU(
            token_dim, d_model, num_layers=1, bias=True, batch_first=True,
            dropout=0.0, bidirectional=False,
        )
        self.bilinear = nn.Linear(d_model, referent_dim, bias=False)
        self.referent_layer_norm = nn.LayerNorm(
            referent_dim, elementwise_affine=False, eps=R.LAYER_NORM_EPS
        )
        self.message_layer_norm = nn.LayerNorm(
            referent_dim, elementwise_affine=False, eps=R.LAYER_NORM_EPS
        )
        self.log_score_scale = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(p=0.0)
        self.referent_dim = referent_dim

    def forward(self, referents, messages):
        token_embeddings, _ = self.gru(messages)
        message_embeddings = token_embeddings[:, -1, ...]

        projected = self.bilinear(message_embeddings)
        projected = (
            self.message_layer_norm(projected) * self.log_score_scale.exp()
        )

        referents = self.referent_layer_norm(referents)
        referents = self.dropout(referents)

        scores = torch.einsum("ijh,ih->ij", (referents, projected))
        return scores / math.sqrt(self.referent_dim)


def _legacy_pair(d_model=128, token_dim=TOKEN_DIM):
    """
    At one width throughout. `BilinearDiscriminator` compares at
        `[receiver_language_model] d_model` and the backbone feeds it through an
        interface, so the comparison width and the backbone width are separate
        numbers now; the legacy module had only one. Building both at `d_model`
        is what makes its `bilinear` loadable into the new one.
    """
    torch.manual_seed(11)
    legacy = LegacyBilinearGRUComparer(d_model, token_dim, d_model).eval()

    listener = build_listener(
        "ReceiverGRULM",
        "BilinearDiscriminator",
        d_model,
        language_model_overrides=dict(
            token_embedding_size=token_dim,
            d_model=d_model,
            layers=1,
            bidirectional=False,
        ),
        discriminator_overrides=dict(READOUT_ON),
    ).eval()

    # The two halves of the old module, moved and not rewritten.
    listener.language_model.gru.load_state_dict(legacy.gru.state_dict())
    listener.discriminator.bilinear.load_state_dict(
        legacy.bilinear.state_dict()
    )
    return legacy, listener


def test_the_gru_slot_still_reproduces_the_module_it_replaced():
    """
    The half of the parity that is still exact, and the half this test was
        really protecting: the message readout. Both paths run the same GRU
        over the same message and take timestep -1, so any difference here is
        a difference in the plumbing rather than in floating point.

    `ReceiverGRULM` returns `(batch, 1, d_model)` and `BilinearDiscriminator`
        now indexes the last slot rather than meaning over them, which for one
        slot is the same tensor -- so the change of readout does not reach this
        arm at all. It reaches `ReceiverCrossAttentionLM`, which returns one
        slot per symbol; see `test_the_bilinear_readout_takes_the_eos_slot`.
    """
    legacy, listener = _legacy_pair()
    referents, messages = _inputs(listener)

    with torch.no_grad():
        legacy_readout = legacy.gru(messages)[0][:, -1, ...]
        slot_readout = listener.language_model(
            messages, listener.deliver(R.LANGUAGE_MODEL_REFERENTS, referents)
        )

    assert slot_readout.shape[1] == 1
    assert torch.equal(legacy_readout, slot_readout[:, -1, :])


@pytest.mark.parametrize("d_model", [64, 128, 256])
def test_the_gru_reproduction_is_not_an_artefact_of_one_width(d_model):
    legacy, listener = _legacy_pair(d_model=d_model)
    referents, messages = _inputs(listener, seed=d_model)

    with torch.no_grad():
        legacy_readout = legacy.gru(messages)[0][:, -1, ...]
        slot_readout = listener.language_model(
            messages, listener.deliver(R.LANGUAGE_MODEL_REFERENTS, referents)
        )

    assert torch.equal(legacy_readout, slot_readout[:, -1, :])


def test_the_score_deliberately_no_longer_matches_the_legacy_module():
    """
    The recorded divergence. `LegacyBilinearGRUComparer` is a frozen snapshot
        and its docstring says a deliberate change to the bilinear path should
        be written down here rather than patched into the copy, so: **the
        scores no longer match, in exactly one place, on purpose.**

    The divergence is in four places, all in the input path or the readout, and
        the legacy module still has the GRU half bit-identical.

    One: the ordering. The legacy path is `message_layer_norm(bilinear(m))`,
        which pins the projected message to unit variance; the message now
        reaches `bilinear` through `Receiver`'s message interface, which is a
        learned `nn.Linear` *and* a norm, so the norm sets where `bilinear`
        starts rather than where it ends and there is a projection in between
        that the legacy module has no counterpart for.

    Two: the referents are layer-normed before the dot product, so no candidate
        is read loudly for being large. The legacy module compares them raw.

    Three: that norm is one third of a `LinearInterface` too, so the referents
        also pass through a learned projection that the legacy module does not
        have. Both interfaces are new since the hoist; before it, the norms sat
        inside `BilinearDiscriminator` with no projection of their own.

    Four: `score_scale`. The legacy module holds its volume in the product of
        the backbone's magnitude and its weight, the way jayelm's unnormalised
        `compare` does; this one takes both operands normalised, keeps the exact
        `/sqrt(referent_embedding_size)` that makes the opening `1/sqrt(3)` at
        any width, and puts the volume in one scalar on top. See
        test_score_scale.py.

    Everything else about the pairing is unchanged, which is what the two tests
        above still pin. This one exists so the divergence cannot widen
        silently: it asserts the scores differ, and that they differ *only*
        through those four, by rebuilding the new arithmetic out of the
        listener's own parts.
    """
    legacy, listener = _legacy_pair()
    referents, messages = _inputs(listener)
    discriminator = listener.discriminator

    with torch.no_grad():
        assert not torch.allclose(
            legacy(referents, messages),
            listener(referents, messages),
            atol=1e-6,
        )

        # The new input path and readout, by hand, from the listener's own
        #     tensors -- the interfaces included, which is where three of the
        #     four differences now live.
        encoded = listener.language_model(
            messages, listener.deliver(R.LANGUAGE_MODEL_REFERENTS, referents)
        )
        readout = listener.deliver(R.DISCRIMINATOR_MESSAGE, encoded)[:, -1, :]
        projected = discriminator.bilinear(readout)
        normed = listener.deliver(R.DISCRIMINATOR_REFERENTS, referents)
        raw = torch.einsum("ijh,ih->ij", (normed, projected))
        rebuilt = discriminator.score_scale * (
            raw / math.sqrt(discriminator.referent_embedding_size)
        )

    assert torch.allclose(
        rebuilt, listener(referents, messages), atol=1e-6
    )


def test_the_default_gru_is_jayelms():
    """
    `DEFAULT.toml`'s listener GRU is 1 layer unidirectional at 1024 wide --
        jayelm's, and the baseline rung 1 is meant to reproduce.

    This has been both things. It carried 2 layers bidirectional for a while,
        for parameter parity with the transformer arm *at a shared width of
        256*; because nothing in the ladder set both widths, every rung up to 14
        inherited those keys at 1024 and got a 28.3M listener encoder instead of
        a 4.7M one. Parity is now bought at jayelm's width by deepening the
        transformer arm -- see `test_the_two_listener_arms_are_parameter_matched`
        -- and this test is here so the default cannot drift back silently.
    """
    settings = config_section("receiver_language_model")
    assert settings["layers"] == 1
    assert settings["bidirectional"] is False

    built = build_listener("ReceiverGRULM", "BilinearDiscriminator", REFERENT_DIM)

    assert built.language_model.gru.num_layers == 1
    assert built.language_model.gru.bidirectional is False
    assert built.language_model.output_size == built.language_model.d_model


@pytest.mark.parametrize(
    "config_file",
    [
        "17_shapeworld_receiver_cross_attention_lm.toml",
        "18_birds_receiver_cross_attention_lm.toml",
    ],
)
def test_the_two_listener_arms_are_parameter_matched(config_file):
    """
    Parity is a property of the pair of configs, so assert it as one: build the
        default GRU and the rung's transformer and compare the counts.

    4,687,872 against 4,702,646, which is +0.3%. Note 2 layers bidirectional
        would be 2.5x one layer's parameters and not 2x -- the second layer's
        input is the first's concatenated output, so its `weight_ih` is double
        -- which is the arithmetic that made the shared-256 scheme look cheaper
        than it was at 1024.

    **Measured over the slot alone, and the hoist is what makes that the right
        boundary.** It was +2.1% before, and the gap was almost entirely one
        thing: `ReceiverCrossAttentionLM` owned a `referent_adapter` and
        `ReceiverGRULM` had nothing corresponding, so the ladder was comparing
        an encoder against an encoder-plus-a-projection. That projection is
        `Receiver`'s interface now. Counting the interfaces back in would put
        the backbone's `final_feat_dim` -- a property of the vision model, and a
        different number on the two arms of the ladder -- into a comparison
        about message encoders, which is the confound the count is here to
        exclude. So: slots only.

    Parameter parity is not interface parity: `output_size` is 1024 on the GRU
        against 256 on the transformer, so the message interfaces and the
        discriminators downstream of them differ.
    """
    gru = build_listener(
        "ReceiverGRULM", "BilinearDiscriminator", REFERENT_DIM
    ).language_model
    cross = build_listener(
        "ReceiverCrossAttentionLM", "BilinearDiscriminator", REFERENT_DIM,
        config_file=rung(config_file),
    ).language_model

    n_gru = sum(p.numel() for p in gru.parameters())
    n_cross = sum(p.numel() for p in cross.parameters())

    assert n_gru == 4_687_872
    assert n_cross == 4_702_646
    assert abs(n_cross / n_gru - 1.0) < 0.05

    assert gru.output_size == 1024
    assert cross.output_size == 256


@pytest.mark.parametrize(
    "config_file,language_model",
    [
        ("11_shapeworld_receiver_transformer_autoregressive_lm.toml",
         "ReceiverTransformerAutoregressiveLM"),
        ("12_birds_receiver_transformer_autoregressive_lm.toml",
         "ReceiverTransformerAutoregressiveLM"),
        ("13_shapeworld_receiver_transformer_bidirectional_lm.toml",
         "ReceiverTransformerBidirectionalLM"),
        ("14_birds_receiver_transformer_bidirectional_lm.toml",
         "ReceiverTransformerBidirectionalLM"),
    ],
)
def test_the_transformer_encoders_are_parameter_matched_to_the_gru(
    config_file, language_model
):
    """
    The speaker language model's depth and feedforward -- 7 blocks at
        `ff_inner_size = 512` -- at the listener's width of 256: 4,673,344 for
        encoder, message adapter and `SequencePool` against the GRU's 4,687,872,
        0.997x. Identical on both arms, the mask being the only difference.
    """
    gru = build_listener(
        "ReceiverGRULM", "BilinearDiscriminator", REFERENT_DIM
    ).language_model
    encoder = build_listener(
        language_model, "BilinearDiscriminator", REFERENT_DIM,
        config_file=rung(config_file),
    ).language_model

    n_gru = sum(p.numel() for p in gru.parameters())
    n_encoder = sum(p.numel() for p in encoder.parameters())

    assert n_encoder == 4_673_344
    assert abs(n_encoder / n_gru - 1.0) < 0.05
    assert encoder.output_size == 256


# --------------------------------------------------------------------------
# The slot contract.
# --------------------------------------------------------------------------

ALL_FOUR = pytest.mark.parametrize(
    "language_model,discriminator",
    [
        ("ReceiverGRULM", "BilinearDiscriminator"),
        ("ReceiverTransformerAutoregressiveLM", "BilinearDiscriminator"),
        ("ReceiverTransformerBidirectionalLM", "BilinearDiscriminator"),
        ("ReceiverCrossAttentionLM", "BilinearDiscriminator"),
    ],
    ids=["gru", "causal", "bidirectional", "cross"],
)

# The language models that read the message alone. Only the cross-attention
#     encoder declares a referent width.
READS_THE_MESSAGE_ALONE = (
    "ReceiverGRULM",
    "ReceiverTransformerAutoregressiveLM",
    "ReceiverTransformerBidirectionalLM",
)


def _four_cell(language_model, discriminator, **kwargs):
    """
    Every cell from rung 17, which states widths every slot can build:
        DEFAULT's `[receiver_language_model] d_model = 1024` does not divide its
        `heads = 5`, and that key is the GRU's.
    """
    return build_listener(
        language_model, discriminator, REFERENT_DIM,
        config_file=rung(CROSS_RUNG), **kwargs
    )


@ALL_FOUR
def test_every_pairing_scores_every_candidate(language_model, discriminator):
    listener = _four_cell(language_model, discriminator).eval()
    referents, messages = _inputs(listener)

    with torch.no_grad():
        scores = listener(referents, messages)

    assert scores.shape == (BATCH, N_OBJ)
    assert torch.isfinite(scores).all()


@ALL_FOUR
def test_every_pairing_scores_on_the_message(language_model, discriminator):
    """
    The guard on everything else here. A listener that ignored the message
        would satisfy most of the properties below and answer no question at
        all.
    """
    listener = _four_cell(language_model, discriminator).eval()
    referents, messages = _inputs(listener)

    with torch.no_grad():
        before = listener(referents, messages)
        after = listener(referents, torch.randn_like(messages))

    assert not torch.allclose(before, after, atol=1e-6)


@ALL_FOUR
def test_the_language_model_returns_a_sequence(language_model, discriminator):
    """
    `(batch, slots, output_size)` from every one, so the discriminator reads
        one shape. The GRU returns its final state and the Transformer encoders
        their pooled vector, each as a length-1 sequence; the cross-attention
        stack returns one slot per symbol.
    """
    listener = _four_cell(language_model, discriminator).eval()
    referents, messages = _inputs(listener)

    with torch.no_grad():
        representation = listener.language_model(
            messages, listener.deliver(R.LANGUAGE_MODEL_REFERENTS, referents)
        )

    assert representation.ndim == 3
    assert representation.shape[0] == BATCH
    assert representation.shape[-1] == listener.language_model.output_size

    expected_slots = 1 if language_model in READS_THE_MESSAGE_ALONE else SEQ
    assert representation.shape[1] == expected_slots


@ALL_FOUR
def test_the_discriminator_is_sized_from_the_language_model(
    language_model, discriminator
):
    """
    Not from a config key restating the width. `2 * d_model` for a
        bidirectional GRU and `d_model` for the Transformer stacks, and no
        arithmetic makes those agree, so a key would only ever be a key that
        could be wrong.
    """
    listener = _four_cell(language_model, discriminator)
    width = listener.language_model.output_size

    # The message interface is what reads the encoder's output now, so this is
    #     the one place the sizing shows. What the discriminator itself asks for
    #     is `message_input_size`, and the interface bridges the two.
    interface = listener.interfaces[R.DISCRIMINATOR_MESSAGE]
    assert interface.adapter.in_features == width
    assert interface.output_size == listener.discriminator.message_input_size

    # The discriminator declares the encoder's own width, so the interface is
    #     square and `bilinear` still reads at `output_size`.
    assert listener.discriminator.message_input_size == width
    assert listener.discriminator.bilinear.in_features == width


@ALL_FOUR
def test_only_the_cross_attention_slot_reads_the_candidate_set(
    language_model, discriminator
):
    """
    Half of the uniform signature's cost, and it is paid deliberately: the GRU
        and the Transformer encoders take `referents` and do nothing with it,
        because dispatching on class at the call site would be worse. The
        cross-attention encoder reads them, which is its entire point -- and
        the reason it is the top rung, since a set summary lets it score
        "which cluster" without the message.
    """
    listener = _four_cell(language_model, discriminator).eval()
    referents, messages = _inputs(listener)

    perturbed = referents.clone()
    perturbed[:, 0, :] += 5.0

    with torch.no_grad():
        before = listener.language_model(
            messages, listener.deliver(R.LANGUAGE_MODEL_REFERENTS, referents)
        )
        after = listener.language_model(
            messages, listener.deliver(R.LANGUAGE_MODEL_REFERENTS, perturbed)
        )

    if language_model in READS_THE_MESSAGE_ALONE:
        # And structurally, not just numerically: this slot declares no
        #     referent width, so there is no interface and `Receiver` hands it
        #     `None`. There is no tensor for it to ignore.
        assert R.LANGUAGE_MODEL_REFERENTS not in listener.interfaces
        assert listener.language_model.referent_input_size is None
        assert torch.equal(before, after)
    else:
        assert not torch.allclose(before, after, atol=1e-6)


@ALL_FOUR
def test_each_referent_interface_draws_its_own_mask(
    language_model, discriminator
):
    """
    One mask per referent interface, drawn independently, at the rate
        `[receiver] dropout` names.

    This assertion used to be its opposite: `Receiver` masked once and handed
        the same tensor to both slots, and this test pinned the equality. The
        argument for that was that a per-slot mask would regularise the listener
        at a rate no config key names -- and it was a good argument about
        masking *one* tensor twice, which is not what this is. Since the hoist
        each slot reads its own projected copy of the referents, and each copy
        is masked once at the documented rate. So what is checked here is that
        the rate holds on every interface and that the draws are independent;
        the two slots' inputs are not required to be equal, and on the pairings
        that have two referent interfaces they must not be.

    Checked through `Receiver` itself rather than through the test shim, since
        the whole claim is about where the masks live.
    """
    listener = _four_cell(language_model, discriminator, dropout=0.5)
    receiver = R.Receiver(
        # The identity in place of a backbone: this test is about the masks, and
        #     a real backbone would only put a stage upstream of them.
        nn.Identity(),
        REFERENT_DIM,
        nn.Embedding(8, listener.token_embedding_size),
        listener.language_model,
        listener.discriminator,
        dropout=0.5,
    ).train()

    seen = {}
    handles = [
        receiver.language_model.register_forward_pre_hook(
            lambda _module, args: seen.__setitem__("language_model", args[1])
        ),
        receiver.discriminator.register_forward_pre_hook(
            lambda _module, args: seen.__setitem__("discriminator", args[0])
        ),
    ]

    referents = torch.randn(BATCH, N_OBJ, REFERENT_DIM)
    messages = torch.randn(
        BATCH,
        getattr(listener, "message_length", SEQ),
        listener.token_embedding_size,
    )
    # `Receiver` embeds the message with `messages @ token_embedding.weight`,
    #     so the "message" it wants is one-hot-shaped. The shim sidesteps that;
    #     here it cannot.
    receiver.token_embedding = nn.Embedding(
        listener.token_embedding_size, listener.token_embedding_size
    )
    with torch.no_grad():
        receiver.token_embedding.weight.copy_(
            torch.eye(listener.token_embedding_size)
        )

    torch.manual_seed(3)
    receiver(referents, messages)

    for handle in handles:
        handle.remove()

    masked = [
        tensor for tensor in seen.values() if tensor is not None
    ]
    # One per referent interface: only the cross-attention encoder declares a
    #     referent width, so on the other pairings the language model is handed
    #     `None` and there is one mask rather than two.
    expected = 1 if language_model in READS_THE_MESSAGE_ALONE else 2
    assert len(masked) == expected
    assert (seen["language_model"] is None) == (expected == 1)

    for tensor in masked:
        # The rate, on every one of them. 0.5 over 20 candidates' worth of
        #     features is far enough from 0 and 1 that a loose band is still a
        #     real check.
        share = (tensor == 0.0).float().mean().item()
        assert 0.3 < share < 0.7, share

    if expected == 2:
        # Independent draws, which is what makes them the two masks the rate
        #     describes rather than a correlated pair. The widths may differ, so
        #     compare the patterns over the axes they share.
        first = (seen["language_model"] == 0.0).float().mean(-1)
        second = (seen["discriminator"] == 0.0).float().mean(-1)
        assert not torch.equal(first, second)


@ALL_FOUR
def test_the_mask_removes_features_and_not_candidates(
    language_model, discriminator
):
    """
    Element-wise over `(batch, n_objects, features)`, on every referent
        interface. A mask that removed whole candidates would leak the label
        ordering, which is the first half of the tensor.

    The one property of the mask that the hoist must not be allowed to change,
        so it is asserted per interface rather than once on a dropout that
        `Receiver` no longer owns.
    """
    listener = _four_cell(language_model, discriminator, dropout=0.5).train()

    referent_interfaces = [
        (name, interface)
        for name, interface in listener.interfaces.items()
        if name.endswith("referents")
    ]
    assert referent_interfaces

    for name, interface in referent_interfaces:
        referents = torch.ones(BATCH, N_OBJ, interface.output_size)

        torch.manual_seed(5)
        masked = interface.dropout(referents)

        surviving = (masked != 0.0).float().mean(-1)
        assert (surviving > 0.0).all(), f"{name} dropped a whole candidate"
        assert (surviving < 1.0).all(), f"{name} masked no candidate at all"


@ALL_FOUR
def test_no_discriminator_reads_the_token_embedding(
    language_model, discriminator
):
    """
    The one-encoder invariant, checked structurally. The discriminator reads
        whatever the language model produced, and nothing in it may be sized
        from `token_embedding_size`, which is what a second encoder would need.
    """
    listener = build_listener(
        language_model,
        discriminator,
        REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
        language_model_overrides=dict(token_embedding_size=TOKEN_DIM),
    )

    assert not any(
        isinstance(module, nn.GRU)
        for module in listener.discriminator.modules()
    )
    assert not any(
        TOKEN_DIM in tuple(parameter.shape)
        for parameter in listener.discriminator.parameters()
    ), f"{discriminator} is sized from the token embedding"


# --------------------------------------------------------------------------
# The Transformer encoders.
# --------------------------------------------------------------------------

TRANSFORMER_ENCODERS = pytest.mark.parametrize(
    "language_model",
    ["ReceiverTransformerAutoregressiveLM", "ReceiverTransformerBidirectionalLM"],
    ids=["causal", "bidirectional"],
)


def _encoder(language_model):
    return build_listener(
        language_model, "BilinearDiscriminator", REFERENT_DIM,
        config_file=rung(
            "11_shapeworld_receiver_transformer_autoregressive_lm.toml"
        ),
    ).eval()


@TRANSFORMER_ENCODERS
def test_the_transformer_encoders_never_see_the_candidates(language_model):
    """
    The point of the class. Like the GRU, and unlike the cross-attention
        encoder, it declares no referent width, so `Receiver` builds it no
        interface and the encoding of a message is the same whatever it is
        compared against.

    Two different candidate sets with the same message, through the listener's
        own delivery and then handed in directly: one pooled vector, identical
        both times.
    """
    listener = _encoder(language_model)
    encoder = listener.language_model
    referents, messages = _inputs(listener)
    others = torch.randn_like(referents) * 3.0 + 1.0

    assert encoder.referent_input_size is None
    assert encoder.message_input_size is None
    assert encoder.output_size == encoder.d_model
    assert R.LANGUAGE_MODEL_REFERENTS not in listener.interfaces

    with torch.no_grad():
        first = encoder(
            messages, listener.deliver(R.LANGUAGE_MODEL_REFERENTS, referents)
        )
        second = encoder(
            messages, listener.deliver(R.LANGUAGE_MODEL_REFERENTS, others)
        )
        handed_in = encoder(messages, others)

    assert first.shape == (BATCH, 1, encoder.d_model)
    assert torch.equal(first, second)
    assert torch.equal(first, handed_in)


@TRANSFORMER_ENCODERS
def test_the_transformer_encoders_pool_to_one_slot(language_model):
    """
    A `SequencePool` over every position, returned as a length-1 sequence, so
        `BilinearDiscriminator`'s last-slot read is the identity here as it is
        for the GRU -- and not the last position of the stack.
    """
    encoder = _encoder(language_model).language_model
    _, messages = _inputs(_encoder(language_model))

    with torch.no_grad():
        positions = encoder.encode(messages)
        pooled = encoder(messages, None)

    assert positions.shape == (BATCH, SEQ, encoder.d_model)
    assert torch.allclose(pooled[:, 0, :], encoder.pool(positions), atol=1e-6)
    assert not torch.allclose(pooled[:, 0, :], positions[:, -1, :], atol=1e-4)


@pytest.mark.parametrize("changed", [2, 4, SEQ - 1])
def test_the_causal_encoder_reads_only_prefixes(changed):
    """
    The causal arm's mask, checked where it acts: before the pool, a position's
        output depends on that position and the ones before it, so changing a
        later token moves nothing earlier. The unmasked arm, built from the same
        seed, is the control -- the same change reaches every position there.
    """
    causal = _encoder("ReceiverTransformerAutoregressiveLM").language_model
    unmasked = _encoder("ReceiverTransformerBidirectionalLM").language_model
    _, messages = _inputs(_encoder("ReceiverTransformerAutoregressiveLM"))

    edited = messages.clone()
    edited[:, changed, :] = torch.randn_like(edited[:, changed, :])

    with torch.no_grad():
        before, after = causal.encode(messages), causal.encode(edited)
        open_before, open_after = unmasked.encode(messages), unmasked.encode(edited)

    assert torch.allclose(
        before[:, :changed, :], after[:, :changed, :], atol=1e-6
    )
    assert not torch.allclose(
        before[:, changed:, :], after[:, changed:, :], atol=1e-4
    )
    assert not torch.allclose(
        open_before[:, :changed, :], open_after[:, :changed, :], atol=1e-4
    )


@TRANSFORMER_ENCODERS
def test_the_class_chooses_the_encoders_arm(language_model):
    """
    The arm is the class, as on the speaker: `bidirectional` in the config is
        overridden, whichever way it points. Rung 17 states `true` for the
        cross-attention encoder, which is the case worth checking.
    """
    built = build_listener(
        language_model, "BilinearDiscriminator", REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
    ).language_model

    assert built.bidirectional is (
        language_model == "ReceiverTransformerBidirectionalLM"
    )


def test_the_shared_transformer_encoder_is_not_selectable():
    """
    `ReceiverTransformerLM` is the implementation both arms share, and a config
        naming it would take one rate for two arms. `validate_config` refuses it
        and names the subclasses, as it does `SenderTransformerLM`.
    """
    config = parse_config.get_config()
    config["receiver"]["language_model"] = "ReceiverTransformerLM"

    with pytest.raises(
        parse_config.InvalidConfig, match="ReceiverTransformerBidirectionalLM"
    ):
        parse_config.validate_config(config)


def test_the_bilinear_readout_takes_the_eos_slot():
    """
    The message readout, and the one place the change of readout bites.

    `BilinearDiscriminator` used to mean over slots. `ReceiverGRULM` returns
        one, so that was the identity there; `ReceiverCrossAttentionLM` returns
        one per message position, so meaning diluted the readout across every
        symbol. It now takes the last slot, which is the speaker's reserved EOS
        position -- fixed-length messages make that positionally determined and
        so a constant learned vector, which is a CLS query in all but name, and
        a causal stack reaches it having read the whole message.
    """
    listener = build_listener(
        "ReceiverCrossAttentionLM", "BilinearDiscriminator", REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
    ).eval()
    referents, messages = _inputs(listener)

    with torch.no_grad():
        slots = listener.deliver(
            R.DISCRIMINATOR_MESSAGE,
            listener.language_model(
                messages,
                listener.deliver(R.LANGUAGE_MODEL_REFERENTS, referents),
            ),
        )
        adapted = listener.deliver(R.DISCRIMINATOR_REFERENTS, referents)

    # One slot per message position, so the readout is a real choice here.
    assert slots.shape[1] == listener.message_length
    assert not torch.allclose(slots[:, -1, :], slots.mean(1), atol=1e-5)

    bilinear = listener.discriminator
    with torch.no_grad():
        taken = bilinear(adapted, slots)
        from_eos = bilinear(adapted, slots[:, -1:, :])
        from_mean = bilinear(adapted, slots.mean(1, keepdim=True))

    assert torch.allclose(taken, from_eos, atol=1e-6)
    assert not torch.allclose(taken, from_mean, atol=1e-5)


def _standardise(scores):
    """
    The per-game standardisation `receiver.standardise` did, kept here as the
        arithmetic behind the test below. The helper itself went on 2026-10-01
        with the attention discriminator, its last caller.
    """
    centred = scores - scores.mean(1, keepdim=True)
    spread = centred.std(dim=1, keepdim=True, unbiased=False)
    return centred / spread.clamp(min=1e-6)


def test_a_scale_on_a_standardised_path_would_have_been_inert():
    """
    Why the readout no longer standardises, kept as the measurement behind
        `485b38e` rather than as a live justification.

    `standardise` subtracts a mean and divides by a spread, both homogeneous of
        degree one in a positive multiplier, so anything in front of it that
        only sets magnitude cancels -- a scalar exactly, and a weight matrix in
        its radial component. That is what made a whole class of parameters
        unable to learn their own magnitude, and it is the arithmetic worth
        keeping pinned even though nothing in the forward path does it now.

    Exactly, in arithmetic; to float32 rounding, in fact -- which is why the
        comparison below is `allclose` and the gradient one is against a
        tolerance rather than against zero. A gradient at 1e-8 is not a
        parameter that learns.
    """
    scores = torch.randn(BATCH, N_OBJ, generator=torch.Generator().manual_seed(1))

    for scale in (0.01, 1.0, 37.0):
        assert torch.allclose(
            _standardise(scale * scores), _standardise(scores), atol=1e-6
        )

    # The same claim on the module itself, which is where it would have bitten,
    #     and stated as a ratio: what the scale gets through `standardise`
    #     against what the identical module gets without it.
    referents = torch.randn(BATCH, N_OBJ, REFERENT_DIM)
    message_repr = torch.randn(BATCH, 1, 64)

    gradients = {}
    for name, wrap in (("standardised", _standardise), ("raw", lambda x: x)):
        torch.manual_seed(7)
        discriminator = R.BilinearDiscriminator(REFERENT_DIM, 64)
        wrap(discriminator(referents, message_repr)).pow(2).sum().backward()
        # The weight's *radial* component -- how much of its gradient wants it
        #     longer rather than turned. That is the part a volume lives in,
        #     and the part `standardise` removes.
        weight = discriminator.bilinear.weight
        direction = weight.detach() / weight.detach().norm()
        gradients[name] = abs((weight.grad * direction).sum().item())

    assert gradients["raw"] > 0.0
    assert gradients["standardised"] < 1e-3 * gradients["raw"]


def test_the_pair_can_still_go_quiet():
    """
    The freedom that is deliberately left open. A listener that has nothing to
        say must be able to say it quietly, or it is committing before the
        message carries anything, which is what took out the fixed-gain
        readout.

    It is `ScoreVolume.log_score_scale` -- one cheap knob on an elevated
        learning rate. Being cheap was the objection, on the grounds that the same
        scalar multiplied the gradient going back to the speaker; that objection
        did not survive AdamW's `m / sqrt(v)` and per-submodule clipping, both
        of which divide a uniform factor out. See test_score_scale.py.

    """
    listener = build_listener(
        "ReceiverCrossAttentionLM", "BilinearDiscriminator", REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
        discriminator_overrides=dict(READOUT_ON),
    ).eval()
    discriminator = listener.discriminator
    referents, messages = _inputs(listener)

    with torch.no_grad():
        loud = listener(referents, messages)
        discriminator.log_score_scale.fill_(math.log(0.01))
        quiet = listener(referents, messages)

    assert quiet.std().item() < 0.05 * loud.std().item()


# --------------------------------------------------------------------------
# Resetting.
# --------------------------------------------------------------------------

@ALL_FOUR
def test_reset_parameters_leaves_nothing_trained(language_model, discriminator):
    """
    Every submodule holding a parameter, the scalars included. A reset that
        skipped one would restart a run with the previous one's opinion
        surviving in it. See docs/anecdotes.md.
    """
    listener = _four_cell(language_model, discriminator)
    before = {
        name: parameter.detach().clone()
        for name, parameter in listener.named_parameters()
    }

    for parameter in listener.parameters():
        with torch.no_grad():
            parameter.add_(1.0)

    # Through the shim's own `reset_parameters`, which mirrors `Receiver`'s:
    #     the interfaces are the listener's parameters too, and leaving them out
    #     is exactly the bug docs/anecdotes.md records.
    listener.reset_parameters()

    # broccoli owns these and does not re-draw them, which is correct for both:
    #     `rotary_embedding.freqs` is a deterministic function of position, so
    #     there is nothing to draw, and `swish_beta` is the activation's own
    #     parameter rather than the block's. Excluded by name rather than by
    #     loosening the assertion, so that a *new* untouched parameter still
    #     fails.
    BROCCOLI_INERT = ("rotary_embedding.freqs", "swish_beta")
    unchanged = [
        name
        for name, parameter in listener.named_parameters()
        if torch.equal(parameter, before[name] + 1.0)
        and not name.endswith(BROCCOLI_INERT)
    ]
    assert not unchanged, f"reset_parameters missed {unchanged}"


def test_resetting_returns_the_readout_to_its_opening():
    """
    The offset and the volume reset through `ScoreVolume.reset_score_volume`.
    """
    listener = build_listener(
        "ReceiverCrossAttentionLM", "BilinearDiscriminator", REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
        discriminator_overrides=dict(READOUT_ON),
    )
    discriminator = listener.discriminator

    with torch.no_grad():
        discriminator.score_bias.fill_(1.0)
        discriminator.log_score_scale.fill_(2.0)

    discriminator.reset_parameters()

    assert discriminator.score_bias.item() == 0.0
    assert discriminator.score_scale.item() == pytest.approx(1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

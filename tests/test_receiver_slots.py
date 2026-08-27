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

The slots make that a 2x2:

                             BilinearDiscriminator  AttentionDiscriminator
    ReceiverGRULM             the historical arm            new
    ReceiverCrossAttentionLM           new            the attention arm

The first section below is the safety net for the refactor: at `dropout = 0` and
one unidirectional layer, `ReceiverGRULM + BilinearDiscriminator` must reproduce
the pre-split module bit for bit. It is the only pairing that can be pinned that
exactly -- the other three did not exist -- so everything else here is a
property test.

It has to be at `dropout = 0` because the mask moved. `Receiver` now applies it
once to the raw referent embeddings and hands the same masked tensor to both
slots, where both of the modules it replaced applied it *after* their own norm.
A LayerNorm following dropout renormalises the corrupted vector, so the two are
genuinely different operations and there is no seed at which they agree.
"""

import math

import pytest
import torch
import torch.nn as nn

import _bootstrap
from _bootstrap import build_listener, config_section, rung

from models import receiver as R


REFERENT_DIM = 320
BATCH, N_OBJ, SEQ = 6, 10, 7

# Deliberately unlike every other width in these tests, so
#     `test_no_discriminator_reads_the_token_embedding` cannot pass by
#     coincidence.
TOKEN_DIM = 37

CROSS_RUNG = "15_shapeworld_receiver_cross_attention_lm.toml"


def _inputs(listener, seed=0):
    generator = torch.Generator().manual_seed(seed)
    referents = torch.randn(
        BATCH, N_OBJ, listener.referent_embedding_size, generator=generator
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


def _legacy_pair(d_model=128, referent_dim=REFERENT_DIM, token_dim=TOKEN_DIM):
    torch.manual_seed(11)
    legacy = LegacyBilinearGRUComparer(referent_dim, token_dim, d_model).eval()

    listener = build_listener(
        "ReceiverGRULM",
        "BilinearDiscriminator",
        referent_dim,
        language_model_overrides=dict(
            token_embedding_size=token_dim,
            d_model=d_model,
            layers=1,
            bidirectional=False,
        ),
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
        slot_readout = listener.language_model(messages, referents)

    assert slot_readout.shape[1] == 1
    assert torch.equal(legacy_readout, slot_readout[:, -1, :])


@pytest.mark.parametrize("d_model", [64, 128, 256])
def test_the_gru_reproduction_is_not_an_artefact_of_one_width(d_model):
    legacy, listener = _legacy_pair(d_model=d_model)
    referents, messages = _inputs(listener, seed=d_model)

    with torch.no_grad():
        legacy_readout = legacy.gru(messages)[0][:, -1, ...]
        slot_readout = listener.language_model(messages, referents)

    assert torch.equal(legacy_readout, slot_readout[:, -1, :])


def test_the_score_deliberately_no_longer_matches_the_legacy_module():
    """
    The recorded divergence. `LegacyBilinearGRUComparer` is a frozen snapshot
        and its docstring says a deliberate change to the bilinear path should
        be written down here rather than patched into the copy, so: **the
        scores no longer match, in exactly one place, on purpose.**

    The divergence is in three places, all in the readout, and the legacy
        module still has the GRU half bit-identical.

    One: the ordering. The legacy path is `message_layer_norm(bilinear(m))`,
        which pins the projected message to unit variance;
        `BilinearDiscriminator` runs `bilinear(message_layer_norm(m))`, so the
        norm sets where `bilinear` starts rather than where it ends.

    Two: the referents are layer-normed before the dot product, so no candidate
        is read loudly for being large. The legacy module compares them raw.

    Three: `score_scale`. The legacy module holds its volume in the product of
        the backbone's magnitude and its weight, the way jayelm's unnormalised
        `compare` does; this one normalises both operands, keeps the exact
        `/sqrt(referent_embedding_size)` that makes the opening `1/sqrt(3)` at
        any width, and puts the volume in one scalar on top. See
        test_score_scale.py.

    Everything else about the pairing is unchanged, which is what the two tests
        above still pin. This one exists so the divergence cannot widen
        silently: it asserts the scores differ, and that they differ *only*
        through those three, by rebuilding the new arithmetic out of the
        module's own parts.
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

        # The new ordering and readout, by hand, from the module's own tensors.
        readout = listener.language_model(messages, referents)[:, -1, :]
        projected = discriminator.bilinear(
            discriminator.message_layer_norm(readout)
        )
        normed = discriminator.referent_layer_norm(referents)
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
    [CROSS_RUNG, "16_birds_receiver_cross_attention_lm.toml"],
)
def test_the_two_listener_arms_are_parameter_matched(config_file):
    """
    Parity is a property of the pair of configs, so assert it as one: build the
        default GRU and the rung's transformer and compare the counts.

    4,687,872 against 4,784,566, which is +2.1%. Note 2 layers bidirectional
        would be 2.5x one layer's parameters and not 2x -- the second layer's
        input is the first's concatenated output, so its `weight_ih` is double
        -- which is the arithmetic that made the shared-256 scheme look cheaper
        than it was at 1024.

    Parameter parity is not interface parity: `output_size` is 1024 on the GRU
        against 256 on the transformer, so the discriminators downstream differ.
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
    assert n_cross == 4_784_566
    assert abs(n_cross / n_gru - 1.0) < 0.05

    assert gru.output_size == 1024
    assert cross.output_size == 256


# --------------------------------------------------------------------------
# The slot contract.
# --------------------------------------------------------------------------

ALL_FOUR = pytest.mark.parametrize(
    "language_model,discriminator",
    [
        ("ReceiverGRULM", "BilinearDiscriminator"),
        ("ReceiverGRULM", "AttentionDiscriminator"),
        ("ReceiverCrossAttentionLM", "BilinearDiscriminator"),
        ("ReceiverCrossAttentionLM", "AttentionDiscriminator"),
    ],
    ids=["gru+bilinear", "gru+attention", "cross+bilinear", "cross+attention"],
)


def _four_cell(language_model, discriminator, **kwargs):
    """
    Every cell from rung 11, which is the only config that states widths both
        slots can build: DEFAULT's `[receiver_language_model] d_model = 1024`
        does not divide its `heads = 5`, and that key is the GRU's.
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
    `(batch, slots, output_size)` from both, so either discriminator can
        consume either language model. The GRU returns its final state as a
        length-1 sequence; a length-1 cross-attention memory is legal.
    """
    listener = _four_cell(language_model, discriminator).eval()
    referents, messages = _inputs(listener)

    with torch.no_grad():
        representation = listener.language_model(messages, referents)

    assert representation.ndim == 3
    assert representation.shape[0] == BATCH
    assert representation.shape[-1] == listener.language_model.output_size

    expected_slots = 1 if language_model == "ReceiverGRULM" else SEQ
    assert representation.shape[1] == expected_slots


@ALL_FOUR
def test_the_discriminator_is_sized_from_the_language_model(
    language_model, discriminator
):
    """
    Not from a config key restating the width. `2 * d_model` for a
        bidirectional GRU and `d_model` for the decoder stack, and no
        arithmetic makes those agree, so a key would only ever be a key that
        could be wrong.
    """
    listener = _four_cell(language_model, discriminator)
    width = listener.language_model.output_size

    if discriminator == "BilinearDiscriminator":
        assert listener.discriminator.bilinear.in_features == width
    else:
        assert listener.discriminator.memory_adapter.in_features == width
        assert listener.discriminator.bilinear.bilinear.in_features == width


@ALL_FOUR
def test_the_gru_slot_ignores_the_candidate_set(language_model, discriminator):
    """
    Half of the uniform signature's cost, and it is paid deliberately: the GRU
        takes `referents` and does nothing with it, because dispatching on
        class at the call site would be worse. The cross-attention encoder
        reads them, which is its entire point.
    """
    listener = _four_cell(language_model, discriminator).eval()
    referents, messages = _inputs(listener)

    perturbed = referents.clone()
    perturbed[:, 0, :] += 5.0

    with torch.no_grad():
        before = listener.language_model(messages, referents)
        after = listener.language_model(messages, perturbed)

    if language_model == "ReceiverGRULM":
        assert torch.equal(before, after)
    else:
        assert not torch.allclose(before, after, atol=1e-6)


@ALL_FOUR
def test_both_slots_see_the_same_masked_referents(language_model, discriminator):
    """
    `Receiver` masks once. Two masks -- one per slot -- would regularise the
        listener at a rate no config key names, and only in the two pairings
        where both slots read the referents.

    Checked through `Receiver` itself rather than through the test shim, since
        the whole claim is about where the dropout lives.
    """
    listener = _four_cell(language_model, discriminator, dropout=0.5)
    receiver = R.Receiver(
        nn.Identity(),
        nn.Embedding(8, listener.token_embedding_size),
        listener.language_model,
        listener.discriminator,
        dropout=0.5,
    ).train()

    seen = []
    handles = [
        receiver.language_model.register_forward_pre_hook(
            lambda _module, args: seen.append(args[1])
        ),
        receiver.discriminator.register_forward_pre_hook(
            lambda _module, args: seen.append(args[0])
        ),
    ]

    referents = torch.randn(BATCH, N_OBJ, REFERENT_DIM)
    messages = torch.randn(
        BATCH,
        getattr(listener, "message_length", SEQ),
        listener.token_embedding_size,
    )
    # `Receiver` embeds the message with `messages @ token_embedding.weight`,
    #     so the "message" it wants is one-hot-shaped. The shim above sidesteps
    #     that; here it cannot.
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

    assert len(seen) == 2
    assert torch.equal(seen[0], seen[1])
    # And the mask really is on -- otherwise the equality above is vacuous.
    assert (seen[0] == 0.0).any()


@ALL_FOUR
def test_the_mask_removes_features_and_not_candidates(
    language_model, discriminator
):
    """
    Element-wise over `(batch, n_objects, features)`. A mask that removed whole
        candidates would leak the label ordering, which is the first half of
        the tensor.
    """
    listener = _four_cell(language_model, discriminator, dropout=0.5).train()
    referents = torch.ones(BATCH, N_OBJ, REFERENT_DIM)

    torch.manual_seed(5)
    masked = listener.input_dropout(referents)

    surviving = (masked != 0.0).float().mean(-1)
    assert (surviving > 0.0).all(), "a whole candidate was dropped"
    assert (surviving < 1.0).all(), "no candidate was masked at all"


def test_no_discriminator_reads_the_token_embedding():
    """
    The one-encoder invariant, checked structurally. `AttentionDiscriminator`
        carries a bilinear path, and that is a second *comparison*: it reads
        whatever the language model produced. Nothing in either discriminator
        may be sized from `token_embedding_size`, which is what a second
        encoder would need.
    """
    for discriminator in ("BilinearDiscriminator", "AttentionDiscriminator"):
        listener = build_listener(
            "ReceiverCrossAttentionLM",
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
# The mix.
# --------------------------------------------------------------------------

def _attention_discriminator(**overrides):
    return build_listener(
        "ReceiverCrossAttentionLM",
        "AttentionDiscriminator",
        REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
        discriminator_overrides=overrides,
    ).discriminator


def test_the_mix_opens_essentially_at_the_bilinear_comparison():
    """
    0.116 at the default floor and logit, which is the configuration measured
        bootstrapping under nuisance. The attention path is present enough to
        be learning and not enough to be deciding.
    """
    discriminator = _attention_discriminator()
    assert discriminator.mix_weight.item() == pytest.approx(0.116, abs=5e-3)


@pytest.mark.parametrize("logit", [-50.0, -8.0, 0.0, 8.0, 50.0])
def test_the_mix_stays_inside_its_bounds(logit):
    discriminator = _attention_discriminator()
    with torch.no_grad():
        discriminator.mix_logit.fill_(logit)

    weight = discriminator.mix_weight.item()
    assert discriminator.mix_floor <= weight <= 1.0


def test_the_floor_is_a_parameterisation_and_not_a_clamp():
    """
    The bug this design is built around. `torch.clamp` has zero gradient below
        its bound, so a mixing weight that drifted under the floor would weld
        there permanently and the attention path could never come back. Cost an
        afternoon in the prototype.

    Well below the floor's logit the sigmoid form still moves: strictly above
        the floor, and with a gradient that is small but not zero. The clamp it
        is contrasted with would give exactly the floor and exactly zero.
    """
    discriminator = _attention_discriminator()
    with torch.no_grad():
        discriminator.mix_logit.fill_(-8.0)

    weight = discriminator.mix_weight
    assert weight.item() > discriminator.mix_floor

    weight.backward()
    assert discriminator.mix_logit.grad is not None
    assert discriminator.mix_logit.grad.item() > 0.0

    under_a_clamp = torch.tensor(-8.0, requires_grad=True)
    clamped = torch.clamp(under_a_clamp, min=discriminator.mix_floor)
    clamped.backward()
    assert clamped.item() == pytest.approx(discriminator.mix_floor)
    assert under_a_clamp.grad.item() == 0.0


def test_the_attention_path_gets_gradient_at_the_opening_mix():
    """
    Why the floor exists at all. At a weight of exactly 0 the whole
        `referent_decoder` would receive nothing and could never earn its way
        in, whatever it might have learned.
    """
    listener = build_listener(
        "ReceiverCrossAttentionLM", "AttentionDiscriminator", REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
    ).train()
    referents, messages = _inputs(listener)

    listener(referents, messages).sum().backward()

    decoder_gradients = [
        parameter.grad
        for parameter in listener.discriminator.referent_decoder.parameters()
        if parameter.grad is not None
    ]
    assert decoder_gradients
    assert any(gradient.abs().max() > 0.0 for gradient in decoder_gradients)
    assert listener.discriminator.mix_logit.grad.abs().item() > 0.0
    # The branch weights are the volume now, so they are what must be receiving
    #     gradient where `log_mix_scale` used to be checked.
    assert listener.discriminator.decision.weight.grad.abs().max() > 0.0
    assert (
        listener.discriminator.bilinear.bilinear.weight.grad.abs().max() > 0.0
    )


def test_standardise_runs_in_the_telemetry_and_not_in_the_forward_path():
    """
    One job now, where it used to have two.

    `485b38e` took it out of `ScoreVolume.readout`. It had been dividing each
        game by the spread of its own candidate scores, which put the game's own
        margin in the denominator; both of `BilinearDiscriminator`'s operands
        arrive layer-normed, so the score is already backbone-independent and
        already opens at `1/sqrt(3)` without it. What remains is the telemetry
        block, where it runs per branch and gives `path_agreement` its Pearson-r
        reading and `mix_share` its like-for-like comparison.

    So the forward path must *not* leave a per-game spread of 1 -- that was the
        signature of the centring, and the test asserted it until this commit --
        while `standardise`'s own arithmetic stays pinned, because both
        telemetry readings depend on it.
    """
    listener = build_listener(
        "ReceiverCrossAttentionLM", "AttentionDiscriminator", REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
    ).eval()
    referents, messages = _inputs(listener)

    # Not standardised: the readout is a volume, so the spread the score comes
    #     out at is the one the branches and `score_scale` made, not 1.
    with torch.no_grad():
        scores = listener(referents, messages)
    spreads = scores.std(1, unbiased=False)
    assert not torch.allclose(spreads, torch.ones_like(spreads), atol=1e-2)

    scores = torch.randn(BATCH, N_OBJ) * 17.0 + 4.0
    standardised = R.standardise(scores)

    assert torch.allclose(
        standardised.mean(1), torch.zeros(BATCH), atol=1e-5
    )
    assert torch.allclose(
        standardised.std(1, unbiased=False), torch.ones(BATCH), atol=1e-5
    )


def test_standardising_survives_a_game_whose_candidates_all_score_alike():
    """
    The clamp inside `standardise` is safe where the one on `mix_logit` would
        not be: nothing learns through this bound, and 0/0 is the alternative.
    """
    scores = torch.full((BATCH, N_OBJ), 3.0)
    assert torch.isfinite(R.standardise(scores)).all()


def test_path_agreement_is_the_within_game_correlation():
    """
    The claim the metric rests on: both operands are already per-game zero-mean
        and unit-spread, so the mean of their product *is* Pearson's r.
    """
    generator = torch.Generator().manual_seed(2)
    first = torch.randn(BATCH, N_OBJ, generator=generator)
    second = torch.randn(BATCH, N_OBJ, generator=generator)

    reported = (R.standardise(first) * R.standardise(second)).mean().item()

    by_hand = []
    for game in range(BATCH):
        a = first[game] - first[game].mean()
        b = second[game] - second[game].mean()
        by_hand.append(
            (a * b).sum().item() / (a.norm().item() * b.norm().item())
        )

    assert reported == pytest.approx(sum(by_hand) / len(by_hand), abs=1e-5)


def test_the_mix_columns_are_set_on_every_forward():
    listener = build_listener(
        "ReceiverCrossAttentionLM", "AttentionDiscriminator", REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
    ).eval()
    discriminator = listener.discriminator

    assert math.isnan(discriminator.mix_alpha)
    assert math.isnan(discriminator.path_agreement)

    with torch.no_grad():
        listener(*_inputs(listener))

    assert discriminator.mix_alpha == pytest.approx(
        discriminator.mix_weight.item()
    )
    assert -1.0 <= discriminator.path_agreement <= 1.0
    assert math.isfinite(discriminator.decision_spread)


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
        "ReceiverCrossAttentionLM", "AttentionDiscriminator", REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
    ).eval()
    referents, messages = _inputs(listener)

    with torch.no_grad():
        slots = listener.language_model(messages, referents)

    # One slot per message position, so the readout is a real choice here.
    assert slots.shape[1] == listener.message_length
    assert not torch.allclose(slots[:, -1, :], slots.mean(1), atol=1e-5)

    bilinear = listener.discriminator.bilinear
    with torch.no_grad():
        taken = bilinear(referents, slots)
        from_eos = bilinear(referents, slots[:, -1:, :])
        from_mean = bilinear(referents, slots.mean(1, keepdim=True))

    assert torch.allclose(taken, from_eos, atol=1e-6)
    assert not torch.allclose(taken, from_mean, atol=1e-5)


def test_the_two_arms_build_the_same_bilinear_comparison():
    """
    The comparison is the same module either way -- same class, same weights --
        which is the invariant `AttentionDiscriminator`'s docstring claims when
        it says the mix's `a -> mix_floor` limit is the module that was measured
        bootstrapping and not a lookalike.

    They differ by their readout, and only there. `score_scale=False` on the
        composed path is what makes them differ, and it gates both of
        `ScoreVolume`'s scalars: the composed path gets neither a volume nor an
        offset. Both would be degenerate with the outer module's. The branch is
        multiplied by `1 - mix_weight` and read out downstream, so a scale on it
        says what `mix_logit` already says, and an inner constant across
        candidates says what the outer `score_bias` already says.
    """
    attention = _attention_discriminator()
    bilinear = build_listener(
        "ReceiverGRULM", "BilinearDiscriminator", REFERENT_DIM
    ).discriminator

    assert type(attention.bilinear) is type(bilinear)
    assert bilinear.learns_score_scale
    assert not attention.bilinear.learns_score_scale

    readout_scalars = ("log_score_scale", "score_bias")

    def _weights(module):
        return sorted(
            name for name, _ in module.named_parameters()
            if not name.endswith(readout_scalars)
        )

    assert _weights(attention.bilinear) == _weights(bilinear)
    assert not any(
        name.endswith(readout_scalars)
        for name, _ in attention.bilinear.named_parameters()
    )
    # And the outer module has exactly one of each.
    assert sorted(
        name for name, _ in attention.named_parameters()
        if name.endswith(readout_scalars)
    ) == ["log_score_scale", "score_bias"]


def test_the_composed_bilinear_weight_reaches_the_mixed_score():
    """
    The composed path's weight sets its branch's share of the score. That is
        what keeps `mix_share` different from `mix_alpha`, and it is why the
        branch norms stay load-bearing on this arm.

    Since `485b38e` the score's *spread* moves with it too. It used not to --
        the readout standardised, so a rescale of one branch changed the mix and
        nothing else, and this test asserted the spread held to within 2%. With
        the centring gone the volume is shared with the weights again on both
        arms: `bilinear_weight_norm` means magnitude as well as direction, and
        that is exactly what makes it readable as a metric.
    """
    listener = build_listener(
        "ReceiverCrossAttentionLM", "AttentionDiscriminator", REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
    ).eval()
    referents, messages = _inputs(listener)
    discriminator = listener.discriminator

    with torch.no_grad():
        before = listener(referents, messages)
        opening_share = discriminator.mix_share
        discriminator.bilinear.bilinear.weight.mul_(37.0)
        after = listener(referents, messages)

    assert not torch.allclose(before, after, atol=1e-5)
    assert discriminator.mix_share < opening_share
    assert (
        after.std(1, unbiased=False).mean().item()
        > 10.0 * before.std(1, unbiased=False).mean().item()
    )


def test_a_scale_on_a_standardised_path_would_have_been_inert():
    """
    Why the readout no longer standardises, kept as the measurement behind
        `485b38e` rather than as a live justification. It used to be the reason
        `AttentionDiscriminator` builds its composed path with
        `score_scale=False`; that reason is now degeneracy with `mix_logit`,
        which is asserted in
        `test_the_two_arms_build_the_same_bilinear_comparison` above.

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
            R.standardise(scale * scores), R.standardise(scores), atol=1e-6
        )

    # The same claim on the module itself, which is where it would have bitten,
    #     and stated as a ratio: what the scale gets through `standardise`
    #     against what the identical module gets without it.
    referents = torch.randn(BATCH, N_OBJ, REFERENT_DIM)
    message_repr = torch.randn(BATCH, 1, 64)

    gradients = {}
    for name, wrap in (("standardised", R.standardise), ("raw", lambda x: x)):
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

    It has lived in `log_mix_scale`, then in the two branch weights, and is now
        `ScoreVolume.log_score_scale` -- one cheap knob on an elevated learning
        rate again. Being cheap was the objection, on the grounds that the same
        scalar multiplied the gradient going back to the speaker; that objection
        did not survive AdamW's `m / sqrt(v)` and per-submodule clipping, both
        of which divide a uniform factor out. See test_score_scale.py.

    What this does not pin, deliberately, is that neither path can go quiet
        *alone*. Standardising each branch would guarantee it; nothing does, and
        `mix_share` against `mix_alpha` is what watches it.
    """
    listener = build_listener(
        "ReceiverCrossAttentionLM", "AttentionDiscriminator", REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
    ).eval()
    discriminator = listener.discriminator
    referents, messages = _inputs(listener)

    with torch.no_grad():
        loud = listener(referents, messages)
        discriminator.log_score_scale.fill_(math.log(0.01))
        quiet = listener(referents, messages)

    assert quiet.std().item() < 0.05 * loud.std().item()


def test_mix_floor_is_validated():
    with pytest.raises(ValueError, match="mix_floor"):
        _attention_discriminator(mix_floor=1.0)
    with pytest.raises(ValueError, match="mix_floor"):
        _attention_discriminator(mix_floor=-0.1)


# --------------------------------------------------------------------------
# Resetting.
# --------------------------------------------------------------------------

@ALL_FOUR
def test_reset_parameters_leaves_nothing_trained(language_model, discriminator):
    """
    Every submodule holding a parameter, the scalars included. A reset that
        skipped `mix_logit` would restart a run with the previous one's
        opinion about attention. See docs/anecdotes.md.
    """
    listener = _four_cell(language_model, discriminator)
    before = {
        name: parameter.detach().clone()
        for name, parameter in listener.named_parameters()
    }

    for parameter in listener.parameters():
        with torch.no_grad():
            parameter.add_(1.0)

    listener.language_model.reset_parameters()
    listener.discriminator.reset_parameters()

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


def test_resetting_returns_the_mix_and_the_readout_to_their_opening():
    """
    The offset resets through `ScoreVolume.reset_score_volume` now rather than
        through this module's own `reset_parameters`, which is the point of
        moving it: `BilinearDiscriminator` gets the same reset from the same
        place, where before it had no offset to reset.
    """
    listener = build_listener(
        "ReceiverCrossAttentionLM", "AttentionDiscriminator", REFERENT_DIM,
        config_file=rung(CROSS_RUNG),
    )
    discriminator = listener.discriminator
    opening = discriminator.mix_weight.item()

    with torch.no_grad():
        discriminator.mix_logit.fill_(3.0)
        discriminator.score_bias.fill_(1.0)
        discriminator.log_score_scale.fill_(2.0)

    discriminator.reset_parameters()

    assert discriminator.mix_weight.item() == pytest.approx(opening)
    assert discriminator.score_bias.item() == 0.0
    assert discriminator.score_scale.item() == pytest.approx(1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

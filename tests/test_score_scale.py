"""
Tests for the listener's score scale in code/models/receiver.py.

Runnable without pytest:  python tests/test_score_scale.py

`BilinearGRUComparer` scores a referent by a dot product with a projection of
the message, and until now the magnitude of that score was whatever the vision
model happened to emit. That is the same defect `e3fcabd` fixed in three places
on the speaker, arriving one agent later: on ViT2 the score's size was set by an
`nn.BatchNorm1d` at the end of broccoli's classification head, and on ResNet18
by the trunk's own normalisation, so the two arms of the ladder agreed only by
coincidence of their internals -- per batch, and differently at eval, where
BatchNorm switches to running statistics the training pass never used.

Three mechanisms replace it, and the tests are organised around which does what.

Both operands are layer-normalised without an affine, which is what makes the
score independent of the backbone. The referent norm is *not* a global rescale
-- it normalises each candidate separately -- so it changes which object wins,
and that is the point: without it an object can score highly for being large
rather than for matching the message.

The dot product is divided by `sqrt(referent_embedding_size)`. Both operands
leave LayerNorm at norm `sqrt(d)`, so without this the score would open at
`sqrt(d)` and a 512-wide ResNet18 listener would open 1.26x louder than a
320-wide ViT2 one, both far past the point where BCE is calibrated.

`log_score_scale` is then the one degree of freedom left over the score's
magnitude. It multiplies the message operand, which is shared across the objects
of a game, so it cannot change the decision -- only the confidence, and through
that the loss and every gradient. `29b18ea` is why it is free rather than
floored: a healthy pair dips ~0.2 log-units below its opening while the message
is still noise and comes back, and flooring that cost fifteen epochs.
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import models.builder  # noqa: E402
import models.receiver as R  # noqa: E402
import parse_config  # noqa: E402

CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "experiments", "ablation", "configs"
)

REFERENT_DIM = 512
BATCH, N_OBJ, SEQ = 8, 20, 7


def _comparer(referent_dim=REFERENT_DIM, **overrides):
    config = parse_config.get_config()["receiver_comparer"]
    config = {**config, **overrides}
    torch.manual_seed(0)
    return R.BilinearGRUComparer(referent_dim, **config)


def _inputs(comparer, referent_scale=1.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    referents = referent_scale * torch.randn(
        BATCH, N_OBJ, comparer.referent_embedding_size, generator=generator
    )
    messages = torch.randn(
        BATCH, SEQ, comparer.token_embedding_size, generator=generator
    )
    return referents, messages


# --------------------------------------------------------------------------
# The norms, and that neither carries an affine.
# --------------------------------------------------------------------------

def test_neither_operand_norm_has_an_affine():
    comparer = _comparer()
    assert comparer.referent_layer_norm.weight is None
    assert comparer.referent_layer_norm.bias is None
    assert comparer.message_layer_norm.weight is None
    assert comparer.message_layer_norm.bias is None


def test_the_cross_attention_comparer_agrees():
    """The other comparer's referent norm, for the same reason."""
    config = parse_config.get_config(
        os.path.join(CONFIG_DIR, "12_birds_receiver_cross_attention.toml")
    )
    comparer = R.TransformerCrossAttentionComparer(
        320, **config["receiver_comparer"]
    )
    assert comparer.referent_layer_norm.weight is None
    assert comparer.referent_layer_norm.bias is None


# 1e-4 sits well below where `nn.LayerNorm`'s 1e-5 default would give out --
#     the incoming variance there is 1e-8 -- which is what `LAYER_NORM_EPS`
#     is for. At the default, referent scale 0.01 alone came out 4.5% off.
@pytest.mark.parametrize("referent_scale", [1e-4, 0.01, 1.0, 100.0])
def test_scores_are_independent_of_the_referent_magnitude(referent_scale):
    """
    A backbone emitting features a hundred times larger must not thereby make
        its listener a hundred times more confident. This is the property the
        whole change exists for.
    """
    comparer = _comparer().eval()
    referents, messages = _inputs(comparer, referent_scale=referent_scale)
    with torch.no_grad():
        scores = comparer(referents, messages)

    reference = _comparer().eval()
    with torch.no_grad():
        expected = reference(*_inputs(reference, referent_scale=1.0))

    # Relative, because the claim is scale invariance. 1e-3 sits above float32
    #     rounding (6.7e-5 at the smallest scale here) and far below what a
    #     failure would look like: at `nn.LayerNorm`'s 1e-5 default, referent
    #     scale 0.01 came out 4.5% adrift.
    assert torch.allclose(scores, expected, rtol=1e-3, atol=1e-6)


def test_the_referent_norm_is_not_a_global_rescale():
    """
    It normalises each candidate separately, so it can and must change which
        object wins. Enlarging one candidate alone must not promote it.
    """
    comparer = _comparer().eval()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        before = comparer(referents, messages)

    inflated = referents.clone()
    inflated[:, 3, :] *= 50.0
    with torch.no_grad():
        after = comparer(inflated, messages)

    assert torch.allclose(before, after, atol=1e-4)


def test_an_unnormalised_referent_would_have_been_promoted():
    """
    The counterfactual the test above is worth checking against: the same
        inflation, scored the way it was scored before this change.
    """
    comparer = _comparer().eval()
    referents, messages = _inputs(comparer)
    inflated = referents.clone()
    inflated[:, 3, :] *= 50.0

    with torch.no_grad():
        token_embeddings, _ = comparer.gru(messages)
        projected = comparer.bilinear(token_embeddings[:, -1, ...])
        raw = torch.einsum("ijh,ih->ij", (inflated, projected))

    assert raw[:, 3].abs().mean() > 10.0 * raw.abs().mean()


# --------------------------------------------------------------------------
# The unit, and where the scale opens.
# --------------------------------------------------------------------------

def test_the_scale_opens_at_one():
    assert _comparer().score_scale.item() == pytest.approx(1.0)


@pytest.mark.parametrize("referent_dim", [320, 512])
def test_the_untrained_score_opens_near_unit_variance(referent_dim):
    """
    What `1/sqrt(referent_embedding_size)` buys: the opening confidence is the
        same whichever backbone the rung mounts, rather than growing as
        `sqrt(d)` with its width.
    """
    comparer = _comparer(referent_dim=referent_dim).eval()
    with torch.no_grad():
        scores = comparer(*_inputs(comparer))

    assert 0.3 < scores.std().item() < 3.0


@pytest.mark.parametrize("referent_dim", [320, 512])
def test_untrained_bce_opens_near_ln_2(referent_dim):
    """
    The reason the opening confidence matters. A listener that opens by
        shouting wrong answers makes muting the fast descent direction, which
        is the state `e3fcabd` was written about.
    """
    comparer = _comparer(referent_dim=referent_dim).eval()
    with torch.no_grad():
        scores = comparer(*_inputs(comparer))

    labels = torch.zeros(BATCH, N_OBJ)
    labels[:, : N_OBJ // 2] = 1.0
    loss = F.binary_cross_entropy_with_logits(scores, labels).item()

    assert loss < 2.0 * math.log(2.0)


# --------------------------------------------------------------------------
# What the scale can and cannot do.
# --------------------------------------------------------------------------

def test_the_scale_cannot_change_the_decision():
    """
    It multiplies an operand shared across the objects of a game, so it scales
        every score together. `train.py` reads the decision as `scores > 0` and
        the reference-game branch as an argmax; neither moves.
    """
    comparer = _comparer().eval()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        quiet = comparer(referents, messages)
        with torch.no_grad():
            comparer.log_score_scale.fill_(math.log(37.0))
        loud = comparer(referents, messages)

    assert torch.equal(quiet > 0, loud > 0)
    assert torch.equal(quiet.argmax(1), loud.argmax(1))
    assert torch.allclose(loud, 37.0 * quiet, atol=1e-4)


def test_the_scale_does_change_the_loss():
    """
    Which is the whole point of having it: BCE is not scale-invariant, so this
        is the listener's one control over its own confidence.
    """
    comparer = _comparer().eval()
    referents, messages = _inputs(comparer)
    labels = torch.zeros(BATCH, N_OBJ)
    labels[:, : N_OBJ // 2] = 1.0

    with torch.no_grad():
        quiet = F.binary_cross_entropy_with_logits(
            comparer(referents, messages), labels
        ).item()
        comparer.log_score_scale.fill_(math.log(37.0))
        loud = F.binary_cross_entropy_with_logits(
            comparer(referents, messages), labels
        ).item()

    assert loud > quiet


def test_the_scale_receives_gradient():
    comparer = _comparer()
    referents, messages = _inputs(comparer)
    labels = torch.zeros(BATCH, N_OBJ)
    labels[:, : N_OBJ // 2] = 1.0

    F.binary_cross_entropy_with_logits(
        comparer(referents, messages), labels
    ).backward()

    assert comparer.log_score_scale.grad is not None
    assert comparer.log_score_scale.grad.abs().item() > 0.0


def test_bilinear_cannot_reach_the_score_magnitude():
    """
    `message_layer_norm` sits on `bilinear`'s output, so growing `W` changes
        the direction of the comparison and not its volume. This is what leaves
        `score_scale` as the only route, and it is the difference between the
        ordering we shipped and the one where the norm sat on the GRU state.
    """
    comparer = _comparer().eval()
    referents, messages = _inputs(comparer)
    with torch.no_grad():
        before = comparer(referents, messages)
        comparer.bilinear.weight.mul_(100.0)
        after = comparer(referents, messages)

    assert torch.allclose(before, after, atol=1e-4)


def test_reset_parameters_returns_the_scale_to_its_opening():
    comparer = _comparer()
    with torch.no_grad():
        comparer.log_score_scale.fill_(math.log(37.0))
    comparer.reset_parameters()

    assert comparer.score_scale.item() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The optimiser wiring.
# --------------------------------------------------------------------------

def _pair_and_optimiser(config_file):
    config = parse_config.get_config(os.path.join(CONFIG_DIR, config_file))
    config["cuda"] = False

    class _Dataset:
        n_feats = (3, 224, 224)
        name = "cub"

    class _Loader:
        dataset = _Dataset()

    built = models.builder.build_models({"train": _Loader()}, config)
    return config, built["pair"], built["optimiser"]


def test_the_scale_lands_in_a_group_at_its_own_rate():
    config, pair, optimiser = _pair_and_optimiser("02_birds_baseline.toml")
    wanted = config["optimiser"]["score_scale_lr"]
    assert wanted != config["optimiser"]["lr"]

    scale = pair.receiver.comparer.log_score_scale
    holding = [
        group for group in optimiser.param_groups
        if any(p is scale for p in group["params"])
    ]

    assert len(holding) == 1
    assert holding[0]["lr"] == wanted
    assert holding[0]["weight_decay"] == 0.0


def test_the_cross_attention_comparer_is_skipped_rather_than_raising():
    """
    It has no `log_score_scale` by construction, so `score_scale_lr` is
        inapplicable there in the way `[sender_language_model] heads` is
        inapplicable to a GRU speaker -- not missing.
    """
    _, pair, _ = _pair_and_optimiser("12_birds_receiver_cross_attention.toml")
    assert not hasattr(pair.receiver.comparer, "log_score_scale")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

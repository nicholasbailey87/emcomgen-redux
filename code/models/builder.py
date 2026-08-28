"""
Model building logic
"""

from . import base
from . import sender as sender
from . import receiver as receiver

from .backbone import vision

from torch import nn

from gradboard.optimiser import get_optimiser

def _regroup(optimiser, selected, lr):
    """
    Move an already-selected list of parameters into a group of its own at
        `lr`, leaving every other parameter where `get_optimiser` put it.

    Split out from `split_out_parameter`, which is the only caller: this is
        the part that actually moves anything, once the parameters have been
        chosen. Must run before `PASS` is constructed, which deep-copies the
        groups once and thereafter scales each from its own recorded base lr.
        See docs/training.md.

    Args:
        optimiser: the optimiser returned by `get_optimiser`
        selected: the parameters to move, already filtered
        lr: learning rate for the new group

    Returns:
        The same optimiser, mutated in place.
    """
    identities = {id(p) for p in selected}

    for group in optimiser.param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in identities]

    # `weight_decay` 0.0 to match what `get_optimiser` gave all of these; see
    #     docs/training.md.
    optimiser.add_param_group(
        {"params": selected, "lr": lr, "weight_decay": 0.0}
    )

    return optimiser


def split_out_parameter(optimiser, pair, suffix, lr, config_key):
    """
    Move every parameter of `pair` whose name ends in `suffix` into a parameter
        group of its own at `lr`.

    For the lone scalars of `SPLIT_LEARNING_RATES`, each of which is one tensor
        with a distinctive name.

    Args:
        optimiser: the optimiser returned by `get_optimiser`
        pair: the `base.Pair` whose parameters are being regrouped
        suffix: the `named_parameters` suffix identifying them
        lr: learning rate for the new group
        config_key: the `[optimiser]` key that asked for this, named in the
            error so a rename says which knob went quiet

    Returns:
        The same optimiser, mutated in place.
    """
    selected = [
        p for name, p in pair.named_parameters()
        if name.endswith(suffix)
    ]

    if not selected:
        raise RuntimeError(
            f"No `{suffix}` found on the pair, so `{config_key}` would "
            "silently do nothing. Has the parameter been renamed?"
        )

    return _regroup(optimiser, selected, lr)


# `(config key, parameter suffix, applies to)`, in the order the groups are
#     added. The test gates on the architecture rather than on finding the
#     parameter: a GRU speaker has no polarity tag, and
#     a `BilinearDiscriminator` has no mixing weight, so for those the key is
#     inapplicable rather than broken. See docs/training.md -- and do not read a
#     gate as a verdict on the parameter.
SPLIT_LEARNING_RATES = (
    (
        "logit_scale_lr",
        "log_logit_scale",
        lambda pair: True,
    ),
    (
        "polarity_embedding_lr",
        "polarity_embedding",
        lambda pair: isinstance(
            pair.sender.language_model, sender.SenderTransformerLM
        ),
    ),
    (
        # The listener's volume, and the counterpart of `logit_scale_lr` above:
        #     both are a lone scalar in front of a normalised quantity, both
        #     travel at most `lr * steps`, and both need to be able to move
        #     within a run.
        #
        # Elevated on purpose, and that was once the objection: at 2e-3 the
        #     listener could squash its own logits fast, which multiplied down
        #     the gradient reaching the speaker. `a9a6a9c` answered that by
        #     deleting the scalar, which left the volume in a 320x320 matrix
        #     that moved 1.3% in thirty epochs; `7b10d47` answered it by hiding
        #     the scale from the backward pass. Both were answering a coupling
        #     that never reached the optimiser. AdamW updates by `m / sqrt(v)`,
        #     so a uniform factor on a parameter's gradient cancels, and
        #     `train.py`'s `clip_gradients` renormalises each submodule to
        #     `clip_grad_norm` whenever it binds -- which at recorded speaker
        #     norms of ~10 against a ceiling of 1.0 it does. A fast calibration
        #     is just a fast calibration.
        #
        # One key covers both discriminators: `ScoreVolume` puts the same
        #     `log_score_scale` on each, so the `mix_scale_lr` that used to
        #     move `AttentionDiscriminator`'s own scalar has no successor.
        "score_scale_lr",
        "log_score_scale",
        lambda pair: True,
    ),
    (
        # The listener's threshold, and the offset half of the same readout.
        #     `train.py` decides on `lis_scores > 0`, so this is the parameter
        #     that places the scores against that origin, and like the volume it
        #     is a lone scalar whose whole travel is bounded by `lr * steps`.
        #
        # One key covers both discriminators, exactly as `score_scale_lr` does
        #     and for the same reason: `ScoreVolume` puts the same `score_bias`
        #     on each. It replaces `AttentionDiscriminator.mix_bias`, which had
        #     no key at all and so sat at the base 1e-4 -- at birds' 194 steps an
        #     epoch that bounded its whole travel at 0.58 over thirty epochs,
        #     against a score whose own opening spread is 0.577.
        "score_bias_lr",
        "score_bias",
        lambda pair: True,
    ),
    (
        # Moves the parameter reported as `train_mix_alpha`. Named for the
        #     parameter rather than the column so the suffix beside it is
        #     obviously the same thing. A mixing weight and not a volume, which
        #     is why it survived the round that took the volumes out.
        "mix_logit_lr",
        "mix_logit",
        lambda pair: isinstance(
            pair.receiver.discriminator, receiver.AttentionDiscriminator
        ),
    ),
    (
        # The one scalar standing between the contrast stage and the identity.
        #     It opens at exactly zero, and at the base rate a lone scalar
        #     cannot travel further than `lr * steps` -- 62 steps an epoch on
        #     birds -- so without this the stage would stay shut for most of a
        #     run. See `sender.ExampleContrast`.
        "contrast_gate_lr",
        "contrast_gate",
        lambda pair: pair.sender.contrast is not None,
    ),
)




def build_models(dataloaders, config):
    n_feats = dataloaders["train"].dataset.n_feats

    # None of this applies in these experiments.
    assert not config['receiver_only']
    assert not config['copy_receiver']
    assert not config['share_language_model']
    assert not config['share_feat_model']
    assert not len(n_feats) == 1

    # Set up sender
    sender_class = getattr(sender, config['sender']['class'])
    sender_feature_model_class = getattr(vision, config['sender']['feature_model'])
    sender_prototyper_class = getattr(sender, config['sender']['prototyper'])
    sender_language_model_class = getattr(sender, config['sender']['language_model'])

    sender_feature_model = sender_feature_model_class(
        n_feats=n_feats,
        **config['sender_feature_model']
    )
    sender_prototyper = sender_prototyper_class(sender_feature_model.final_feat_dim)
    sender_language_model = sender_language_model_class(
        sender_feature_model.final_feat_dim,
        **config['sender_language_model']
    )

    # A boolean rather than a class name, because there is one of these or there
    #     is nothing: the stage is a residual on the referents, so "off" is
    #     `None` and not another module. `False` builds the speaker that existed
    #     before it, and `True` opens at that speaker exactly -- see
    #     `sender.ExampleContrast`.
    sender_contrast = (
        sender.ExampleContrast(
            sender_feature_model.final_feat_dim,
            **config['sender_contrast']
        )
        if config['sender']['contrast']
        else None
    )

    sender_ = sender_class(
        feat_model = sender_feature_model,
        prototyper = sender_prototyper,
        language_model = sender_language_model,
        contrast = sender_contrast,
        vision_dropout = config['sender']['vision_dropout'],
        prototype_dropout = config['sender']['prototype_dropout']
    )

    # Set up receiver
    receiver_class = getattr(receiver, config['receiver']['class'])
    receiver_feature_model_class = getattr(vision, config['receiver']['feature_model'])
    receiver_language_model_class = getattr(
        receiver, config['receiver']['language_model']
    )
    receiver_discriminator_class = getattr(
        receiver, config['receiver']['discriminator']
    )

    receiver_feature_model = receiver_feature_model_class(
        n_feats=n_feats,
        **config['receiver_feature_model']
    )
    receiver_token_embedding_module = nn.Embedding(
        config['sender_language_model']['vocabulary'] + 4, # +4 for PAD, SOS, EOS, UNK
        config['receiver_language_model']['token_embedding_size']
    )
    if (
        ('message_length' in config['receiver_language_model'])
        and
        (
            config['receiver_language_model']['message_length']
            !=
            config['sender_language_model']['message_length']
        )
    ):
        raise ValueError(
            "receiver_language_model.message_length, if it exists, "
            "must be equal to sender_language_model.message_length"
        )

    receiver_language_model = receiver_language_model_class(
        receiver_feature_model.final_feat_dim,
        **config['receiver_language_model']
    )

    # The discriminator is sized from the language model rather than from the
    #     config: which width the message arrives at is the encoder's business
    #     -- `2 * d_model` for a bidirectional GRU, `d_model` for the decoder
    #     stack -- and a config key restating it is a key that can be wrong.
    receiver_discriminator = receiver_discriminator_class(
        receiver_feature_model.final_feat_dim,
        receiver_language_model.output_size,
        **config['receiver_discriminator']
    )

    receiver_ = receiver_class(
        feature_model = receiver_feature_model,
        token_embedding_module=receiver_token_embedding_module,
        language_model = receiver_language_model,
        discriminator = receiver_discriminator,
        dropout = config['receiver']['dropout'],
    )

    pair = base.Pair(sender_, receiver_)

    if config['cuda']:
        pair = pair.cuda()
    
    optimiser = get_optimiser(
        pair,
        config['sender_language_model']['d_model'],
        lr=config['optimiser']['lr'],
        weight_decay=config['optimiser']['weight_decay']
    )

    base_lr = config['optimiser']['lr']

    for config_key, suffix, applies_to in SPLIT_LEARNING_RATES:
        lr = config['optimiser'].get(config_key, base_lr)

        if lr != base_lr and applies_to(pair):
            optimiser = split_out_parameter(
                optimiser, pair, suffix, lr, config_key
            )

    return {
        "pair": pair,
        "optimiser": optimiser,
    }
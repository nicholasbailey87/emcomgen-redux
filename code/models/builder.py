"""
Model building logic
"""

from . import base
from . import sender as sender
from . import receiver as receiver

from .backbone import vision

from torch import nn

from gradboard.optimiser import get_optimiser

def is_transformer_param(name):
    return name.startswith("sender.transformer") or name.startswith("sender.cls_emb")


def split_out_parameter(optimiser, pair, suffix, lr, config_key):
    """
    Move every parameter of `pair` whose name ends in `suffix` into a parameter
        group of its own at `lr`, leaving every other parameter where
        `get_optimiser` put it.

    Must run before `PASS` is constructed, which deep-copies the groups once and
        thereafter scales each from its own recorded base lr. See
        docs/training.md.

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

    identities = {id(p) for p in selected}

    for group in optimiser.param_groups:
        group["params"] = [p for p in group["params"] if id(p) not in identities]

    # `weight_decay` 0.0 to match what `get_optimiser` gave both of these; see
    #     docs/training.md.
    optimiser.add_param_group(
        {"params": selected, "lr": lr, "weight_decay": 0.0}
    )

    return optimiser


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

    sender_ = sender_class(
        feat_model = sender_feature_model,
        prototyper = sender_prototyper,
        language_model = sender_language_model,
        vision_dropout = config['sender']['vision_dropout'],
        prototype_dropout = config['sender']['prototype_dropout']
    )

    # Set up receiver
    receiver_class = getattr(receiver, config['receiver']['class'])
    receiver_feature_model_class = getattr(vision, config['receiver']['feature_model'])
    receiver_comparer_class = getattr(receiver, config['receiver']['comparer'])
    
    receiver_feature_model = receiver_feature_model_class(
        n_feats=n_feats,
        **config['receiver_feature_model']
    )
    receiver_token_embedding_module = nn.Embedding(
        config['sender_language_model']['vocabulary'] + 4, # +4 for PAD, SOS, EOS, UNK
        config['receiver_comparer']['token_embedding_size']
    )
    if (
        ('message_length' in config['receiver_comparer'])
        and
        (
            config['receiver_comparer']['message_length']
            !=
            config['sender_language_model']['message_length']
        )
    ):
        raise ValueError(
            "receiver_comparer.message_length, if it exists, "
            "must be equal to sender_language_model.message_length"
        )

    receiver_comparer = receiver_comparer_class(
        receiver_feature_model.final_feat_dim,
        **config['receiver_comparer']
    )

    receiver_ = receiver_class(
        feature_model = receiver_feature_model,
        token_embedding_module=receiver_token_embedding_module,
        comparer = receiver_comparer
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

    logit_scale_lr = config['optimiser'].get('logit_scale_lr', base_lr)

    if logit_scale_lr != base_lr:
        optimiser = split_out_parameter(
            optimiser, pair, "log_logit_scale", logit_scale_lr, "logit_scale_lr"
        )

    # Gated on the speaker class rather than on finding the parameter: a GRU
    #     speaker has no polarity tag, so the key is inapplicable rather than
    #     broken. See docs/training.md.
    polarity_embedding_lr = config['optimiser'].get(
        'polarity_embedding_lr', base_lr
    )

    if (
        polarity_embedding_lr != base_lr
        and isinstance(pair.sender.language_model, sender.SenderTransformerLM)
    ):
        optimiser = split_out_parameter(
            optimiser,
            pair,
            "polarity_embedding",
            polarity_embedding_lr,
            "polarity_embedding_lr",
        )

    # Gated on the comparer, for the reason the polarity tag is gated on the
    #     speaker: `TransformerCrossAttentionComparer` has no learnable scale.
    #     See docs/training.md -- and do not read the gate as a verdict on the
    #     parameter.
    score_scale_lr = config['optimiser'].get('score_scale_lr', base_lr)

    if (
        score_scale_lr != base_lr
        and isinstance(pair.receiver.comparer, receiver.BilinearGRUComparer)
    ):
        optimiser = split_out_parameter(
            optimiser,
            pair,
            "log_score_scale",
            score_scale_lr,
            "score_scale_lr",
        )

    return {
        "pair": pair,
        "optimiser": optimiser,
    }
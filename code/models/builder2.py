"""
Model building logic
"""

from . import base
from . import sender2 as sender
from . import receiver2 as receiver

from .backbone import vision

from torch import optim, nn


def is_transformer_param(name):
    return name.startswith("sender.transformer") or name.startswith("sender.cls_emb")


def build_models(dataloaders, config):
    n_feats = dataloaders["train"].dataset.n_feats

    # Putting these additional checks in as this stuff
    #     should never apply in my experiments
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
        vision_dropout = config['sender']['vision_dropout']
    )

    # Set up receiver
    receiver_class = getattr(receiver, config['receiver']['class'])
    receiver_feature_model_class = getattr(vision, config['receiver']['feature_model'])
    receiver_comparer_class = getattr(receiver, config['receiver']['comparer'])
    
    receiver_feature_model = receiver_feature_model_class(n_feats=n_feats)
    receiver_token_embedding_module = nn.Embedding(
        config['sender_language_model']['vocabulary'] + 2, # +2 for SOS and EOS
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
        comparer = receiver_comparer,
        vision_dropout = config['receiver']['vision_dropout']
    )

    pair = base.Pair(sender_, receiver_)

    if config['cuda']:
        pair = pair.cuda()
    
    optimiser = optim.AdamW(pair.parameters(), lr=config['optimiser']['lr'])
    
    return {
        "pair": pair,
        "optimiser": optimiser,
    }
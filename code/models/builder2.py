"""
Model building logic
"""


from . import base
from . import sender2 as sender
from . import receiver2 as receiver

from .backbone import vision
# from .backbone import feature
from .backbone import BACKBONES

from torch import optim, nn

from data.shapeworld import SHAPES, COLORS

# In my work, this is never used:
DEFAULT_MODELS = {
    "shapeworld": vision.Conv4,
    "cub": vision.ResNet18,
}


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

    # if len(n_feats) == 1:  # Feature based; use mlp

    #     def feat_fn(which):
    #         if which == "sender":
    #             output_size = config.sender.arguments.d_model
    #             n_layers = config.sender.arguments.layers
    #         else:
    #             output_size = config.receiver.arguments.d_model
    #             n_layers = config.receiver.arguments.layers
    #         return feature.FeatureMLP(
    #             input_size=n_feats[0],
    #             output_size=output_size,
    #             n_layers=n_layers,
    #         )

    # else:

    #     def feat_fn(which):
    #         # To use comm, make this conv4.
    #         if args.pretrained_feat_model:
    #             if dataloaders["train"].dataset.name == "shapeworld":
    #                 raise NotImplementedError
    #             return vision.PretrainedResNet18()
    #         elif args.backbone is None:
    #             # Made the change below so that we MUST specify the backbone
    #                 raise ValueError("backbone must be specified")
    #         else:
    #             return BACKBONES[args.backbone](n_feats=n_feats)

    # if config.receiver_only:
    #     sender_feat_model = None
    # else:
    #     sender_feat_model = (
    #         BACKBONES[config.sender.arguments.image_encoder](n_feats=n_feats)
    #     )
    #     print("Parameters of sender vision backbone:")
    #     print(sum(p.numel() for p in sender_feat_model.parameters() if p.requires_grad))

    # Set up sender
    sender_class = getattr(sender, config['sender']['class'])
    sender_feature_model_class = getattr(vision, config['sender']['feature_model'])
    sender_prototyper_class = getattr(sender, config['sender']['prototyper'])
    sender_language_model_class = getattr(sender, config['sender']['language_model'])

    sender_feat_model = sender_feature_model_class(n_feats=n_feats)
    sender_prototyper = sender_prototyper_class(config['sender']['arguments']['d_model'])
    sender_language_model = sender_language_model_class(
        receiver_feature_model.final_feat_dim,
        config['sender']['token_embedding_size'],
        config['sender']['d_model'],
        config['sender']['vocabulary'],
        config['sender']['layers'],
        config['sender']['bidirectional']
    )

    sender_ = sender_class(
        feat_model = sender_feat_model,
        prototyper = sender_prototyper,
        language_model = sender_language_model,
        dropout = config['sender']['arguments']['dropout']
    )

    # Set up receiver
    receiver_class = getattr(receiver, config['receiver']['class'])
    receiver_feature_model_class = getattr(vision, config['receiver']['feature_model'])
    receiver_comparer_class = getattr(receiver, config['receiver']['comparer'])
    
    receiver_feature_model = receiver_feature_model_class(n_feats=n_feats)
    receiver_token_embedding_module = nn.Embedding(
        config['sender']['arguments']['vocabulary'] + 3,
        config['receiver']['arguments']['token_embedding_size']
    )
    receiver_comparer = receiver_comparer_class(
        receiver_feature_model.final_feat_dim,
        config
    )

    receiver_ = receiver_class(
        feature_model = receiver_feature_model,
        token_embedding_module=receiver_token_embedding_module,
        comparer = receiver_comparer,
        dropout = config['receiver']['arguments']['dropout']
    )
    # if config['share_language_model']:
    #     assert sender_language_model is not None
    #     receiver_language_model = sender_language_model
    # else:
    #     receiver_language_model = None
        # nn.GRU(
        #     config['receiver']['arguments']['embedding_size'],
        #     config['receiver']['arguments']['d_model']
        # )
        # print("Parameters of receiver language_model:")
        # print(sum(p.numel() for p in receiver_language_model.parameters() if p.requires_grad))

    # if config.receiver_only or config.copy_receiver:
    #     # Copy entire teacher internal state
    #     if config.receiver_only:
    #         sender_ = None
    #         sender_size = None
    #     else:
    #         sender_ = sender.Copysender(
    #             sender_feat_model,
    #             dropout=args.dropout,
    #             prototype=args.prototype,
    #             n_transformer_heads=args.n_transformer_heads,
    #             n_transformer_layers=args.n_transformer_layers,
    #         )
    #         sender_size = sender_.emb_size
    #     receiver_ = receiver.Copyreceiver(
    #         receiver_feat_model, message_size=sender_size, dropout=args.dropout
    #     )
    # else:
    # (account for SOS, EOS, UNK)

    # sender_embs = nn.Embedding(
    #     config['sender']['arguments']['vocabulary'] + 3,
    #     config['sender']['arguments']['embedding_size']
    # )
    # receiver_embs = nn.Embedding(
    #     config['receiver']['arguments']['vocabulary'] + 3,
    #     config['receiver']['arguments']['embedding_size']
    # )

    # sender_class = getattr(sender, config['sender']['class'])

    # receiver_class = getattr(receiver, config['receiver']['class'])
    
    # sender_ = sender_class(
    #     sender_feat_model,
    #     sender_language_model,
    #     sender_embs,
    #     hidden_size=config['sender']['arguments']['d_model'],
    #     dropout=config['sender']['arguments']['dropout'],
    #     tau=config['sender']['arguments']['temperature'],
    #     prototype=config['sender']['arguments']['prototype'],
        # n_transformer_heads=args.n_transformer_heads,
        # n_transformer_layers=args.n_transformer_layers,
    # )
    
    # receiver_ = receiver_class(
    #     receiver_feat_model,
    #     receiver_language_model,
    #     receiver_embs,
    #     message_size=config['receiver']['arguments']['d_model'],
    #     dropout=config['receiver']['arguments']['dropout'],
    # )

    pair = base.Pair(sender_, receiver_)

    if config['cuda']:
        pair = pair.cuda()

    # Optimization
    # opt_params = [
    #     {
    #         "params": [
    #             p
    #             for name, p in pair.named_parameters()
    #             if not is_transformer_param(name)
    #         ],
    #         "lr": config.optimiser.lr,
    #     }
    # ]
    # if args.prototype == "transformer":
    #     opt_params.append(
    #         {
    #             "params": [
    #                 p
    #                 for name, p in pair.named_parameters()
    #                 if is_transformer_param(name)
    #             ],
    #             "lr": args.transformer_lr
    #             if args.transformer_lr is not None
    #             else args.lr,
    #         }
    #     )

    # TODO: import a library that will build my optimizer based on config
    
    optimiser = optim.AdamW(pair.parameters(), lr=config['optimiser']['lr'])
    
    return {
        "pair": pair,
        "optimiser": optimiser,
    }
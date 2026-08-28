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

    The shared half of `split_out_parameter` and `split_out_module`, which
        differ only in how they choose `selected`. Must run before `PASS` is
        constructed, which deep-copies the groups once and thereafter scales
        each from its own recorded base lr. See docs/training.md.

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
        with a distinctive name. `split_out_module` is the counterpart for "every
        tensor in this submodule", which no suffix can express.

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


# Substrings marking a tensor as an *input* to the network rather than a
#     transformation within it. Matched case-insensitively anywhere in the
#     `named_parameters` name, as `gradboard.optimiser`'s
#     `EXCLUDE_FROM_WEIGHT_DECAY` does, rather than enumerated: the same rule
#     then covers a module added later without anyone having to remember this
#     list. Covers `polarity_embedding`, `label_embedding`, `token_embedding`,
#     `query` and `token_embedding`. It used to cover `output_query` too; the
#     speaker's two arms now read the message off the tail of the latent array
#     rather than through a learned readout, so that parameter is gone. The rule
#     is unchanged -- "query" still matches `query`.
MUP_EMBEDDING_LIKE = ("embedding", "query")


def is_mup_exempt(name, parameter):
    """
    Whether `parameter` keeps the base learning rate rather than the muP one.

    muP's rule is per *tensor type*, not per module: only the matrices that map
        one width to another take a learning rate in 1/fan_in. Two kinds do not.

        Anything with fewer than two dimensions -- biases, norm gains, and every
        learned scalar -- has no fan-in to scale by, and its update is not a
        matrix-vector product whose variance grows with width.

        Embedding-like tensors take a Theta(1) rate under muP because their
        "fan-in" is a one-hot index rather than a width. Every one of them here
        is also Theta(1)-*initialised* -- `nn.init.normal_(std=1.0)` in
        `SenderTransformerLM.reset_parameters` -- so scaling their rate by width
        would be scaling against an init that never shrank.

    The exemption is load-bearing for a second reason: it makes the muP groups
        disjoint from every entry in `SPLIT_LEARNING_RATES` by construction.
        `log_logit_scale`, `log_score_scale`, `score_bias`, `mix_logit` and
        `contrast_gate` are all 0-d; `polarity_embedding` is 2-d but matches
        "embedding". So no parameter can be claimed twice however the two loops
        are ordered.

    Args:
        name: the parameter's name, relative to the module being split out
        parameter: the tensor itself

    Returns:
        True if it should stay at the base rate.
    """
    if parameter.dim() < 2:
        return True

    return any(
        fragment in name.lower() for fragment in MUP_EMBEDDING_LIKE
    )


def split_out_module(optimiser, module, lr, config_key, exempt=is_mup_exempt):
    """
    Move every non-exempt parameter of `module` into a group of its own at `lr`.

    Selects on the module object, so unlike `split_out_parameter` there is no
        name to match and nothing a rename can quietly break. "Every tensor in
        `sender.language_model`" is not expressible as a suffix, which is why
        this is a sibling rather than an argument to that one.

    Must run before `PASS` is constructed; see `_regroup`.

    Args:
        optimiser: the optimiser returned by `get_optimiser`
        module: the submodule whose parameters are being regrouped
        lr: learning rate for the new group
        config_key: what asked for this, for the caller's own reporting
        exempt: `(name, parameter) -> bool`, called with names relative to
            `module`. Defaults to muP's rule; see `is_mup_exempt`.

    Returns:
        The same optimiser, mutated in place. Unchanged if every parameter of
            `module` is exempt -- a module that is nothing but norms and biases
            is a legitimate thing to ask about, and an empty group would only
            add a row to what `PASS` has to zip.
    """
    # `requires_grad` filtered to match `get_optimiser`, which skips frozen
    #     parameters entirely. Without it a frozen tensor could be moved *into*
    #     the optimiser by this function, which is the opposite of what
    #     regrouping is for. `ViT2` has ten of them -- the blocks' rotary
    #     `freqs` -- and they are 1-d, so today the exemption catches them
    #     first; the filter is here so that stays true of a 2-d one.
    selected = [
        p for name, p in module.named_parameters()
        if p.requires_grad and not exempt(name, p)
    ]

    if not selected:
        return optimiser

    return _regroup(optimiser, selected, lr)


def mup_width(module):
    """
    The fan-in the muP rule is keyed on, or None if the module is out of scope.

    `d_model` is the width every transformer-shaped module in this repo states,
        and it is exact for the attention projections that dominate the
        parameter count. Reading it as an attribute rather than from the config
        is deliberate: several of these widths are *derived* --
        `SenderTransformerLM` and `AttentionPrototyper` take theirs from the
        vision model's `final_feat_dim` -- so a config key would be a second
        statement of the same number, able to be wrong.

    Absent `d_model` means out of scope, and three things fall out that way, all
        of them correctly.

        The convolutional backbones. `ResNet18.final_feat_dim` is a hardcoded
        512 with no width to vary, muP's rules are stated for transformers, and
        a ResNet at 1e-4 is what jayelm tuned.

        `AveragePrototyper` and `nn.Embedding`, which have no matrices between
        widths at all.

        `BilinearDiscriminator`, which reads nothing from its config table: its
        single tensor is `nn.Linear(message_width, referent_embedding_size)`,
        whose fan-in is the language model's `output_size`. With the listener
        GRU restored to jayelm's 1024 that fan-in *is* the reference width, so
        the factor would be 1.0 and the group would be a group at base rate.

    Args:
        module: a constructed submodule of the pair

    Returns:
        Its width as an int, or None.
    """
    return getattr(module, "d_model", None)


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


# `(name, selector, width_fn)`, the modules the muP learning-rate rule applies
#     to. Shaped after `train.py`'s `CLIP_GROUPS` and read the same way -- pick
#     the module off the constructed pair rather than name its parameters --
#     but note this is not that list. `CLIP_GROUPS` omits `sender.contrast`,
#     which falls to its `other` group; here the stage has a width of its own
#     and wants a rate to match. `receiver.token_embedding` is the other
#     difference and goes the other way: it is in `CLIP_GROUPS` and is out of
#     scope here, because an embedding table has no fan-in to scale by.
#
# A selector may return None -- `sender.contrast` is `None` when the stage is
#     off -- and `mup_width` may return None for a module with no `d_model`.
#     Both mean "leave this module at the base rate".
MUP_MODULES = (
    ("sender_vision", lambda pair: pair.sender.feat_model, mup_width),
    ("sender_prototyper", lambda pair: pair.sender.prototyper, mup_width),
    ("sender_contrast", lambda pair: pair.sender.contrast, mup_width),
    ("sender_language_model", lambda pair: pair.sender.language_model, mup_width),
    ("receiver_vision", lambda pair: pair.receiver.feature_model, mup_width),
    (
        "receiver_language_model",
        lambda pair: pair.receiver.language_model,
        mup_width,
    ),
    (
        "receiver_discriminator",
        lambda pair: pair.receiver.discriminator,
        mup_width,
    ),
)


def resolve_mup_learning_rates(pair, base_lr, reference_width):
    """
    Apply `lr(module) = base_lr * reference_width / fan_in(module)` across
        `MUP_MODULES`.

    One rate per module keyed on `d_model`, not one per tensor keyed on that
        tensor's own fan-in. Exact for the attention projections, which are
        square in `d_model` and are most of the parameters; approximate for the
        feedforward inner layers and for adapters that read a foreign width. The
        rule is a heuristic being applied at module granularity, and saying so
        is cheaper than a per-tensor scheme nobody can check by eye.

    Note what muP does and does not promise here. It transfers a tuned learning
        rate across a change of *width* in one architecture. Two of the changes
        this is being asked to cover are not that: the speaker's language model
        goes GRU -> Transformer as well as 1024 -> 320, and `ViT2` replaces a
        ResNet that has no width to speak of. Principled heuristic, not
        transfer guarantee.

    Args:
        pair: the constructed `base.Pair`
        base_lr: `[optimiser] lr`, the rate the reference width was tuned at
        reference_width: `[optimiser] mup_reference_width`

    Returns:
        A list of `(name, module, lr)` for the modules in scope, and a flat
            `{name: lr}` covering every module that was *built*, in-scope or
            not, for `save_args` to record. The second is a superset of the
            first: an exempt module reports the base rate, which is what its
            parameters will actually be trained at.
    """
    in_scope = []
    resolved = {}

    for name, select, width_of in MUP_MODULES:
        module = select(pair)

        if module is None:
            continue

        width = width_of(module)

        if width is None:
            resolved[name] = base_lr
            continue

        lr = base_lr * reference_width / width

        in_scope.append((name, module, lr))
        resolved[name] = lr

    return in_scope, resolved


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

    # muP first. It claims whole modules and `SPLIT_LEARNING_RATES` claims lone
    #     named tensors, and `is_mup_exempt` keeps the two disjoint by
    #     construction rather than by ordering -- every scalar in that table is
    #     0-d and `polarity_embedding` matches "embedding". Running muP first
    #     anyway, so that if the exemption rule is ever loosened the elevated
    #     scalar rates are the ones that survive.
    mup_in_scope, resolved_mup_lrs = resolve_mup_learning_rates(
        pair, base_lr, config['optimiser']['mup_reference_width']
    )
    
    for name, module, lr in mup_in_scope:
        if lr != base_lr:
            optimiser = split_out_module(optimiser, module, lr, name)
    
    # Written back so `save_args` records what was *built*. Several of these
    #     widths are derived rather than declared, so a config key is not
    #     evidence the module was built that way -- see docs/training.md. Note
    #     the rate here is the one the module's matrices got: its biases, norms,
    #     scalars and embedding tables stayed at the base rate, and any scalar
    #     named below moved again.
    config['optimiser']['resolved_mup_lrs'] = resolved_mup_lrs

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
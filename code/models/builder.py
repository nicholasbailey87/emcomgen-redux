"""
Model building logic
"""

from . import base
from . import sender as sender
from . import receiver as receiver

from .backbone import vision
from .model_util import ReferentAdapter

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


# `(name, selector)` for the modules that get a gradient-clipping group and a
#     learning rate of their own. The module is picked off the constructed pair
#     rather than named through its parameters, because "every tensor in
#     `sender.language_model`" is not expressible as a suffix and an attribute
#     lookup cannot be quietly broken by a rename the way a suffix match can.
#
# A selector may return None -- `sender.contrast` is `None` when the stage is
#     off -- which means the group does not exist on this pair. Nothing else
#     built so far is optional.
#
# One table serving two mechanisms that used to disagree. `train.py` had its own
#     `CLIP_GROUPS`, which omitted `sender.contrast` and so clipped the whole
#     stage under the `other` catch-all -- on every rung with the stage on,
#     `other` *was* the contrast stage under a misleading name. `MUP_MODULES`
#     was a second list, which included the contrast stage and omitted the
#     listener's embedding table. Neither was derivable from the other and
#     nothing held them in step, so adding a module to one was a silent partial
#     change. Add a module here and it is clipped and rateable at once.
MODULE_GROUPS = (
    ("sender_vision", lambda pair: pair.sender.feat_model),
    # Its own group rather than folded into the vision model it follows. The
    #     adapter is the one stage whose *input* width is the backbone's and
    #     whose output width is the language model's, so a gradient norm taken
    #     across the pair of them would be read as the backbone's and is not.
    #     It is also the stage a backbone swap changes the shape of, which is
    #     exactly what a per-module clip column is for.
    ("sender_adapter", lambda pair: pair.sender.adapter),
    ("sender_prototyper", lambda pair: pair.sender.prototyper),
    ("sender_contrast", lambda pair: pair.sender.contrast),
    ("sender_language_model", lambda pair: pair.sender.language_model),
    ("receiver_vision", lambda pair: pair.receiver.feature_model),
    ("receiver_adapter", lambda pair: pair.receiver.adapter),
    ("receiver_token_embedding", lambda pair: pair.receiver.token_embedding),
    ("receiver_language_model", lambda pair: pair.receiver.language_model),
    ("receiver_discriminator", lambda pair: pair.receiver.discriminator),
)


# `(name, applies to)` for the scaling scalars, each of which is a clipping
#     group to itself. The name is also the `named_parameters` suffix: there is
#     one tensor per name, it is 0-d, and the group is that one tensor.
#
# Why they are not left inside their modules. `clip_grad_norm_` takes one norm
#     across a whole group and scales every member by one factor, so a scalar
#     sharing a group with a thousand matrices is renormalised by *their* norm.
#     At recorded speaker norms of ~10 against `clip_grad_norm = 1.0` that is a
#     tenfold attenuation, applied on every step that binds, to a parameter
#     whose whole travel is already bounded by `lr * steps` -- and these are the
#     parameters the run's ignition waits on. Alone in a group, a scalar is
#     clipped by its own magnitude or not at all.
#
# Why these four and not the other two. These are the scalars that *scale*
#     something: a score volume, a mixing weight, a gate, and the speaker's
#     channel. `log_logit_scale` is the last of those and needs the group for
#     exactly the reason above: it is one 0-d tensor that would otherwise sit
#     inside `sender_language_model`, and be renormalised against a whole
#     module's norm.
#     `score_bias` is an offset and `polarity_embedding` is a 2-d tag, and both
#     belong to the norm of the module producing the output they modify. Both
#     still take a rate of their own through `SPLIT_LEARNING_RATES` -- a clip
#     group and a learning rate are separate questions, and this table answers
#     only the first.
#
# The gate is on the architecture rather than on finding the parameter, exactly
#     as `SPLIT_LEARNING_RATES`'s is: a `BilinearDiscriminator` has no mixing
#     weight and a speaker without the contrast stage has no gate, so for those
#     the group is inapplicable rather than missing, and `group_parameters`
#     raises if an applicable one matches nothing.
SCALAR_GROUPS = (
    # Both of these are now gated on a config flag as well as on the
    #     architecture. `[receiver_discriminator] normalise_score = false`
    #     leaves the listener with no volume and `[sender_language_model]
    #     normalise_logits = false` leaves the speaker with no channel scale,
    #     so on those rungs the group is inapplicable rather than missing --
    #     the same distinction `mix_logit` and `contrast_gate` already make,
    #     read off the module that owns the parameter rather than off the
    #     config, so the gate and the parameter cannot disagree.
    (
        "log_score_scale",
        lambda pair: pair.receiver.discriminator.learns_score_scale,
    ),
    (
        "log_logit_scale",
        lambda pair: pair.sender.language_model.normalises_logits,
    ),
    (
        "mix_logit",
        lambda pair: isinstance(
            pair.receiver.discriminator, receiver.AttentionDiscriminator
        ),
    ),
    ("contrast_gate", lambda pair: pair.sender.contrast is not None),
)


SCALAR_SUFFIXES = tuple(name for name, _ in SCALAR_GROUPS)


# Every group name, in reporting order, with the catch-all last. `train.py`
#     reports one gradient-norm column per entry on every rung, NaN where the
#     group does not exist on that architecture, so that the metrics header
#     keeps its shape across a resume -- the same rule the contrast columns
#     follow.
GROUP_NAMES = (
    tuple(name for name, _ in MODULE_GROUPS) + SCALAR_SUFFIXES + ("other",)
)


def claimed_separately(name, parameter=None):
    """
    Whether `parameter` has a group of its own rather than belonging to its
        module's.

    The scaling scalars of `SCALAR_GROUPS`, and nothing else. One statement
        covering both mechanisms: they are clipped alone, and they keep whatever
        rate `SPLIT_LEARNING_RATES` gives them rather than inheriting their
        module's. A module group that also claimed them would clip them twice --
        once alone and once inside the module's norm, which they would inflate
        on the way -- and would make `contrast_gate_lr = lr` mean "follow the
        contrast stage" rather than the documented "no override".

    Args:
        name: the parameter's name, relative to whatever is being walked. Every
            one of these is 0-d with a distinctive name, so a suffix match is
            unambiguous at any depth.
        parameter: unused. Present so this can be passed as `split_out_module`'s
            `exclude`, which calls it with both.

    Returns:
        True if another group has it.
    """
    return name.endswith(SCALAR_SUFFIXES)


def group_parameters(pair):
    """
    Partition `pair`'s parameters across `SCALAR_GROUPS` and `MODULE_GROUPS`.

    The single definition of what is clipped together; `train.py`'s
        `clip_gradients` walks this, and so do the tests that assert the
        partition is total. The scalars are claimed first so that the module
        groups can be stated as "everything else in the module", which is what
        makes the exclusion one fact rather than two lists that have to agree.

    Args:
        pair: the constructed `base.Pair`

    Returns:
        A list of `(name, parameters)` covering every entry of `GROUP_NAMES`, in
            that order. A group that does not exist on this pair, or that exists
            and holds nothing, is an empty list rather than an omission --
            `AveragePrototyper` has no parameters at all, and that is not the
            same thing as a group having gone missing. The final `other` entry
            is whatever nothing claimed: anything in it is a module somebody
            added without adding it here, and its presence is the alarm rather
            than the fix.

    Raises:
        RuntimeError: if a scalar group applies to this pair but matches no
            parameter, or if two module groups claim the same tensor. Both would
            otherwise be silent -- the first clips and steps a scalar with its
            module after all, the second clips and steps a parameter twice --
            and both would read as an architecture result.
    """
    groups = {name: [] for name in GROUP_NAMES}
    owner = {}

    for name, applies_to in SCALAR_GROUPS:
        if not applies_to(pair):
            continue

        selected = [p for n, p in pair.named_parameters() if n.endswith(name)]

        if not selected:
            raise RuntimeError(
                f"`{name}` applies to this pair but no parameter is named for "
                "it, so it would be clipped and stepped with its module "
                "instead. Has the parameter been renamed?"
            )

        groups[name] = selected
        owner.update({id(p): name for p in selected})

    for name, select in MODULE_GROUPS:
        module = select(pair)

        if module is None:
            continue

        selected = []

        for parameter_name, parameter in module.named_parameters():
            if claimed_separately(parameter_name, parameter):
                continue

            held_by = owner.get(id(parameter))

            if held_by is not None:
                raise RuntimeError(
                    f"`{parameter_name}` is in both `{held_by}` and `{name}`. "
                    "The groups must partition the pair: a parameter in two of "
                    "them is clipped twice and, if the two rates differ, "
                    "stepped twice."
                )

            owner[id(parameter)] = name
            selected.append(parameter)

        groups[name] = selected

    groups["other"] = [p for p in pair.parameters() if id(p) not in owner]

    return [(name, groups[name]) for name in GROUP_NAMES]


def split_out_module(optimiser, module, lr, config_key,
                     exclude=claimed_separately):
    """
    Move every parameter of `module` that no other group has claimed into a
        parameter group of its own at `lr`.

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
        exclude: `(name, parameter) -> bool`, called with names relative to
            `module`. Defaults to `claimed_separately`, which holds back the
            scaling scalars so this and `group_parameters` take the same
            view of what a module group contains.

    Returns:
        The same optimiser, mutated in place. Unchanged if every parameter of
            `module` is excluded or frozen -- a module that is nothing but a
            gate is a legitimate thing to ask about, and an empty group would
            only add a row to what `PASS` has to zip.
    """
    # `requires_grad` filtered to match `get_optimiser`, which skips frozen
    #     parameters entirely. Without it a frozen tensor could be moved *into*
    #     the optimiser by this function, which is the opposite of what
    #     regrouping is for. `ViT2` has ten of them -- the blocks' rotary
    #     `freqs`.
    selected = [
        p for name, p in module.named_parameters()
        if p.requires_grad and not exclude(name, p)
    ]

    if not selected:
        return optimiser

    return _regroup(optimiser, selected, lr)


# `(config key, parameter suffix, applies to)`, in the order the groups are
#     added. The test gates on the architecture rather than on finding the
#     parameter: a GRU speaker has no polarity tag, and
#     a `BilinearDiscriminator` has no mixing weight, so for those the key is
#     inapplicable rather than broken. See docs/training.md -- and do not read a
#     gate as a verdict on the parameter.
SPLIT_LEARNING_RATES = (
    (
        "polarity_embedding_lr",
        "polarity_embedding",
        lambda pair: isinstance(
            pair.sender.language_model, sender.SenderTransformerLM
        ),
    ),
    (
        # The listener's volume: a lone scalar in front of a normalised
        #     quantity, whose whole travel is bounded by `lr * steps` and which
        #     has to be able to move within a run. The speaker's channel is its
        #     counterpart, below, under `logit_scale_lr`, and the two share a
        #     rate because they are the same kind of parameter doing the same
        #     job at opposite ends of the channel. See docs/channel.md.
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
        #
        # Inapplicable, not broken, under `normalise_score = false`: there is no
        #     volume for it to move. The key stays live and simply has no
        #     effect, exactly as `mix_logit_lr` does on a bilinear listener.
        "score_scale_lr",
        "log_score_scale",
        lambda pair: pair.receiver.discriminator.learns_score_scale,
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
        #     no key at all and so sat at the base rate -- at today's 5e-5 and
        #     the 156.25 steps an epoch both datasets run, that would bound its
        #     whole travel at 0.23 over thirty epochs, against a score whose own
        #     opening spread is 0.577.
        #
        # The *same* condition as `score_scale_lr` above rather than one of its
        #     own: `learns_score_scale` is a misnomer and gates the offset too,
        #     by design -- an inner constant is annihilated by nothing and would
        #     simply be degenerate with the outer one -- so the one attribute
        #     governs both parameters' existence. Note `score_bias` is not a
        #     `SCALAR_GROUPS` entry: it belongs to its module's clip norm and
        #     takes only a separate rate, so this is the one gate it needs.
        "score_bias_lr",
        "score_bias",
        lambda pair: pair.receiver.discriminator.learns_score_scale,
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
        #     cannot travel further than `lr * steps` -- 156.25 steps an epoch
        #     on both datasets since 2026-08-31 -- so without this the stage
        #     would stay shut for most of a run. See `sender.ExampleContrast`.
        "contrast_gate_lr",
        "contrast_gate",
        lambda pair: pair.sender.contrast is not None,
    ),
    (
        # The speaker's channel scale, and the counterpart of `score_scale_lr`
        #     above: both are lone scalars sitting in front of a normalised
        #     quantity, both reach the loss through
        #     `model_util.scale_without_attenuating`, and both share a rate for
        #     that reason. It opens at 1.0 and is bounded above at
        #     `sender.MAX_LOGIT_SCALE` by `train.py`'s projection, which is not a
        #     reason to slow it down: the point of projecting rather than
        #     clamping is that sitting at the bound costs nothing and leaving it
        #     is free.
        #
        # On every architecture: both speakers mix in `GumbelChannel`, so unlike
        #     `polarity_embedding_lr` and `mix_logit_lr` no *speaker* lacks the
        #     parameter.
        #
        # There is one arm that lacks it, and it is a config setting rather
        #     than an architecture: `normalise_logits = false` removes the
        #     normaliser and the scale together, since a multiplier on a
        #     quantity that is not pinned to unit variance says nothing
        #     `outputs2vocab` was not already free to say.
        "logit_scale_lr",
        "log_logit_scale",
        lambda pair: pair.sender.language_model.normalises_logits,
    ),
)


def resolve_module_learning_rates(config, pair, base_lr):
    """
    Read one learning rate per module group out of `[optimiser.module_lr]`.

    Rates are stated rather than computed. The rule that used to compute them
        multiplied `base_lr` by `reference_width / d_model / layers`; the width
        half was muP and the depth half was a heuristic with no parametrisation
        behind it, and reinstating the width half alone broke rungs 9 and 10
        outright. What that rule actually bought was not a principled exponent
        but the only mechanism in the codebase giving different modules
        different rates. That is worth keeping; the derivation was not. See
        docs/training.md.

    Absent key means `base_lr`, and `parse_config.validate_config` rejects a key
        that names no group -- so a typo raises rather than quietly leaving a
        module at base, which is the failure `split_out_parameter` already
        guards against for the scalars.

    Args:
        config: the parsed config, for `[optimiser] module_lr`
        pair: the constructed `base.Pair`
        base_lr: `[optimiser] lr`

    Returns:
        A list of `(name, module, lr)` for the groups whose rate differs from
            `base_lr` and so need a group of their own, and a flat `{name: lr}`
            covering every module group that was *built*, moved or not, for
            `save_args` to record. The second is a superset of the first: a
            module left at base reports the base rate, which is what its
            parameters are trained at.
    """
    rates = config['optimiser'].get('module_lr') or {}

    to_split = []
    resolved = {}

    for name, select in MODULE_GROUPS:
        module = select(pair)

        if module is None:
            continue

        lr = rates.get(name, base_lr)
        resolved[name] = lr

        if lr != base_lr:
            to_split.append((name, module, lr))

    return to_split, resolved


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
    # Every stage after the backbone is sized from the adapter's output rather
    #     than from `final_feat_dim`, which is the whole point of it: the
    #     speaker runs at its language model's `d_model` and the vision model
    #     emits whatever it emits. See `model_util.ReferentAdapter`.
    sender_referent_width = config['sender_language_model']['d_model']
    sender_adapter = ReferentAdapter(
        sender_feature_model.final_feat_dim,
        sender_referent_width,
        activation=config['sender_language_model']['activation'],
    )
    sender_prototyper = sender_prototyper_class(sender_referent_width)
    sender_language_model = sender_language_model_class(
        sender_referent_width,
        **config['sender_language_model']
    )

    # A boolean rather than a class name, because there is one of these or there
    #     is nothing: the stage is a residual on the referents, so "off" is
    #     `None` and not another module. `False` builds the speaker that existed
    #     before it, and `True` opens at that speaker exactly -- see
    #     `sender.ExampleContrast`.
    sender_contrast = (
        sender.ExampleContrast(
            sender_referent_width,
            **config['sender_contrast']
        )
        if config['sender']['contrast']
        else None
    )

    sender_ = sender_class(
        feat_model = sender_feature_model,
        adapter = sender_adapter,
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

    receiver_referent_width = config['receiver_language_model']['d_model']
    receiver_adapter = ReferentAdapter(
        receiver_feature_model.final_feat_dim,
        receiver_referent_width,
        activation=config['receiver_language_model']['activation'],
    )
    receiver_language_model = receiver_language_model_class(
        receiver_referent_width,
        **config['receiver_language_model']
    )

    # The discriminator is sized from the language model rather than from the
    #     config: which width the message arrives at is the encoder's business
    #     -- `2 * d_model` for a bidirectional GRU, `d_model` for the decoder
    #     stack -- and a config key restating it is a key that can be wrong.
    receiver_discriminator = receiver_discriminator_class(
        receiver_referent_width,
        receiver_language_model.output_size,
        **config['receiver_discriminator']
    )

    receiver_ = receiver_class(
        feature_model = receiver_feature_model,
        adapter = receiver_adapter,
        token_embedding_module=receiver_token_embedding_module,
        language_model = receiver_language_model,
        discriminator = receiver_discriminator,
        dropout = config['receiver']['dropout'],
    )

    pair = base.Pair(sender_, receiver_)

    if config['cuda']:
        pair = pair.cuda()
    
    # `eps` is passed rather than left at `get_optimiser`'s 1e-8 default, and
    #     `add_param_group` fills it into every group `split_out_*` adds below.
    #     See `[optimiser] eps` in DEFAULT.toml for why it is far smaller here.
    optimiser = get_optimiser(
        pair,
        config['sender_language_model']['d_model'],
        lr=config['optimiser']['lr'],
        weight_decay=config['optimiser']['weight_decay'],
        eps=config['optimiser']['eps'],
    )

    base_lr = config['optimiser']['lr']

    # Module groups first. They claim whole modules and `SPLIT_LEARNING_RATES`
    #     claims lone named tensors, and the two are disjoint by construction
    #     rather than by ordering: `claimed_separately` holds back the four
    #     scaling scalars here as well as in `group_parameters`, so the only
    #     tensors both loops can reach are `score_bias` and
    #     `polarity_embedding`, which are deliberately in their module's clip
    #     group and take their rate from the key below. Running the modules
    #     first means those two end up at the rate their own key names.
    module_lrs, resolved_module_lrs = resolve_module_learning_rates(
        config, pair, base_lr
    )

    for name, module, lr in module_lrs:
        optimiser = split_out_module(optimiser, module, lr, name)

    # Written back so `save_args` records what was *built*. `sender_contrast`
    #     is absent when the stage is off, so this says which groups existed as
    #     well as what rate each ran at. Note the rate here is the one the
    #     module's own parameters got: the scalars that clip separately kept
    #     theirs, and `score_bias` and `polarity_embedding` move again below.
    config['optimiser']['resolved_module_lrs'] = resolved_module_lrs

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
from pathlib import Path
import warnings
import toml
import copy

class InvalidConfig(Exception):
    pass


# `[scheduler] lr_schedule_shape` -> the `gradboard.cycles.FN_LIBRARY` name the
#     post-warm-up cycle runs, and whether that shape reads
#     `cool_point_multiplier`.
#
# The config names an *intention*, not a curve. It used to name the curve --
#     any key of `FN_LIBRARY` -- and two of those keys, `ascent` and `triangle`,
#     open at their trough rather than their peak, so putting a warm-up in front
#     of one threw the warm-up's rate away at the handover and re-climbed it over
#     the rest of the run. Nothing in the config said so. Restricting the surface
#     to shapes that open at their peak makes a warm-up compose continuously with
#     whatever follows it, by construction rather than by the reader knowing
#     which curves are safe.
#
# `flat` is `ascent` pinned at `low = high = 1.0`, which is constant at the base
#     rate whatever the generating function does. It takes no floor: there is
#     nothing for a floor to mean when the rate never descends, and a
#     `cool_point_multiplier` sitting unread beside it is how the ten-epoch
#     warm-up came to spend three weeks doing nothing. `validate_config` rejects
#     the combination rather than ignoring it.
#
# `cosine` is `half_cosine`, the falling half -- 1.0 at step zero down to 0.0 at
#     the last step, mapped onto `[cool_point_multiplier, 1.0]`. `FN_LIBRARY`'s
#     own `cosine` is the *full* period, 1.0 down to 0.0 and back up to 1.0, which
#     would end a run at its opening rate. That is a cosine restart and not what
#     "cosine schedule" means anywhere else, so the sentinel maps to the half.
LR_SCHEDULE_SHAPES = {
    "flat": ("ascent", False),
    "cosine": ("half_cosine", True),
}

def recursive_update(store: dict, items: dict) -> dict:
    """
    Update `store` in place with `items`, merging recursively where both hold a
      dict at the same key.
    """
    for k, v in items.items():
        if (k in store) and isinstance(store[k], dict):
            if isinstance(v, dict):
                recursive_update(store[k], items[k])
        else:
            store[k] = v

def parse_toml(filepath: str) -> dict:
    """
    Parse a toml file, e.g. containing the configuration for an experiment.
    """

    with open(str(Path(filepath)), 'r') as f:
        return toml.load(f)

class SafeDict(dict):
    """
    A default dict that raises warnings when keys are absent.
    """
    def __init__(self):
        super().__init__()
    def __missing__(self, key):
        self[key] = None
        warnings.warn(
            f"The config doesn't contain {key}. Defaulting to None."
        )
        return self[key]

def validate_config(config: dict) -> bool:
    """
    Check that the config doesn't contradict itself and has the necessary
    arguments. See docs/training.md.
    """

    if config['use_lang'] and (config['copy_receiver'] or config['receiver_only']):
        raise InvalidConfig(
            "`use_lang` must be false if `copy_receiver` or `receiver_only` is true."
        )

    if config['copy_receiver'] and config['receiver_only']:
        raise InvalidConfig(
            "`copy_receiver` not allowed with `receiver_only`"
        )

    if config['reference_game_xent'] and not config['reference_game']:
        raise InvalidConfig(
            "reference_game_xent=true requires reference_game=true"
        )

    # There is no joint-training objective in this codebase.
    if config['joint_training']:
        raise InvalidConfig(
            "`joint_training` is not implemented and must be false."
        )
    
    if (
        config['sender_language_model']['message_length']
        !=
        config['receiver_language_model']['message_length']
    ):
        raise InvalidConfig(
            "`sender_language_model` message length must be the same as "
            "`receiver_language_model` message length."
        )
    
    # The channel and readout flags, checked here for the same reason and in
    # the same style. All three default to today's behaviour in the modules that
    # read them, so a missing key would run the default arm silently rather than
    # failing -- which is precisely the confusion an experiment folder whose
    # whole treatment is one of these keys cannot survive. Required and
    # boolean, so a `"false"` string cannot read as true either.
    #
    # `scale_score` and `bias_score` replace a single `normalise_score`. That
    # key also gated the `1/sqrt(d)` calibration, which is now unconditional --
    # it is what makes the listener's score open at a stated number rather than
    # a width-dependent one, so it is design and not a rung. What remains
    # configurable is the two scalars of the readout, and they are two keys
    # because they answer two questions: a loudness and a threshold.
    for table, key in (
        ('sender_language_model', 'normalise_logits'),
        ('receiver_discriminator', 'scale_score'),
        ('receiver_discriminator', 'bias_score'),
    ):
        value = config[table].get(key)
        if not isinstance(value, bool):
            raise InvalidConfig(
                f"`{table}.{key}` must be present and a boolean — got "
                f"{value!r}."
            )

    # Keys that no longer exist, rejected by name rather than ignored.
    #
    # Nothing else in this file needs a check like this, because every live key
    # is read by the module that names it and a typo shows up as a missing
    # setting. A *retired* key is the opposite failure: it sits in a config
    # looking like a treatment while the code reads straight past it, and the
    # config still validates. `experiments/silhouette_titration_norms/` is the
    # concrete case -- ten of its fifteen cells set `normalise_score = false`
    # and are named for it, and without this they would run the default arm
    # under a filename saying they did not.
    #
    # Retiring a key means putting it here, with what to do instead. Removing an
    # entry once nothing in the repo names it is fine; the entry is a migration
    # aid, not a permanent record. That is what docs/ is for.
    for table, key, guidance in (
        (
            'receiver_discriminator',
            'normalise_score',
            "it gated the operand norms, the `1/sqrt(d)` calibration and both "
            "readout scalars, and the first two are unconditional now. Use "
            "`scale_score` and `bias_score` -- one per scalar. There is no "
            "setting that removes the calibration",
        ),
        (
            'sender_language_model',
            'estimator',
            "the straight-through Gumbel-softmax estimator is unconditional "
            'since 2026-09-07. `"identity"` lost `lr_sweep_1_cnn` on every arm '
            "of both datasets and the branch is gone, so a config naming "
            "either value would be describing a choice that no longer exists. "
            "See `sender.sample_symbols`",
        ),
    ):
        if key in config.get(table, {}):
            raise InvalidConfig(
                f"`{table}.{key}` no longer exists: {guidance}. See "
                "DEFAULT.toml beside the keys that replaced it."
            )

    # `[optimiser.module_lr]`, one rate per module clip group. Checked here
    # rather than in `build_models` because the whole point of the check is that
    # a key naming no group must *raise*: an unknown key would otherwise sit in
    # the config looking like a setting while the module it was meant for ran at
    # base rate, which is the silent failure `split_out_parameter` already
    # guards against for the scalars. Absent keys are fine and mean base rate.
    #
    # Imported here rather than at module scope so that parsing a config does
    # not pull in torch by way of `models`.
    from models.builder import GROUP_IMPLEMENTATION, MODULE_GROUPS

    group_names = {name for name, _ in MODULE_GROUPS}
    module_lr = config['optimiser'].get('module_lr') or {}

    if not isinstance(module_lr, dict):
        raise InvalidConfig(
            "`optimiser.module_lr` must be a table of group name -> learning "
            f"rate, got {type(module_lr).__name__}."
        )

    for key, rate in module_lr.items():
        if key not in group_names:
            raise InvalidConfig(
                f"`optimiser.module_lr.{key}` names no clip group. The groups "
                f"are {', '.join(sorted(group_names))} — see "
                "`models.builder.MODULE_GROUPS`."
            )

        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or rate <= 0
        ):
            raise InvalidConfig(
                f"`optimiser.module_lr.{key}` must be a positive number, got "
                f"{rate}."
            )

    # `[optimiser.implementation_lr]`, one rate per (module group, implementing
    # class). Checked in the same place and for the same reason as `module_lr`
    # above, with one extra failure to catch: a group named here that has no
    # choice of implementation. Setting
    # `implementation_lr.sender_adapter.LinearInterface` would look like a
    # setting and do nothing, because `resolve_module_learning_rates` only
    # consults this table for the six groups `GROUP_IMPLEMENTATION` covers.
    #
    # Class names are deliberately *not* checked. There is no registry to check
    # against -- `build_models` resolves them with `getattr` on the module -- and
    # the table's whole purpose is to hold rates for classes that no current
    # rung runs, so "unused" cannot mean "wrong" here. See
    # `models.builder.GROUP_IMPLEMENTATION`.
    implementation_lr = config['optimiser'].get('implementation_lr') or {}

    if not isinstance(implementation_lr, dict):
        raise InvalidConfig(
            "`optimiser.implementation_lr` must be a table of group name -> "
            "class name -> learning rate, got "
            f"{type(implementation_lr).__name__}."
        )

    for group, rates_by_class in implementation_lr.items():
        if group not in GROUP_IMPLEMENTATION:
            detail = (
                "it names a clip group, but one whose implementation is not "
                "chosen in the config, so a rate here could never be found"
                if group in group_names
                else "it names no clip group at all"
            )
            raise InvalidConfig(
                f"`optimiser.implementation_lr.{group}` is not keyable: "
                f"{detail}. The keyable groups are "
                f"{', '.join(sorted(GROUP_IMPLEMENTATION))} — see "
                "`models.builder.GROUP_IMPLEMENTATION`. Groups without a "
                "choice take their rate from `optimiser.module_lr`."
            )

        if not isinstance(rates_by_class, dict):
            raise InvalidConfig(
                f"`optimiser.implementation_lr.{group}` must be a table of "
                f"class name -> learning rate, got "
                f"{type(rates_by_class).__name__}."
            )

        for implementation, rate in rates_by_class.items():
            if (
                not isinstance(rate, (int, float))
                or isinstance(rate, bool)
                or rate <= 0
            ):
                raise InvalidConfig(
                    f"`optimiser.implementation_lr.{group}.{implementation}` "
                    f"must be a positive number, got {rate}."
                )

    # `[scheduler]`. The shape is a sentinel rather than a `FN_LIBRARY` name --
    # see `LR_SCHEDULE_SHAPES` for why the surface is this narrow -- and the
    # floor is required by exactly the shapes that descend.
    shape = config['scheduler'].get('lr_schedule_shape')

    if shape not in LR_SCHEDULE_SHAPES:
        raise InvalidConfig(
            f"`scheduler.lr_schedule_shape` must be one of "
            f"{', '.join(sorted(LR_SCHEDULE_SHAPES))}, got {shape!r}. These are "
            "intentions rather than curve names; see "
            "`parse_config.LR_SCHEDULE_SHAPES`."
        )

    _, takes_floor = LR_SCHEDULE_SHAPES[shape]
    floor = config['scheduler'].get('cool_point_multiplier')

    if takes_floor:
        if (
            not isinstance(floor, (int, float))
            or isinstance(floor, bool)
            or not 0.0 <= floor < 1.0
        ):
            raise InvalidConfig(
                f"`scheduler.lr_schedule_shape = {shape!r}` descends, so "
                "`scheduler.cool_point_multiplier` must be present and in "
                f"[0, 1) -- the fraction of the base rate it descends to -- got "
                f"{floor!r}. A floor of 1.0 would be a flat schedule; ask for "
                'that with `lr_schedule_shape = "flat"`.'
            )
    elif floor is not None:
        raise InvalidConfig(
            f"`scheduler.lr_schedule_shape = {shape!r}` does not descend, so "
            f"`scheduler.cool_point_multiplier` ({floor!r}) would never be "
            "read. Remove it. It is rejected rather than ignored because a "
            "scheduler key that looks set and is not is how the ten-epoch "
            "warm-up ran on no rung for three weeks."
        )

    warm_up_epochs = config['scheduler'].get('warm_up_epochs')

    if (
        not isinstance(warm_up_epochs, int)
        or isinstance(warm_up_epochs, bool)
        or warm_up_epochs < 0
    ):
        raise InvalidConfig(
            "`scheduler.warm_up_epochs` must be a non-negative integer, got "
            f"{warm_up_epochs!r}."
        )

    # Both rates are a fraction of games, so they share a check.
    #     `silhouette_fill` used to share it too, back when it was a scalar
    #     fraction of maximum intensity; it is a per-channel colour now and a
    #     list here would raise `TypeError` rather than `InvalidConfig`.
    for key in ('silhouette_p_sender', 'silhouette_p_receiver'):
        p = config['data'][key]
        if not 0.0 <= p <= 1.0:
            raise InvalidConfig(f"`{key}` must be in [0, 1], got {p}.")

    degrees = config['data']['augment_affine_degrees']
    if not isinstance(degrees, (int, float)) or isinstance(degrees, bool):
        raise InvalidConfig(
            f"`augment_affine_degrees` must be a number, got {degrees!r}."
        )
    if not 0.0 <= degrees <= 45.0:
        # 45 is where a rotated square becomes a diamond. Nothing in this
        #     dataset labels one, but a rotation that large is a config error
        #     rather than an experiment, and the ceiling says where the
        #     transform stops being label-preserving in principle.
        raise InvalidConfig(
            f"`augment_affine_degrees` must be in [0, 45], got {degrees}."
        )

    if not isinstance(config['data']['augment_flip'], bool):
        raise InvalidConfig(
            "`augment_flip` must be a boolean, got "
            f"{config['data']['augment_flip']!r}."
        )

    alpha = config['data']['mixup_alpha']
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise InvalidConfig(f"`mixup_alpha` must be a number, got {alpha!r}.")
    if alpha < 0.0:
        # No upper bound to give. `Beta(a, a)` concentrates on 0.5 as `a`
        #     grows, which is a weaker augmentation rather than an invalid one.
        raise InvalidConfig(f"`mixup_alpha` must be >= 0, got {alpha}.")

    # A scalar is broadcast to three channels by `silhouette`, so both forms
    #     are accepted; anything else is a config that will not paint a colour.
    fill = config['data']['silhouette_fill']
    channels = fill if isinstance(fill, (list, tuple)) else [fill]
    if len(channels) not in (1, 3):
        raise InvalidConfig(
            "`silhouette_fill` must be a scalar or a length-3 [R, G, B] "
            f"sequence, got {len(channels)} elements: {fill!r}."
        )
    for c in channels:
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            raise InvalidConfig(
                f"`silhouette_fill` must be numeric, got {fill!r}."
            )
        if not 0.0 <= c <= 1.0:
            raise InvalidConfig(
                f"every channel of `silhouette_fill` must be in [0, 1], got "
                f"{fill!r}."
            )

    if 'dataset' not in config['data']:
        raise InvalidConfig(
            "Config TOML must specify ```\n['data']\ndataset = ...```."
        )

def get_config(
    filepath: str = None,
    defaults: str = str(Path(__file__).resolve().parents[1] / "DEFAULT.toml"),
):
    """
    Combine `DEFAULT.toml` with the user's experiment TOML. See
        docs/training.md for the resolution order.
    """

    defaults = parse_toml(defaults)

    active_defaults = {
        k: v for k, v in defaults.items()
        if k not in ['shapeworld', 'birds']
    }
    birds_defaults = defaults['birds']

    provisional_config = copy.deepcopy(active_defaults)

    if filepath is not None:
        custom_config = parse_toml(filepath)
        recursive_update(provisional_config, custom_config)
    else:
        custom_config = dict()
           
    # Decided by the dataset's *name*, not its location: `train.py` later
    # rewrites this to a fast-storage path.
    dataset_name = Path(provisional_config['data']['dataset']).name
    if dataset_name == 'cub':
        recursive_update(active_defaults, birds_defaults)
    elif dataset_name.startswith('shapeworld'):
        pass # Defaults are already correct for shapeworld
    else:
        raise InvalidConfig(
            f"Dataset must be named 'cub' or 'shapeworld*', got '{dataset_name}'."
        )
    
    actual_config = active_defaults
    recursive_update(actual_config, custom_config)

    safe_config = SafeDict()
    safe_config.update(actual_config)
    
    validate_config(safe_config)

    return safe_config
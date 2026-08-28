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
    
    # Checked here rather than in the speaker's constructor: `SafeDict` only
    # warns on a missing key and hands back None.
    init_energy = config['sender_language_model'].get('init_energy')
    if init_energy is None or not 0.0 < init_energy <= 1.0:
        raise InvalidConfig(
            "`sender_language_model.init_energy` must be present and in "
            f"(0, 1] — it is a fraction of maximum entropy, not a percentage "
            f"— got {init_energy}."
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
    from models.builder import MODULE_GROUPS

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

    for key in ('silhouette_p_sender', 'silhouette_p_receiver'):
        p = config['data'][key]
        if not 0.0 <= p <= 1.0:
            raise InvalidConfig(f"`{key}` must be in [0, 1], got {p}.")

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
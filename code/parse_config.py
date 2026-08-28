from pathlib import Path
import warnings
import toml
import copy

class InvalidConfig(Exception):
    pass

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
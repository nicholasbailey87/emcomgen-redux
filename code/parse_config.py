from collections import defaultdict
from pathlib import Path
import warnings
import toml
import copy

class InvalidConfig(Exception):
    pass

def recursive_update(store: dict, items: dict) -> dict:
    """
    Takes two dicts and updates the first with the contents of the second,
      merging the values of any keys whose values are dictionaries in
      both `store` and `items`
    
    Args:
      store: the dictionary to be updated
      items: the dictionary providing the updates
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
    Check that the config doesn't contradict itself and has the necessary arguments.

    Based on some lines in code/io_util.py
    """

    if (
        # Remembering that True == 1 in Python...
        config['force_reference_game']
        + config['force_concept_game']
        + config['force_setref_game']
     ) > 1:
        raise InvalidConfig(
            "Only one of the following can be true: `force_reference_game`,"
            " `force_concept_game`, `force_setref_game`."
        )
        
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
    
    if (
        config['sender_language_model']['message_length']
        !=
        config['receiver_comparer']['message_length']
    ):
        raise InvalidConfig(
            "`sender_language_model` message length must be the same as "
            "`receiver_comparer` message length."            
        )
    
    if 'dataset' not in config['data']:
        raise InvalidConfig(
            "Config TOML must specify ```\n['data']\ndataset = ...```."
        )

def get_config(filepath: str = None, defaults: str = "../config/DEFAULT.toml"):
    """
    This function gets a custom experiment config by combining some default
        parameters with a TOML file specified by the user.

    First get the plain defaults and the dataset-specific defaults.

    Then create a provisional custom config by overwriting the generic
        default config with the provided custom config.
    
    Parse the required data set from the provisional custom config (we don't
        parse directly from the provided custom TOML file just in case it doesn't
        specify a data set) and create an updated default by overwriting
        parameters in the generic default with the parameters from the
        data-specific defaults.
    
    Create the final custom config by overwriting any parameters in the updated
        defaults that are also found in the custom TOML.
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
           
    if provisional_config['data']['dataset'] == '../data/cub':
        recursive_update(active_defaults, birds_defaults)
    elif provisional_config['data']['dataset'] == '../data/shapeworld':
        pass # Defaults are already correct for shapeworld
    else:
        raise InvalidConfig(
            "Dataset must be '../data/cub' or '../data/shapeworld'."
        )
    
    actual_config = active_defaults
    recursive_update(actual_config, custom_config)

    safe_config = SafeDict()
    safe_config.update(actual_config)
    
    validate_config(safe_config)

    return safe_config
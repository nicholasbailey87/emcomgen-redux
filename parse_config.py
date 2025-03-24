import toml
from collections import defaultdict

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

    with open(filepath, 'r') as f:
        return toml.load(f)

def get_config(filepath: str = None, defaults: str = "config/DEFAULT.toml"):

    defaults = parse_toml(defaults)

    config = {
        k: v for k, v in defaults.items()
        if k not in ['shapeworld', 'birds']
    }

    if filepath is not None:
        custom_config = parse_toml(filepath)
        recursive_update(config, custom_config)
    else:
        custom_config = dict()
    
    # TODO: make this part of a separate validate_config function that runs at the end of this parse function once the intended config has been resolved in the proper order of precedence
    if 'dataset' not in config['data']:
        raise InvalidConfig(
            "Config TOML must specify ```\n['data']\ndataset = ...```."
        )
        
    if config['data']['dataset'] == 'data/cub':
        birds_config = defaults['birds']
        recursive_update(birds_config, custom_config)
        recursive_update(config, birds_config)
    elif config['data']['dataset'] == 'data/shapeworld':
        shapeworld_config = defaults['shapeworld']
        recursive_update(shapeworld_config, custom_config)
        recursive_update(config, shapeworld_config)
    else:
        raise InvalidConfig(
            "Dataset must be 'data/cub' or 'data/shapeworld'."
        )

    recursive_update(config, custom_config)
    
    safe_config = defaultdict(lambda: None)
    safe_config.update(config)
    
    return safe_config

    

    

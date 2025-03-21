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

def get_config(filepath: str, defaults: str = "config/DEFAULT.toml"):

    defaults = parse_toml(defaults)
    config = parse_toml(filepath)
    
    if 'data_set' not in config:
        raise InvalidConfig('Config TOML must specify `data_set`.')

    data_agnostic_defaults = {
        k: v for k, v in defaults.items()
        if k not in ['shapeworld', 'birds']
    }

    if config['data_set'] == 'data/cub':
        recursive_update(data_agnostic_defaults, defaults['birds'])
    elif config['data_set'] == 'data/shapeworld':
        recursive_update(data_agnostic_defaults, defaults['shapeworld'])
    else:
        raise InvalidConfig(
            "Data set must be 'data/cub' or 'data/shapeworld'."
        )

    recursive_update(data_agnostic_defaults, config)
    
    safe_config = defaultdict(lambda: None)
    safe_config.update(config)
    
    return safe_config

    

    

"""
Model building utils
"""


import torch.nn as nn

from broccoli.activation import ReLU, GELU, SquaredReLU, SwiGLU


# Name -> broccoli activation, so that `activation` can be set from a TOML
#     config. This mirrors the map broccoli's own `ViT.__init__` applies to
#     string arguments, and deliberately offers the same four options.
#     `TransformerEncoder` has no such map and takes the class (or, for the
#     GLU variants, the factory) directly, so the lookup has to happen here.
ACTIVATIONS = {
    "ReLU": ReLU,
    "GELU": GELU,
    "SquaredReLU": SquaredReLU,
    "SwiGLU": SwiGLU,
}


def get_activation(name):
    """
    Look up a broccoli activation by name.

    Raises
    ------
    ValueError
        If `name` is not a known activation. A config typo should say what the
        valid options are, rather than surfacing as a bare KeyError from deep
        inside model construction.
    """
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Valid options: {', '.join(sorted(ACTIVATIONS))}."
        )


def reset_sequential(seq):
    for layer in seq:
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()

"""
Model building utils
"""


import torch.nn as nn


def reset_sequential(seq):
    for layer in seq:
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()

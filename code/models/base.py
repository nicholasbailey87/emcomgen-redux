"""
Combine sender and receiver for easier training
"""

from torch import nn


class Pair(nn.Module):
    def __init__(self, sender, receiver):
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.xent_criterion = nn.CrossEntropyLoss()

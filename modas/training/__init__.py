"""
MODAS Training Scripts

- train_v1: V1 sparse coding training
- train_a1: A1 sparse coding training
- train_atl: ATL binding training
"""

from modas.training.train_v1 import train_v1, V1Trainer
from modas.training.train_a1 import train_a1, A1Trainer
from modas.training.train_atl import train_atl, ATLTrainer

__all__ = [
    "train_v1",
    "train_a1",
    "train_atl",
    "V1Trainer",
    "A1Trainer",
    "ATLTrainer",
]

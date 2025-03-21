"""Initialization functions"""

import random

import numpy
import torch


def set_seed(seed: int = 0):
    """Set random seed for python random, numpy and torch."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)

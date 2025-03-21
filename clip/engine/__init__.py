"""Module with train and evaluation functions"""

from .train import train, train_iterations
from .evaluation import evaluate

__all__ = ["train", "train_iterations", "evaluate"]

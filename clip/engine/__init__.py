"""Module with train and evaluation functions"""

from .train import (
    train,
    train_iterations,
    train_iterations_embeds,
    train_reclip_iterations,
    train_fusion_iterations,
)
from .evaluation import (
    evaluate, 
    evaluate_fusion_lazy, 
    evaluate_embeddings
)

__all__ = [
    "train",
    "train_iterations",
    "train_iterations_embeds",
    "train_reclip_iterations",
    "train_fusion_iterations",
    "evaluate",
    "evaluate_fusion_lazy",
    "evaluate_embeddings",
]

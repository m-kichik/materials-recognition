"""CLIP contrastive loss from paper (https://arxiv.org/abs/2103.00020)"""

import torch
import torch.nn.functional as F


def clip_loss(
    image_features: torch.Tensor, text_features: torch.Tensor
) -> torch.Tensor:
    """
    Computes the contrastive loss for CLIP-style models.

    The loss encourages correct image-text pairs to have higher cosine similarity while penalizing incorrect pairs.
    It is computed as the average of two cross-entropy losses: one for image-to-text similarity
    and one for text-to-image similarity.

    Args:
        image_features (torch.Tensor): A tensor of shape (batch_size, feature_dim) representing image embeddings.
        text_features (torch.Tensor): A tensor of shape (batch_size, feature_dim) representing text embeddings.

    Returns:
        torch.Tensor: A scalar tensor representing the computed loss.
    """
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = image_features @ text_features.T
    labels = torch.arange(len(logits)).to(logits.device)

    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss

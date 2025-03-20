"""CLIP evaluation: cosine similarity, recall, MRR"""

import random
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    k_values: Tuple[int, ...] = (1, 5, 10),
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluates a model on an image-text retrieval task using cosine similarity.

    Args:
        model (torch.nn.Module): The model to be evaluated. It should support `encode_image` and `encode_text` methods.
        dataloader (DataLoader): DataLoader providing batches of (images, captions).
        k_values (Tuple[int, ...], optional): Tuple of K values for Recall@K computation. Defaults to (1, 5, 10).
        device (str, optional): Device to perform computations on ("cpu" or "cuda"). Defaults to "cpu".

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics:
            - Recall@K for each specified K
            - Mean Reciprocal Rank (MRR)
            - Mean Cosine Similarity
    """
    model.eval()

    all_ranks = []
    cosine_similarities = []

    with torch.no_grad():
        for images, captions in (pbar := tqdm(dataloader)):
            pbar.set_description("evaluation")
            images = images.to(device)
            if len(captions) == 2:
                captions = random.choice(captions)
            text_tokens = clip.tokenize(captions).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity_matrix = image_features @ text_features.T

            cosine_similarities.append(similarity_matrix.diag().cpu())

            ranks = similarity_matrix.argsort(descending=True, dim=-1)
            ground_truth = torch.arange(image_features.size(0), device=device)
            rank_positions = (ranks == ground_truth.unsqueeze(1)).nonzero()[:, 1] + 1

            all_ranks.append(rank_positions.cpu())

        all_ranks = torch.cat(all_ranks)

        cosine_similarities = torch.cat(cosine_similarities)

        # Compute Recall@K
        recalls = {
            f"Recall@{k}": (all_ranks <= k).float().mean().item() for k in k_values
        }

        # Compute Mean Reciprocal Rank (MRR)
        mrr = (1.0 / all_ranks.float()).mean().item()

        # Compute Mean Cosine Similarity
        mean_cosine_similarity = cosine_similarities.mean().item()

        metrics = {
            **recalls,
            "MRR": mrr,
            "Mean Cosine Similarity": mean_cosine_similarity,
        }

        return metrics

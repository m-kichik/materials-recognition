"""CLIP evaluation: cosine similarity, recall, MRR"""

import random
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    # pearsonr,
    # silhouette_score,
    # adjusted_rand_score,
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
@torch.inference_mode()
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
    all_times = []

    for batch in (pbar := tqdm(dataloader)):
        pbar.set_description("evaluation")

        images = batch["images"].to(device)
        captions = batch["captions"]

        start = time.perf_counter()
        image_features = model.encode_image(images)
        text_features = model.encode_text(captions)
        elapsed_time = time.perf_counter() - start
        all_times.append(elapsed_time)

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

        return metrics, np.mean(all_times)


@torch.no_grad()
@torch.inference_mode()
def evaluate_fusion_lazy(
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
    all_times = []

    with torch.no_grad():
        for batch in (pbar := tqdm(dataloader)):
            images = batch["images"].to(device)
            captions = batch["captions"]
            embeddings = batch["embeddings"]

            pbar.set_description("evaluation")
            images = images.to(device)
            embeddings = embeddings.to(device)
            if len(captions) == 2:
                captions = random.choice(captions)

            start = time.perf_counter()
            image_features = model.encode_image(images, embeddings)
            text_features = model.encode_text(captions)
            elapsed_time = time.perf_counter() - start
            all_times.append(elapsed_time)

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

        return metrics, np.mean(all_times)


# evaluation for text pretraining
def evaluate_extrinsic(
    embeddings: np.ndarray, C: np.ndarray, M: np.ndarray, k_knn: int = 5
) -> dict:
    """
    Extrinsic evaluation using classification and retrieval.

    embeddings: (N, D)
    C: (N, N_cat) one-hot categories
    M: (N, N_mat) multi-hot materials
    """
    metrics = {}
    N, D = embeddings.shape
    labels_cat = C.argmax(axis=1)
    labels_mat = M.argmax(axis=1)  # primary material for k-NN

    # Train/test split
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train, test = idx[:split], idx[split:]

    X_tr, X_te = embeddings[train], embeddings[test]
    y_cat_tr, y_cat_te = labels_cat[train], labels_cat[test]
    y_mat_tr, y_mat_te = labels_mat[train], labels_mat[test]

    # k-NN classification
    knn_cat = KNeighborsClassifier(n_neighbors=k_knn).fit(X_tr, y_cat_tr)
    pred_cat = knn_cat.predict(X_te)
    acc_cat = accuracy_score(y_cat_te, pred_cat)
    p_cat, r_cat, f1_cat, _ = precision_recall_fscore_support(
        y_cat_te, pred_cat, average="weighted", zero_division=0
    )

    knn_mat = KNeighborsClassifier(n_neighbors=k_knn).fit(X_tr, y_mat_tr)
    pred_mat = knn_mat.predict(X_te)
    acc_mat = accuracy_score(y_mat_te, pred_mat)
    p_mat, r_mat, f1_mat, _ = precision_recall_fscore_support(
        y_mat_te, pred_mat, average="weighted", zero_division=0
    )

    metrics.update(
        {
            "knn_acc_cat": acc_cat,
            "knn_precision_cat": p_cat,
            "knn_recall_cat": r_cat,
            "knn_f1_cat": f1_cat,
            "knn_acc_mat": acc_mat,
            "knn_precision_mat": p_mat,
            "knn_recall_mat": r_mat,
            "knn_f1_mat": f1_mat,
        }
    )

    # Retrieval mAP
    sim = X_te @ X_tr.T  # (N_te, N_tr)
    ap_cats = []
    ap_mats = []
    for i in range(sim.shape[0]):
        true_cat = (y_cat_tr == y_cat_te[i]).astype(int)
        ap_cats.append(average_precision_score(true_cat, sim[i]))
        true_mat = (y_mat_tr == y_mat_te[i]).astype(int)
        ap_mats.append(average_precision_score(true_mat, sim[i]))
    metrics["map_cat"] = np.mean(ap_cats)
    metrics["map_mat"] = np.mean(ap_mats)

    return metrics


@torch.no_grad()
@torch.inference_mode()
def evaluate_text(
    model: torch.nn.Module,
    dataloader: DataLoader,
    S_cat: torch.Tensor,
    S_mat: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, float]:
    model.eval()

    all_embeddings = []
    all_C = []
    all_M = []

    for batch in (pbar := tqdm(dataloader)):
        pbar.set_description("evaluation")

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        captions = batch["captions"].to(device)
        text_features = model.encode_text(captions).cpu()
        text_features = F.normalize(text_features, dim=-1)
        all_embeddings.append(text_features.cpu())
        all_C.append(batch["categories_matrix"].cpu())
        all_M.append(batch["materials_matrix"].cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    C = torch.cat(all_C, dim=0).numpy()
    M = torch.cat(all_M, dim=0).numpy()

    metrics = {}
    # metrics.update(evaluate_intrinsic(embeddings, C, M, S_cat, S_mat))
    metrics.update(evaluate_extrinsic(embeddings, C, M))

    return metrics

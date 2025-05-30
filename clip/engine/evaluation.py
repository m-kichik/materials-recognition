"""CLIP evaluation: cosine similarity, recall, MRR"""

import logging
import random
import time
from typing import Dict, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
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

# bad bad practice
import warnings

warnings.filterwarnings("ignore")


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
def build_similarity_targets(
    C: torch.Tensor, M: torch.Tensor, S_cat: torch.Tensor, S_mat: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """
    Build batch-wise target similarity matrices for categories and materials.

    C: (B, N_cat) one-hot category matrix
    M: (B, N_mat) multi-hot material matrix
    S_cat: (N_cat, N_cat) category similarity
    S_mat: (N_mat, N_mat) material similarity

    Returns:
      T_cat: (B, B) tensor with categorical similarities
      T_mat: (B, B) tensor with material similarities (normalized)
    """
    # Category targets: index via one-hot
    cat_idx = C.argmax(dim=1)  # (B,)
    T_cat = S_cat[cat_idx.unsqueeze(1), cat_idx.unsqueeze(0)]  # (B,B)

    # Material targets: normalized multi-hot similarity
    raw_mat = M @ S_mat @ M.t()  # (B, B)
    row_sum = M.sum(dim=1, keepdim=True)  # (B,1)
    col_sum = M.sum(dim=1, keepdim=True)  # (B,1)
    denom = (row_sum @ col_sum.t()).clamp_min(1)
    T_mat = raw_mat / denom  # (B, B)
    return T_cat, T_mat


def log_msg(msg: str):
    logging.info(msg)
    print(msg)


def evaluate_intrinsic(
    embeddings: np.ndarray,
    C: np.ndarray,
    M: np.ndarray,
    S_cat: np.ndarray,
    S_mat: np.ndarray,
    sample_pairs: int = 100000,
) -> dict:
    """
    Intrinsic evaluation using one-hot and multi-hot inputs.

    embeddings: (N, D)
    C: (N, N_cat) one-hot
    M: (N, N_mat) multi-hot
    S_cat: (N_cat, N_cat)
    S_mat: (N_mat, N_mat)

    Returns Pearson correlations, silhouette scores, and ARI for categories and materials.
    """
    metrics = {}
    N, D = embeddings.shape

    # Sample pairs for Pearson
    idx_i = np.random.randint(0, N, sample_pairs)
    idx_j = np.random.randint(0, N, sample_pairs)
    cos_sims = np.sum(embeddings[idx_i] * embeddings[idx_j], axis=1)

    # Build batch targets
    C_t = torch.from_numpy(C).float()
    M_t = torch.from_numpy(M).float()
    S_cat_t = torch.from_numpy(S_cat).float()
    S_mat_t = torch.from_numpy(S_mat).float()
    T_cat_full, T_mat_full = build_similarity_targets(C_t, M_t, S_cat_t, S_mat_t)
    T_cat = T_cat_full.numpy()
    T_mat = T_mat_full.numpy()

    cat_targets = T_cat[idx_i, idx_j]
    mat_targets = T_mat[idx_i, idx_j]

    # metrics["pearson_cat"] = pearsonr(cos_sims, cat_targets)[0]
    metrics["pearson_mat"] = pearsonr(cos_sims, mat_targets)[0]
    log_msg("evaluation: calculated pearson........")

    # Clustering metrics
    # Category clustering
    # labels_cat = C.argmax(axis=1)
    # k_cat = len(np.unique(labels_cat))
    # kmc = KMeans(n_clusters=k_cat, random_state=0).fit(embeddings)
    # metrics["silhouette_cat"] = silhouette_score(embeddings, labels_cat)
    # metrics["ari_cat"] = adjusted_rand_score(labels_cat, kmc.labels_)
    # log_msg("evaluation: calculated category clustering.......")

    # # Material clustering: sample top 50 frequent materials
    # mat_counts = M.sum(axis=0)
    # top_mat = np.argsort(-mat_counts)[:50]
    # mask = M[:, top_mat].sum(axis=1) > 0
    # emb_sub = embeddings[mask]
    # labels_mat_sub = M[mask][:, top_mat].argmax(axis=1)
    # k_mat = len(np.unique(labels_mat_sub))
    # kmm = KMeans(n_clusters=k_mat, random_state=0).fit(emb_sub)
    # metrics["silhouette_mat"] = silhouette_score(emb_sub, labels_mat_sub)
    # metrics["ari_mat"] = adjusted_rand_score(labels_mat_sub, kmm.labels_)
    # log_msg("evaluation: calculated materials clustering......")

    return metrics


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
    log_msg("evaluation: calculated k-nn for categories.....")

    knn_mat = KNeighborsClassifier(n_neighbors=k_knn).fit(X_tr, y_mat_tr)
    pred_mat = knn_mat.predict(X_te)
    acc_mat = accuracy_score(y_mat_te, pred_mat)
    p_mat, r_mat, f1_mat, _ = precision_recall_fscore_support(
        y_mat_te, pred_mat, average="weighted", zero_division=0
    )
    log_msg("evaluation: calculated k-nn for materials....")

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
        # true_cat = (y_cat_tr == y_cat_te[i]).astype(int)
        # ap_cats.append(average_precision_score(true_cat, sim[i]))
        true_mat = (y_mat_tr == y_mat_te[i]).astype(int)
        ap_mats.append(average_precision_score(true_mat, sim[i]))
    # metrics["map_cat"] = np.mean(ap_cats)
    metrics["map_mat"] = np.mean(ap_mats)
    log_msg("evaluation: calculated map...")

    return metrics


@torch.no_grad()
@torch.inference_mode()
def evaluate_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    S_cat: torch.Tensor,
    S_mat: torch.Tensor,
    mode: str = "captions",
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

        if mode == "captions":
            captions = batch["captions"]
            text_features = model.encode_text(captions, **batch).cpu()
            text_features = F.normalize(text_features, dim=-1)
            all_embeddings.append(text_features.cpu())
        elif mode == "images":
            images = batch["images"]
            image_features = model.encode_image(**batch)
            # image_features = model.encode_image(images, **batch)
            image_features = F.normalize(image_features, dim=-1)
            all_embeddings.append(image_features.cpu())
        all_C.append(batch["categories_matrix"].cpu())
        all_M.append(batch["materials_matrix"].cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    C = torch.cat(all_C, dim=0).numpy()
    M = torch.cat(all_M, dim=0).numpy()

    metrics = {}
    metrics.update(evaluate_intrinsic(embeddings, C, M, S_cat, S_mat))
    metrics.update(evaluate_extrinsic(embeddings, C, M))

    return metrics

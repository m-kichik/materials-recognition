import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt

import clip

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.materials_dataset import MaterialsDataset
from utils import set_seed

set_seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def find_class_for_image(class_embeddings, image_embedding):
    """
    Find the index of the most similar class for an image based on cosine similarity between embeddings.

    Args:
        class_embeddings: List of torch tensors, each representing a class embedding
        image_embedding: torch tensor representing the image embedding

    Returns:
        int: Index of the most similar class
        torch.Tensor: Tensor of cosine similarity scores for all classes
    """
    # Ensure image embedding is a tensor and has correct shape (1, embedding_dim)
    image_embedding = (
        image_embedding.unsqueeze(0) if image_embedding.dim() == 1 else image_embedding
    )

    # Stack class embeddings into a tensor (n_classes, embedding_dim)
    if isinstance(class_embeddings, list):
        class_embeddings = torch.stack(class_embeddings)

    # Calculate cosine similarities
    similarities = F.cosine_similarity(image_embedding, class_embeddings)

    # Find the most similar class index
    most_similar_idx = torch.argmax(similarities).item()

    return most_similar_idx, similarities


def calculate_classification_metrics(y_true, y_pred, num_classes=23):
    """
    Calculate classification metrics for multiclass classification.

    Args:
        y_true: List or array of ground truth class indices
        y_pred: List or array of predicted class indices
        num_classes: Total number of classes (default 23)

    Returns:
        dict: Dictionary containing all metrics
        matplotlib.figure: Confusion matrix plot
    """
    # Convert to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "precision_micro": precision_score(y_true, y_pred, average="micro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "recall_micro": recall_score(y_true, y_pred, average="micro"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    # Calculate per-class precision, recall, f1
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    for i in range(num_classes):
        metrics[f"precision_class_{i}"] = precision_per_class[i]
        metrics[f"recall_class_{i}"] = recall_per_class[i]
        metrics[f"f1_class_{i}"] = f1_per_class[i]

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    return metrics, plt.gcf()


def main():
    model_name = "ViT-B/32"
    val_batch_size = 1

    data_path = "/Users/rito4ka/dev/diploma/data"
    val_images_path = f"{data_path}/val2017_cropped/"
    captions_path = f"{data_path}/classification/val_classification_2017.json"
    cats_path = f"{data_path}/classification/materials_general_classification.json"

    model, preprocess = clip.load(model_name, device=device)
    model = model.to(torch.float32)

    with open(captions_path, "r") as f:
        val_captions = json.load(f)
        val_data = [{"image": k, "caption": v} for k, v in val_captions.items()]

    with torch.no_grad(), open(cats_path, "r") as f:
        all_materials = json.load(f)
        all_materials_embeddings = [
            model.encode_text(clip.tokenize([m[1]]).to(device)) for m in all_materials
        ]
        all_materials_embeddings = torch.stack(
            [emb / emb.norm(dim=-1, keepdim=True) for emb in all_materials_embeddings]
        )
        all_materials = {m[1]: m[0] for m in all_materials}

    eval_dataset = MaterialsDataset(val_images_path, val_data, preprocess=preprocess)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )

    gt_classes = []
    pred_classes = []
    for images, captions in tqdm(eval_loader):
        image_features = model.encode_image(images.to(device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        idx, _ = find_class_for_image(all_materials_embeddings, image_features)

        gt_classes.append(all_materials[captions[0]])
        pred_classes.append(idx)

    metrics, cm_plot = calculate_classification_metrics(gt_classes, pred_classes, 23)

    print("Classification Metrics:")
    for metric, value in metrics.items():
        if not metric.startswith(("precision_class", "recall_class", "f1_class")):
            print(f"{metric:20}: {value:.4f}")

    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    main()

"""CLIP contrastive loss from paper (https://arxiv.org/abs/2103.00020)"""
from typing import Callable, List

import torch
import torch.nn.functional as F
import wandb


def vanilla_clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    freeze_text: bool = False,
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

    loss_images = F.cross_entropy(logits, labels)
    if freeze_text:
        return loss_images, loss_images, None
    else:
        loss_text = F.cross_entropy(logits.T, labels)
        loss = (loss_images + loss_text) / 2
        return loss, loss_images, loss_text


class CLIPLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07, log_wandb: bool = False):
        """
        CLIP contrastive loss.

        Args:
            temperature (float): Temperature parameter for scaling logits. Default: 0.07
        """
        super().__init__()
        self.temperature = temperature
        self.logit_scale = torch.nn.Parameter(
            torch.ones([]) * torch.tensor(1.0 / temperature).log()
        )

        self.log_wandb = log_wandb

    def forward(self, image_features, text_features, **kwargs):
        """
        Compute the CLIP loss between image and text features.

        Args:
            image_features (torch.Tensor): Normalized image features [batch_size, feature_dim]
            text_features (torch.Tensor): Normalized text features [batch_size, feature_dim]

        Returns:
            tuple: (total loss, image-text contrastive loss, text-image contrastive loss)
        """
        if self.log_wandb and wandb.run is not None:
            wandb.log(
                {"train/criterion_log_scale": self.logit_scale.data.item()},
                commit=False,
            )
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        batch_size = image_features.shape[0]
        labels = torch.arange(
            batch_size, dtype=torch.long, device=image_features.device
        )

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        return loss, loss_i, loss_t
    

class ReCLIPLoss(torch.nn.Module):
    def __init__(self, class_weights: torch.tensor = None, lambda_ce: float = 0.1, temperature: float = 0.07, log_wandb: bool = False):
        """
        CLIP contrastive loss.

        Args:
            temperature (float): Temperature parameter for scaling logits. Default: 0.07
        """
        super().__init__()
        self.class_weights = class_weights
        self.lambda_ce = lambda_ce
        self.temperature = temperature
        self.logit_scale = torch.nn.Parameter(
            torch.ones([]) * torch.tensor(1.0 / temperature).log()
        )

        self.log_wandb = log_wandb

    def forward(self, image_features, text_features, cls_pred, cls_gt, **kwargs):
        """
        Compute the CLIP loss between image and text features.

        Args:
            image_features (torch.Tensor): Normalized image features [batch_size, feature_dim]
            text_features (torch.Tensor): Normalized text features [batch_size, feature_dim]

        Returns:
            tuple: (total loss, image-text contrastive loss, text-image contrastive loss)
        """
        if self.log_wandb and wandb.run is not None:
            wandb.log(
                {"train/criterion_log_scale": self.logit_scale.data.item()},
                commit=False,
            )

        ce_loss = F.cross_entropy(
            cls_pred,
            cls_gt,
            weight=self.class_weights,
            reduction="none",
        )
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        batch_size = image_features.shape[0]
        labels = torch.arange(
            batch_size, dtype=torch.long, device=image_features.device
        )

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2 + self.lambda_ce * ce_loss.mean()

        return loss, loss_i, loss_t, ce_loss.mean()


class SigLIPLoss(torch.nn.Module):
    def __init__(self, init_temperature=10.0, init_bias=-10.0, log_wandb: bool = False):
        """
        SigLIP loss for language-image pre-training.

        Args:
            init_temperature (float): initial temperature parameter (t).
                                      t is stored as log(t) as a learnable parameter.
            init_bias (float): initial bias term.
        """
        super().__init__()
        self.t_prime = torch.nn.Parameter(torch.log(torch.tensor(init_temperature)))
        self.bias = torch.nn.Parameter(torch.tensor(init_bias))

        self.log_wandb = log_wandb

    def forward(self, img_emb, txt_emb, **kwargs):
        """
        Compute the SigLIP loss from the given image and text embeddings.

        Args:
            img_emb (torch.Tensor): Image embeddings tensor of shape (N, D)
            txt_emb (torch.Tensor): Text embeddings tensor of shape (N, D)

        Returns:
            torch.Tensor: A scalar loss value.
        """
        if self.log_wandb and wandb.run is not None:
            wandb.log(
                {
                    "train/criterion_t_prime": self.t_prime.data.item(),
                    "train/criterion_bias": self.bias.data.item(),
                },
                commit=False,
            )

        zimg = F.normalize(img_emb, p=2, dim=1)
        ztxt = F.normalize(txt_emb, p=2, dim=1)

        t = torch.exp(self.t_prime)

        logits = torch.matmul(zimg, ztxt.t()) * t + self.bias

        batch_size = img_emb.size(0)
        labels = 2 * torch.eye(batch_size, device=img_emb.device) - 1

        loss = -F.logsigmoid(labels * logits).sum() / batch_size

        return (
            loss,
            torch.tensor([0]),
            torch.tensor([0]),
        )  # for conformity with vanilla loss and logging

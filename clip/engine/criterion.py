"""CLIP contrastive loss from paper (https://arxiv.org/abs/2103.00020)"""

from typing import Dict, Callable, List

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

    def get_t(self):
        return 1 / self.logit_scale.exp()

    def forward(self, image_features, text_features, **kwargs):
        """
        Compute the CLIP loss between image and text features.

        Args:
            image_features (torch.Tensor): Normalized image features [batch_size, feature_dim]
            text_features (torch.Tensor): Normalized text features [batch_size, feature_dim]

        Returns:
            tuple: (total loss, image-text contrastive loss, text-image contrastive loss)
        """
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

        if self.log_wandb and wandb.run is not None:
            wandb.log(
                {
                    "train/criterion_log_scale": self.logit_scale.data.item(),
                    "train/clip_loss": loss.item(),
                    "train/loss_images": loss_i.item(),
                    "train/loss_text": loss_t.item(),
                },
                commit=False,
            )

        return loss


class ReCLIPLoss(torch.nn.Module):
    def __init__(
        self,
        class_weights: torch.tensor = None,
        lambda_ce: float = 0.1,
        temperature: float = 0.07,
        log_wandb: bool = False,
    ):
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
        zimg = F.normalize(img_emb, p=2, dim=1)
        ztxt = F.normalize(txt_emb, p=2, dim=1)

        t = torch.exp(self.t_prime)

        logits = torch.matmul(zimg, ztxt.t()) * t + self.bias

        batch_size = img_emb.size(0)
        labels = 2 * torch.eye(batch_size, device=img_emb.device) - 1

        loss = -F.logsigmoid(labels * logits).sum() / batch_size

        if self.log_wandb and wandb.run is not None:
            wandb.log(
                {
                    "train/criterion_t_prime": self.t_prime.data.item(),
                    "train/criterion_bias": self.bias.data.item(),
                    "train/siglip_loss": loss.item(),
                },
                commit=False,
            )

        return loss


class CLIPMatSIM(torch.nn.Module):
    def __init__(
        self,
        clip_loss,
        S: torch.Tensor,
        lambda_=0.5,
        temperature=0.07,
        eps=1e-8,
        log_wandb: bool = False,
    ):
        super().__init__()
        self.clip_loss = clip_loss

        self.register_buffer("S", S)
        self.lambda_ = lambda_
        # self.tau = temperature
        self.eps = eps

        self.log_wandb = log_wandb

    def forward(self, image_features, text_features, materials_matrix, **kwargs):
        B, D = image_features.shape

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        z = torch.cat([image_features, text_features], dim=0)
        mats = torch.cat([materials_matrix, materials_matrix], dim=0)
        sim = z @ z.t()  # * self.clip_loss.get_t() # / self.tau

        mask = ~torch.eye(2 * B, device=sim.device, dtype=torch.bool)

        soft_w = mats @ self.S @ mats.T
        soft_w = soft_w * mask.float()

        logits = sim - torch.logsumexp(sim * mask, dim=1, keepdim=True)

        row_sum_w = soft_w.sum(dim=1) + self.eps
        log_pos = (soft_w * logits).sum(dim=1) / row_sum_w

        mat_loss = -log_pos.mean()

        clip_loss = self.clip_loss(image_features, text_features, **kwargs)
        loss = clip_loss + self.lambda_ * mat_loss

        if self.log_wandb and wandb.run is not None:
            wandb.log(
                {
                    "train/material_loss": mat_loss.item(),
                    "train/loss": loss.item(),
                },
                commit=False,
            )

        return loss


class TextLoss:
    def __init__(
        self,
        S: Dict[str, torch.Tensor],
        alpha: float = 1.0,  # balance categories in MSE
        beta: float = 1.0,  # balance materials in MSE
        tau: float = 0.1,  # temperature for embeddings similarity
        tau_cat: float = 0.5,  # temperature for categories similarity
        tau_mat: float = 0.5,  # temperature for materials similarity
        gamma: float = 0.1,  # balance between MSE and SupCon
        log_wandb: bool = False,
    ):
        self.S_cat = S.get("S_cat")
        self.S_mat = S.get("S_mat")

        if self.S_cat is None or self.S_mat is None:
            raise ValueError("S_cat and S_mat must be provided in the S dictionary.")

        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.th_cat = tau_cat
        self.th_mat = tau_mat
        self.gamma = gamma

        self.log_wandb = log_wandb

    def mse_multilabel_loss(self, embeddings, T_cat, T_mat, alpha=1.0, beta=1.0):
        sim = embeddings @ embeddings.t()
        loss_cat = F.mse_loss(sim, T_cat)
        loss_mat = F.mse_loss(sim, T_mat)

        if self.log_wandb and wandb.run is not None:
            wandb.log(
                {
                    "train/cat_mse": loss_cat.item(),
                    "train/mat_mse": loss_mat.item(),
                },
                commit=False,
            )

        return alpha * loss_cat + beta * loss_mat

    def supcon_multilabel(
        self, embeddings, T_cat, T_mat, tau=0.1, th_cat=0.5, th_mat=0.5
    ):
        B = embeddings.size(0)
        sim = embeddings @ embeddings.t() / tau
        mask_self = torch.eye(B, device=embeddings.device).bool()
        sim = sim.masked_fill(mask_self, -1e9)

        pos_mask = ((T_cat >= th_cat) | (T_mat >= th_mat)) & ~mask_self
        exp_sim = sim.exp()
        denom = exp_sim.sum(dim=1, keepdim=True)

        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1).clamp_min(1.0)
        loss = -(pos_sum.log() - denom.log().squeeze(1)) / pos_sum
        return loss.mean()

    def __call__(self, text_features, categories_matrix, materials_matrix, **kwargs):
        cat_idx = categories_matrix.argmax(dim=1)
        T_cat = self.S_cat[cat_idx.unsqueeze(1), cat_idx.unsqueeze(0)]
        raw_mat = materials_matrix @ self.S_mat @ materials_matrix.t()
        denom = (
            materials_matrix.sum(dim=1, keepdim=True)
            @ materials_matrix.sum(dim=1, keepdim=True).t()
        ).clamp_min(1)
        T_mat = raw_mat / denom

        mse = self.mse_multilabel_loss(
            text_features, T_cat, T_mat, alpha=self.alpha, beta=self.beta
        )
        supcon = self.supcon_multilabel(
            text_features,
            T_cat,
            T_mat,
            tau=self.tau,
            th_cat=self.th_cat,
            th_mat=self.th_mat,
        )

        loss = mse + self.gamma * supcon

        print(mse.item(), supcon.item())

        if self.log_wandb and wandb.run is not None:
            wandb.log(
                {
                    "train/mse": mse.item(),
                    "train/supcon": supcon.item(),
                    "train/loss": loss.item(),
                },
                commit=False,
            )

        return loss

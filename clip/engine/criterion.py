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


class EmbeddingsLoss:
    def __init__(
        self,
        S: Dict[str, torch.Tensor],
        alpha: float = 1.0,  # balance categories in MSE
        beta: float = 1.0,  # balance materials in MSE
        tau: float = 0.1,  # temperature for embeddings similarity
        tau_cat: float = 0.5,  # temperature for categories similarity
        tau_mat: float = 0.5,  # temperature for materials similarity
        gamma: float = 0.1,  # balance between MSE and SupCon
        momentum: float = 0.9,  # for EMA tracking of losses,
        use_mse: bool = False,
        mode: str = "",
        log_wandb: bool = False,
    ):
        self.S_cat = S.get("S_cat")
        self.S_mat = S.get("S_mat")

        if self.S_cat is None or self.S_mat is None:
            raise ValueError("S_cat and S_mat must be provided in the S dictionary.")

        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.tau_cat = tau_cat
        self.tau_mat = tau_mat
        self.gamma = gamma
        self.momentum = momentum

        self.use_mse = use_mse
        # Initialize EMA for mse and supcon
        self.mse_ema = None
        self.supcon_ema = None
        self.eps = 1e-6

        self.mode = mode
        self.log_wandb = log_wandb

    def mse_multilabel_loss(self, embeddings, T_cat, T_mat):
        sim = embeddings @ embeddings.t()
        loss_cat = F.mse_loss(sim, T_cat)
        loss_mat = F.mse_loss(sim, T_mat)

        if self.log_wandb and wandb.run is not None:
            wandb.log(
                {
                    f"train_embeddings/{self.mode}_cat_mse": loss_cat.item(),
                    f"train_embeddings/{self.mode}_mat_mse": loss_mat.item(),
                },
                commit=False,
            )

        return self.alpha * loss_cat + self.beta * loss_mat

    def supcon_multilabel(self, embeddings, T_cat, T_mat):
        B = embeddings.size(0)
        sim = embeddings @ embeddings.t() / self.tau
        mask_self = torch.eye(B, device=embeddings.device).bool()
        pos_mask = ((T_cat >= self.tau_cat) | (T_mat >= self.tau_mat)) & ~mask_self

        sim_masked = sim.masked_fill(mask_self, float("-inf"))
        denom_log = torch.logsumexp(sim_masked / self.tau, dim=1)

        sim_pos = sim_masked.masked_fill(~pos_mask, float("-inf"))
        num_log = torch.logsumexp(sim_pos / self.tau, dim=1)

        pos_count = pos_mask.sum(dim=1)
        pos_count_clamped = pos_count.clamp_min(1)

        num_log = torch.where(pos_count > 0, num_log, torch.zeros_like(num_log))

        loss = -(num_log - denom_log) / pos_count_clamped
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

        if self.use_mse:
            mse_loss = self.mse_multilabel_loss(text_features, T_cat, T_mat)

        supcon_loss = self.supcon_multilabel(
            text_features,
            T_cat,
            T_mat,
        )

        # Initialize or update EMAs
        if self.use_mse:
            if self.mse_ema is None:
                self.mse_ema = mse_loss.item()
                self.supcon_ema = supcon_loss.item()
            else:
                self.mse_ema = (
                    self.momentum * self.mse_ema + (1 - self.momentum) * mse_loss.item()
                )
                self.supcon_ema = (
                    self.momentum * self.supcon_ema
                    + (1 - self.momentum) * supcon_loss.item()
                )

            # Dynamic gamma: balance to match EMA scales
            dynamic_gamma = (self.mse_ema + self.eps) / (self.supcon_ema + self.eps)

            loss = mse_loss + dynamic_gamma * supcon_loss

            if self.log_wandb and wandb.run is not None:
                wandb.log(
                    {
                        f"train_embeddings/{self.mode}_mse": mse_loss.item(),
                        f"train_embeddings/{self.mode}_ema_mse": self.mse_ema,
                        f"train_embeddings/{self.mode}_ema_supcon": self.supcon_ema,
                        f"train_embeddings/{self.mode}_dynamic_gamma": dynamic_gamma,
                    },
                    commit=False,
                )
        else:
            loss = supcon_loss

        if self.log_wandb and wandb.run is not None:
            wandb.log(
                {
                    f"train_embeddings/{self.mode}_supcon": supcon_loss.item(),
                    f"train_embeddings/{self.mode}_embeds_loss": loss.item(),
                },
                commit=False,
            )

        return loss


class CombinedLoss(torch.nn.Module):
    def __init__(
        self,
        clip_loss,
        image_embeds_loss,
        text_embeds_loss,
        momentum: float = 0.9,
        log_wandb: bool = False,
    ):
        super().__init__()
        self.clip_loss = clip_loss
        self.image_embeds_loss = image_embeds_loss
        self.text_embeds_loss = text_embeds_loss

        # Initialize EMA for clip loss and embeddings loss
        self.use_ema = True
        self.momentum = momentum
        self.clip_ema = None
        self.embeds_ema = None
        self.eps = 1e-6

        self.log_wandb = log_wandb

    def forward(
        self,
        image_features,
        text_features,
        categories_matrix,
        materials_matrix,
        **kwargs,
    ):
        clip_loss = self.clip_loss(image_features, text_features, **kwargs)

        if self.image_embeds_loss is not None:
            image_embeds_loss = self.image_embeds_loss(
                image_features, categories_matrix, materials_matrix, **kwargs
            )
        else:
            image_embeds_loss = 0
        if self.text_embeds_loss is not None:
            text_embeds_loss = self.text_embeds_loss(
                text_features, categories_matrix, materials_matrix, **kwargs
            )
            # embeds_loss = (image_embeds_loss + text_embeds_loss) / 2
        else:
            # embeds_loss = image_embeds_loss
            text_embeds_loss = 0

        embeds_loss = (image_embeds_loss + text_embeds_loss) / 2

        if self.clip_ema is None:
            self.clip_ema = clip_loss.item()
            self.embeds_ema = embeds_loss.item()
        else:
            self.clip_ema = (
                self.momentum * self.clip_ema + (1 - self.momentum) * clip_loss.item()
            )
            self.embeds_ema = (
                self.momentum * self.embeds_ema
                + (1 - self.momentum) * embeds_loss.item()
            )

        # Dynamic gamma: balance to match EMA scales
        dynamic_gamma = (self.clip_ema + self.eps) / (self.embeds_ema + self.eps)

        if self.embeds_ema < 1e-4:
            self.use_ema = False
        else:
            self.use_ema = True

        if not self.use_ema:
            dynamic_gamma = 1

        loss = clip_loss + dynamic_gamma * embeds_loss

        if self.log_wandb and wandb.run is not None:
            wandb.log(
                {
                    "train/combined_dynamic_gamma": dynamic_gamma,
                    "train/loss": loss.item(),
                },
                commit=False,
            )

        return loss

from typing import List

import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel

import wandb


class ConcatFusion(nn.Module):
    def __init__(
            self,
            out_size: int = 512,
            device: str = "cpu",
    ):
        super().__init__()

        self.device = device

        self.context_proj = nn.LazyLinear(out_features=out_size).to(device)

        self.mlp = nn.Sequential(
            nn.LazyLinear(out_features=out_size * 2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(out_size * 2, out_size),
        )
        self.mlp.to(device)

    def forward(self, clip_features: torch.tensor, context_embeddings: torch.tensor):
        context_features = self.context_proj(context_embeddings.to(self.device))

        fused_embeddings = self.mlp(torch.cat([clip_features, context_features], dim=-1))

        return fused_embeddings


class MHAFusion(nn.Module):
    def __init__(
        self,
        out_size: int = 512,
        num_heads: int = 8,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device

        self.context_proj = nn.LazyLinear(out_features=out_size).to(device)
        # self.clip_proj = nn.Linear(out_size, out_size).to(device)

        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=out_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        ).to(device)

        # self.alpha = nn.Parameter(torch.ones([]) * torch.tensor(0.5), requires_grad=True).to(device)

        self.output_proj = nn.Linear(out_size, out_size).to(device)

    def forward(self, clip_features: torch.Tensor, context_embeddings: torch.Tensor):
        # if wandb.run is not None:
        #     wandb.log(
        #         {"train/fusion_alpha": self.alpha.data.item()},
        #         commit=False,
        #     )

        clip_features = clip_features.to(self.device)
        # clip_features = self.clip_proj(clip_features.to(self.device))
        context_features = self.context_proj(context_embeddings.to(self.device))

        attn_output, attn_weights = self.fusion_attn(
            clip_features,
            context_features,
            context_features
        )

        fused = attn_output.squeeze(1)
        # fused = self.alpha * clip_features + (1 - self.alpha) * attn_output.squeeze(1)

        fused_embeddings = self.output_proj(fused)

        return fused_embeddings


class LFCLIP(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_clip:bool = False,
        clip_ckpt:str = "",
        fusion_type: str = "concat",
        num_heads: int = 8,
        mode: str = "train",
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device

        self.clip = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.freeze_clip = freeze_clip
        if freeze_clip:
            self.load_state_dict(torch.load(clip_ckpt))
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.hidden_size = self.clip.config.vision_config.hidden_size

        self.fusion_type = fusion_type

        self.mode = mode
        if mode == "inference":
            raise NotImplementedError("Ping author to implement fair pipeline.")
        
        if fusion_type == "concat":
            self.fusion = ConcatFusion(out_size=512, device=device)
        elif fusion_type == "mha":
            self.fusion = MHAFusion(out_size=512, num_heads=num_heads, device=device)
        else:
            raise NotImplementedError(f"Fusion type {fusion_type} is not implemented.")

    def preprocess(self, images):
        images = self.processor(images=[images], return_tensors="pt", padding=True)
        return images["pixel_values"].squeeze()

    def encode_image(
        self, images: torch.tensor, embeddings: torch.tensor, **kwargs
    ):
        if self.freeze_clip:
            with torch.no_grad():
                clip_features = self.clip.get_image_features(pixel_values=images)
        else:
            clip_features = self.clip.get_image_features(pixel_values=images)

        clip_features = F.normalize(clip_features, dim=-1)

        fused_embeddings = self.fusion(clip_features, embeddings)

        return fused_embeddings

    def encode_text(self, captions: List[str], **kwargs):
        if self.freeze_clip:
            with torch.no_grad():
                text_inputs = self.processor(text=captions, return_tensors="pt", padding=True)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        else:
            text_inputs = self.processor(text=captions, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        return self.clip.get_text_features(**text_inputs)

    def forward(
        self,
        images,
        context_embeddings,
        captions,
    ):
        images_embeddings = self.encode_image(images, context_embeddings)
        captions_embeddings = self.encode_text(captions)

        return images_embeddings, captions_embeddings

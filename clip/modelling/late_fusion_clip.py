from typing import List

import PIL

import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel


class LFCLIP(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_clip:bool = False,
        clip_ckpt:str = "",
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

        self.mode = mode
        if mode == "inference":
            raise NotImplementedError("Ping author to implement fair pipeline.")

        self.context_proj = nn.LazyLinear(out_features=self.hidden_size).to(device)

        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=num_heads, batch_first=True
        ).to(device)

        self.fusion_token = nn.Parameter(
            self.clip.text_model.embeddings.token_embedding.weight[-1].clone()
        ).to(device)

        if self.clip.text_model.config.hidden_size != self.hidden_size:
            self.fusion_token_proj = nn.Linear(
                self.clip.text_model.config.hidden_size,
                self.hidden_size
            ).to(device)
        else:
            self.fusion_token_proj = nn.Identity().to(device)

        self.proj = nn.Linear(self.hidden_size, 512).to(device)

    def preprocess(self, images):
        images = self.processor(images=[images], return_tensors="pt", padding=True)
        return images["pixel_values"].squeeze()

    def encode_image(
        self, images: torch.tensor, context_embeddings: torch.tensor
    ):
        if self.freeze_clip:
            with torch.no_grad():
                vision_outputs = self.clip.vision_model(pixel_values=images)
        else:
            vision_outputs = self.clip.vision_model(pixel_values=images)
        clip_features = vision_outputs.last_hidden_state

        context_features = self.context_proj(context_embeddings.to(self.device))
        context_features = context_features.unsqueeze(1)

        batch_size = clip_features.size(0)
        fusion_tokens = self.fusion_token_proj(self.fusion_token)  # [1, 768]
        fusion_tokens = fusion_tokens.repeat(batch_size, 1, 1)
        clip_features = torch.cat([fusion_tokens, clip_features[:, 1:, :]], dim=1)

        fused_features, _ = self.fusion_attn(
            query=clip_features, key=context_features, value=context_features
        )

        fused_embeddings = fused_features[:, 0, :]
        fused_embeddings = self.proj(fused_embeddings)

        return fused_embeddings

    def encode_text(self, captions: List[str]):
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

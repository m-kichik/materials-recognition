from typing import List

import PIL

import torch
import torch.nn as nn

from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel


class CLIP(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device

        if clip_model_name.startswith("openai/clip"):
            self.clip = CLIPModel.from_pretrained(clip_model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        elif clip_model_name.startswith("google/siglip"):
            self.clip = AutoModel.from_pretrained(
                clip_model_name, 
                device_map=device,
                )
            self.processor = AutoProcessor.from_pretrained(clip_model_name)
        else:
            raise NotImplementedError(f"Support for {clip_model_name} is not implemented.")

    def preprocess(self, images):
        images = self.processor(images=[images], return_tensors="pt", padding=True)
        return images["pixel_values"].squeeze()

    def encode_image(
        self, images: torch.tensor
    ):
        return self.clip.get_image_features(pixel_values=images)

    def encode_text(self, captions: List[str]):
        text_inputs = self.processor(text=captions, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        return self.clip.get_text_features(**text_inputs)

    def forward(
        self,
        images,
        captions,
    ):
        images_embeddings = self.encode_image(images)
        captions_embeddings = self.encode_text(captions)

        return images_embeddings, captions_embeddings

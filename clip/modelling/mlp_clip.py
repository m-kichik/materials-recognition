from typing import List

import PIL

import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel


class MLPCLIP(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        num_classes: int = 162,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device

        self.clip = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self.mlp.to(device)

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
    
    def classify(self, images_embeddings):
        """
        Classify the images using the MLP.
        """
        return self.mlp(images_embeddings)

    def forward(
        self,
        images,
        captions,
    ):
        images_embeddings = self.encode_image(images)
        captions_embeddings = self.encode_text(captions)

        return images_embeddings, captions_embeddings

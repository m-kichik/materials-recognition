"""Only for building similarity matrix."""

from typing import List

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device

        if model_name == "sentence-transformers/all-mpnet-base-v2":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2", device=device
            )
        else:
            raise NotImplementedError(f"Support for {model_name} is not implemented.")

    def encode_text(self, captions: List[str]):
        embeddings = self.model.encode(captions)
        normalized_embeddings = nn.functional.normalize(
            torch.tensor(embeddings), dim=1
        )
        return normalized_embeddings

    def forward(
        self,
        captions,
    ):
        captions_embeddings = self.encode_text(captions)

        return captions_embeddings

"""
MaterialDataset class for handling image-caption pairs.
"""

import json
import pickle
from typing import List, Dict, Union, Callable, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MaterialsDataset(Dataset):
    """
    A PyTorch dataset for handling image-caption pairs.

    Attributes:
        image_dir (str): Directory where the images are stored.
        preprocess (Callable): Function to preprocess images, likely built with clip.
        data (List[Dict[str, str]]): List of dictionaries containing image filenames and captions.
    """

    def __init__(
        self,
        image_dir: str,
        captions: Union[str, List[Dict[str, str]]],
        materials_path: str = None,
        embeddings_dir: str = None,
        add_materials_prefix: bool = False,
        preprocess: Callable = None,
    ):
        """
        Initializes the MaterialsDataset.

        Args:
            image_dir (str): Path to the directory containing images.
            captions (Union[str, List[Dict[str, str]]]): Either a path to a JSON file containing image-caption pairs
                                                         or a list of dictionaries with keys 'image' and 'caption'.
            embeddings_dir (str): path to directory with embeddings for full images. If None, embeddings will not be
                                   returned as dataset item. Do not set for vanilla CLIP train.
            add_materials_prefix (bool): if True, adds "an object made of " to the caption.
            preprocess (Callable): A function to preprocess images before returning them.

        Raises:
            ValueError: If captions is not a string (JSON file path) or a list.
        """
        self.image_dir = image_dir
        self.add_materials_prefix = add_materials_prefix
        self.preprocess = preprocess

        if isinstance(captions, str):
            with open(captions, "r") as f:
                self.data = json.load(f)
        elif isinstance(captions, list):
            self.data = captions
        else:
            raise ValueError(
                "Captions should be path to json file or list with captions."
            )
        
        if materials_path is not None:
            with open(materials_path, "r") as f:
                materials = json.load(f)
            self.materials = materials["names"]
            self.materials_dict = {name: i for i, name in enumerate(self.materials)}
        else:
            self.materials = None

        if embeddings_dir is not None:
            self.load_embeddings = True
            self.embeddings_dir = embeddings_dir
        else:
            self.load_embeddings = False

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of image-caption pairs.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[any, str]:
        """
        Retrieves an image-caption pair by index.

        Args:
            idx (int): Index of the data sample to retrieve.

        Returns:
            Tuple[any, str]: A tuple (image, caption) where image is the processed image tensor,
                             and caption is the corresponding text description.
        """
        ret_vals = []

        image_path = f"{self.image_dir}/{self.data[idx]['image']}"
        image = Image.open(image_path)
        if self.preprocess is not None:
            image = self.preprocess(image)
        ret_vals.append(image)

        caption = self.data[idx]["caption"].lower().strip()
        if self.add_materials_prefix:
            caption = "an object made of " + caption
        ret_vals.append(caption)

        if self.materials is not None:
            mat_idx = self.materials_dict.get(caption)
            ret_vals.append(mat_idx)

        if self.load_embeddings:
            embedding_path = f"{self.embeddings_dir}/{self.data[idx]['image'][:-4].split('_')[0]}.pkl"
            with open(embedding_path, "rb") as file:
                embedding = torch.tensor(pickle.load(file))

            if embedding.shape[0] == 1:  # we are responsible for the mistakes we made
                embedding = torch.squeeze(embedding)

            ret_vals.append(embedding)

        return tuple(ret_vals)

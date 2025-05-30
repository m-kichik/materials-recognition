"""
MaterialDataset class for handling image-caption pairs.
"""

import json
import pickle
from typing import List, Dict, Union, Callable, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MaterialsDataset(Dataset):
    """
    A PyTorch dataset for handling image-caption pairs with optional image augmentations.

    Attributes:
        image_dir (str): Directory where the images are stored.
        preprocess (Callable): Function to preprocess images, likely built with CLIP.
        augmentations (Callable): Torchvision transforms for data augmentation.
        data (List[Dict[str, str]]): List of dictionaries containing image filenames and captions.
    """

    def __init__(
        self,
        images_dir: str,
        captions: Union[str, List[Dict[str, str]]],
        captions_key: str = "caption",
        materials: Union[bool, str, dict] = None,
        categories: Union[bool, str, dict] = None,
        embeddings_dir: str = None,
        add_materials_prefix: bool = False,
        preprocess: Callable = None,
        augmentations: bool | Callable = None,
    ):
        """
        Initializes the MaterialsDataset with optional augmentations.

        Args:
            images_dir (str): Path to the directory containing images.
            captions (Union[str, List[Dict[str, str]]]): Path to JSON file or list of dicts with 'image' and 'caption'.
            captions_key (str): Key in data for the caption text.
            materials (bool|str|dict): Material mapping or flag to derive materials from data.
            categories (bool|str|dict): Category mapping or flag to derive categories from data.
            embeddings_dir (str): Path to directory with precomputed embeddings (.pkl).
            add_materials_prefix (bool): If True, adds 'an object made of ' to caption.
            preprocess (Callable): Preprocessing function (e.g., CLIP preprocess).
            augmentations (Callable): Torchvision transforms for training-time augmentations.
        """
        self.image_dir = images_dir
        self.add_materials_prefix = add_materials_prefix
        self.preprocess = preprocess

        # Load captions
        if isinstance(captions, str):
            with open(captions, "r") as f:
                self.data = json.load(f)
        elif isinstance(captions, list):
            self.data = captions
        else:
            raise ValueError(
                "Captions should be path to json file or list with captions."
            )
        self.captions_key = captions_key

        # Categories setup
        self.num_categories = None
        if categories is not None:
            if isinstance(categories, dict):
                self.cat2idx = categories
            elif isinstance(categories, str):
                with open(categories, "r") as f:
                    categories = json.load(f)
                self.cat2idx = {name: i for i, name in enumerate(categories)}
            elif isinstance(categories, bool) and categories:
                all_categories = set()
                for item in self.data:
                    cats = item.get("category", []) or []
                    all_categories.update(cats)
                self.cat2idx = {
                    name: i for i, name in enumerate(sorted(all_categories))
                }
            self.num_categories = len(self.cat2idx)

        # Materials setup
        self.num_materials = None
        if materials is not None:
            if isinstance(materials, dict):
                self.mat2idx = materials
            elif isinstance(materials, str):
                with open(materials, "r") as f:
                    materials = json.load(f)
                self.mat2idx = {name: i for i, name in enumerate(materials)}
            elif isinstance(materials, bool) and materials:
                all_materials = set()
                for item in self.data:
                    mats = item.get("material", []) or []
                    all_materials.update(mats)
                self.mat2idx = {name: i for i, name in enumerate(sorted(all_materials))}
            self.num_materials = len(self.mat2idx)

        # Embeddings setup
        if embeddings_dir:
            self.load_embeddings = True
            self.embeddings_dir = embeddings_dir
        else:
            self.load_embeddings = False

        # Augmentations setup
        if isinstance(augmentations, bool) and augmentations:
            self.augmentations = T.Compose(
                [
                    # T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(p=0.25),
                    T.RandomVerticalFlip(p=0.25),
                    T.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    T.RandomGrayscale(p=0.1),
                    T.RandomRotation(degrees=15),
                ]
            )
        else:
            self.augmentations = augmentations

        print(self.augmentations)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        item = self.data[idx]
        image_path = f"{self.image_dir}/{item['image']}"
        image = Image.open(image_path).convert("RGB")

        # Apply augmentations before preprocessing
        if self.augmentations:
            image = self.augmentations(image)

        # CLIP or other preprocess
        if self.preprocess:
            image = self.preprocess(image)

        # Prepare return dict
        ret_vals = {"images": image}

        # Caption
        caption = item.get(self.captions_key, "").lower().strip()[:120]
        if self.add_materials_prefix:
            caption = "an object made of " + caption
        ret_vals["captions"] = caption

        # One-hot categories
        if self.num_categories:
            cat_vec = torch.zeros(self.num_categories, dtype=torch.float)
            for c in item.get("category", []) or []:
                cat_vec[self.cat2idx.get(c, -1)] = 1.0
            ret_vals["categories_matrix"] = cat_vec

        # One-hot materials
        if self.num_materials:
            mat_vec = torch.zeros(self.num_materials, dtype=torch.float)
            for m in item.get("material", []) or []:
                mat_vec[self.mat2idx.get(m, -1)] = 1.0
            ret_vals["materials_matrix"] = mat_vec

        # Embeddings
        if self.load_embeddings:
            emb_name = item["image"].rsplit(".", 1)[0].split("_")[0]
            emb_path = f"{self.embeddings_dir}/{emb_name}.pkl"
            with open(emb_path, "rb") as f:
                emb = torch.tensor(pickle.load(f))
            if emb.dim() == 1:
                ret_vals["embeddings"] = emb
            else:
                ret_vals["embeddings"] = emb.squeeze(0)

        return ret_vals

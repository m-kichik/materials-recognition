"""
MaterialDataset class for handling image-caption pairs.
"""

import json
from typing import List, Dict, Union, Callable, Tuple

from PIL import Image
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
        preprocess: Callable = None,
    ):
        """
        Initializes the MaterialsDataset.

        Args:
            image_dir (str): Path to the directory containing images.
            captions (Union[str, List[Dict[str, str]]]): Either a path to a JSON file containing image-caption pairs
                                                         or a list of dictionaries with keys 'image' and 'caption'.
            preprocess (Callable): A function to preprocess images before returning them.

        Raises:
            ValueError: If captions is not a string (JSON file path) or a list.
        """
        self.image_dir = image_dir
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
        image_path = f"{self.image_dir}/{self.data[idx]['image']}"
        image = Image.open(image_path)
        if self.preprocess is not None:
            image = self.preprocess(image)

        caption = self.data[idx]["caption"]

        return image, caption

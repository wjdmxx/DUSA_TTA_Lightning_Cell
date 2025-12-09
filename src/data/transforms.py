"""Data transforms for task and auxiliary models."""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from typing import Tuple, Optional


class TaskTransform:
    """
    Transforms for discriminative task model.
    Standard ImageNet preprocessing: center crop 224, normalize.
    """

    def __init__(
        self,
        input_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.input_size = input_size
        self.mean = mean
        self.std = std

        self.transform = T.Compose(
            [
                # T.Resize(256),
                T.CenterCrop(input_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image):
        """
        Args:
            image: PIL Image or tensor

        Returns:
            Preprocessed tensor (3, 224, 224)
        """
        return self.transform(image)


class RawImageCollector:
    """
    Keeps raw images (before task preprocessing) for auxiliary model.
    Auxiliary model needs BGR [0, 255] format.
    """

    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, image):
        """
        Args:
            image: PIL Image (RGB)

        Returns:
            Tensor (3, H, W) in RGB [0, 255]
        """
        # Convert to tensor [0, 1] then scale to [0, 255]
        tensor = self.to_tensor(image) * 255.0
        # Convert RGB to BGR for compatibility with original code
        tensor = tensor[[2, 1, 0], :, :]  # RGB -> BGR
        return tensor


def create_tta_transforms(
    task_input_size: int = 224,
    task_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    task_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    """
    Create transforms for TTA.

    Returns:
        task_transform: Transform for task model
        raw_transform: Transform to keep raw images for auxiliary model
    """
    task_transform = TaskTransform(
        input_size=task_input_size,
        mean=task_mean,
        std=task_std,
    )

    raw_transform = RawImageCollector()

    return task_transform, raw_transform


def dual_transform(image, task_transform, raw_transform):
    """
    Apply both task and raw transforms to an image.

    Args:
        image: PIL Image
        task_transform: Transform for task model
        raw_transform: Transform for raw image

    Returns:
        task_image: Preprocessed for task model (3, 224, 224)
        raw_image: Raw image for auxiliary model (3, H, W) in BGR [0, 255]
    """
    task_image = task_transform(image)
    raw_image = raw_transform(image)
    return task_image, raw_image

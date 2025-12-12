"""Data transforms for TTA.

New pipeline:
1. Image -> CenterCrop(224) -> ToTensor() -> [0, 1] tensor
2. In CombinedModel: tensor -> PixelAdapter -> clamp(0, 1)
3A. Discriminative branch: normalize with ImageNet mean/std
3B. Generative branch: resize to 256, scale to [-1, 1], detach()
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from typing import Tuple, Optional


class TTATransform:
    """
    Transform for TTA: CenterCrop + ToTensor only.
    Outputs tensor in [0, 1] range.
    Normalization is done in model forward pass.
    """

    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        self.transform = T.Compose(
            [
                T.CenterCrop(input_size),
                T.ToTensor(),  # Converts to [0, 1]
            ]
        )

    def __call__(self, image):
        """
        Args:
            image: PIL Image

        Returns:
            Tensor (3, 224, 224) in [0, 1] range
        """
        return self.transform(image)


def create_tta_transform(input_size: int = 224) -> TTATransform:
    """
    Create transform for TTA.
    
    Args:
        input_size: Size to center crop to (default 224)

    Returns:
        transform: TTATransform that outputs [0, 1] tensor
    """
    return TTATransform(input_size=input_size)

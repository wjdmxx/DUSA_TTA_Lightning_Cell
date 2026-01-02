"""Data transforms for TTA.

New pipeline:
1. Image -> Resize(input_size) -> ToTensor() -> [0, 1] tensor
2. In CombinedModel: tensor -> PixelAdapter -> clamp(0, 1)
3A. Discriminative branch: normalize with ImageNet mean/std
3B. Generative branch: resize to image_size (e.g., 512), scale to [-1, 1], detach()
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from typing import Tuple, Optional


class TTATransform:
    """
    Transform for TTA: Resize + ToTensor.
    Outputs tensor in [0, 1] range.
    Normalization is done in model forward pass.
    """

    def __init__(self, input_size: int = 224, use_center_crop: bool = False):
        self.input_size = input_size
        self.use_center_crop = use_center_crop
        
        if use_center_crop:
            self.transform = T.Compose(
                [
                    T.CenterCrop(input_size),
                    T.ToTensor(),  # Converts to [0, 1]
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize((input_size, input_size)),
                    T.ToTensor(),  # Converts to [0, 1]
                ]
            )

    def __call__(self, image):
        """
        Args:
            image: PIL Image

        Returns:
            Tensor (3, input_size, input_size) in [0, 1] range
        """
        return self.transform(image)


def create_tta_transform(input_size: int = 224, use_center_crop: bool = False) -> TTATransform:
    """
    Create transform for TTA.
    
    Args:
        input_size: Size to resize/crop to (default 224)
        use_center_crop: If True, use CenterCrop; if False, use Resize

    Returns:
        transform: TTATransform that outputs [0, 1] tensor
    """
    return TTATransform(input_size=input_size, use_center_crop=use_center_crop)

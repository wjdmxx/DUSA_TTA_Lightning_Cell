"""ImageNet-C dataset for TTA."""
import os
from pathlib import Path
from typing import Optional, List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageNetCDataset(Dataset):
    """
    ImageNet-C dataset for test-time adaptation.
    Each corruption type + severity is treated as a separate task.
    """
    
    # ImageNet-C corruption types
    CORRUPTIONS = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression",
        "speckle_noise", "gaussian_blur", "spatter", "saturate",
    ]
    
    SEVERITIES = [1, 2, 3, 4, 5]
    
    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int,
        transform=None,
        raw_transform=None,
    ):
        """
        Args:
            root: Path to ImageNet-C directory (contains corruption folders)
            corruption: Corruption type (e.g., "gaussian_noise")
            severity: Severity level (1-5)
            transform: Transform for task model
            raw_transform: Transform for raw images (auxiliary model)
        """
        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        self.raw_transform = raw_transform
        
        # Build image list
        corruption_dir = self.root / corruption / str(severity)
        if not corruption_dir.exists():
            raise ValueError(f"Corruption directory not found: {corruption_dir}")
        
        self.image_paths = []
        self.labels = []
        
        # ImageNet-C structure: corruption/severity/class/images
        class_folders = sorted([d for d in corruption_dir.iterdir() if d.is_dir()])
        
        for class_idx, class_folder in enumerate(class_folders):
            image_files = sorted(list(class_folder.glob("*.JPEG")) + list(class_folder.glob("*.jpg")))
            for img_path in image_files:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {corruption_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        task_image = self.transform(image) if self.transform else image
        raw_image = self.raw_transform(image) if self.raw_transform else None
        
        return {
            "task_image": task_image,
            "raw_image": raw_image,
            "label": label,
            "image_path": str(img_path),
        }


def create_imagenet_c_tasks(
    root: str,
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    task_transform=None,
    raw_transform=None,
) -> List[Tuple[str, Dataset]]:
    """
    Create list of ImageNet-C task datasets.
    
    Args:
        root: Path to ImageNet-C directory
        corruptions: List of corruptions to use (None = all)
        severities: List of severities to use (None = all)
        task_transform: Transform for task model
        raw_transform: Transform for raw images
    
    Returns:
        List of (task_name, dataset) tuples
    """
    if corruptions is None:
        corruptions = ImageNetCDataset.CORRUPTIONS
    if severities is None:
        severities = ImageNetCDataset.SEVERITIES
    
    tasks = []
    for corruption in corruptions:
        for severity in severities:
            task_name = f"{corruption}_severity{severity}"
            try:
                dataset = ImageNetCDataset(
                    root=root,
                    corruption=corruption,
                    severity=severity,
                    transform=task_transform,
                    raw_transform=raw_transform,
                )
                tasks.append((task_name, dataset))
            except ValueError as e:
                print(f"Warning: Skipping {task_name}: {e}")
    
    return tasks


def custom_collate_fn(batch):
    """
    Custom collate function for TTA data.
    Handles both task images (batched) and raw images (list).
    """
    task_images = torch.stack([item["task_image"] for item in batch])
    raw_images = [item["raw_image"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    image_paths = [item["image_path"] for item in batch]
    
    return {
        "task_images": task_images,
        "raw_images": raw_images,
        "labels": labels,
        "image_paths": image_paths,
    }

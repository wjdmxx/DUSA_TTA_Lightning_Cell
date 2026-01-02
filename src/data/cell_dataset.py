"""Cell dataset for TTA.

A simple ImageFolder-style dataset for cell classification.
Structure: root/class_name/images
"""
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image


class CellDataset(Dataset):
    """
    Cell dataset for test-time adaptation.
    Structure: root/class_name/images (ImageFolder format)
    """

    def __init__(
        self,
        root: str,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            root: Path to dataset directory (contains class folders)
            transform: Transform that outputs [0, 1] tensor
            class_to_idx: Optional dict mapping class names to indices.
                         If None, will be created from sorted folder names.
        """
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise ValueError(f"Dataset directory not found: {self.root}")

        # Build class mapping
        class_folders = sorted([d for d in self.root.iterdir() if d.is_dir()])
        
        if len(class_folders) == 0:
            raise ValueError(f"No class folders found in {self.root}")
        
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {folder.name: idx for idx, folder in enumerate(class_folders)}
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        self.num_classes = len(self.classes)

        # Build image list
        self.image_paths = []
        self.labels = []

        for class_folder in class_folders:
            class_name = class_folder.name
            if class_name not in self.class_to_idx:
                print(f"Warning: Class folder '{class_name}' not in class_to_idx, skipping")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Support common image extensions
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.JPEG", "*.JPG", "*.PNG"]:
                image_files.extend(class_folder.glob(ext))
            image_files = sorted(image_files)
            
            for img_path in image_files:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root}")
        
        print(f"CellDataset: {len(self.image_paths)} images, {self.num_classes} classes")
        for cls_name, cls_idx in self.class_to_idx.items():
            count = sum(1 for l in self.labels if l == cls_idx)
            print(f"  {cls_name} (idx={cls_idx}): {count} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transform: outputs [0, 1] tensor
        image_tensor = self.transform(image) if self.transform else image

        return {
            "image": image_tensor,
            "label": label,
            "index": idx,
            "image_path": str(img_path),
        }


def create_cell_dataset(
    root: str,
    transform=None,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> CellDataset:
    """
    Create Cell dataset.

    Args:
        root: Path to dataset directory
        transform: Transform that outputs [0, 1] tensor
        class_to_idx: Optional class mapping

    Returns:
        CellDataset instance
    """
    return CellDataset(
        root=root,
        transform=transform,
        class_to_idx=class_to_idx,
    )


def create_cell_tasks(
    root: str,
    transform=None,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, Dataset]]:
    """
    Create list of cell task datasets.
    For cell data, we treat the entire dataset as a single task.

    Args:
        root: Path to cell dataset directory
        transform: Transform that outputs [0, 1] tensor
        class_to_idx: Optional class mapping

    Returns:
        List of (task_name, dataset) tuples
    """
    tasks = []
    task_name = "cell_classification"
    
    try:
        dataset = CellDataset(
            root=root,
            transform=transform,
            class_to_idx=class_to_idx,
        )
        tasks.append((task_name, dataset))
    except ValueError as e:
        print(f"Warning: Skipping {task_name}: {e}")

    return tasks


def cell_collate_fn(batch):
    """
    Custom collate function for cell TTA data.
    Returns batched images in [0, 1] range.
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    indices = torch.tensor([item.get("index", i) for i, item in enumerate(batch)])
    image_paths = [item["image_path"] for item in batch]

    return {
        "images": images,  # (B, 3, H, W) in [0, 1]
        "labels": labels,
        "indices": indices,
        "image_paths": image_paths,
    }

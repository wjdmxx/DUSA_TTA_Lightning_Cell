"""Utility functions for metrics."""
import torch
import torch.nn.functional as F
from typing import Tuple


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> dict:
    """
    Calculate top-k accuracy.
    
    Args:
        logits: Model predictions (B, num_classes)
        labels: Ground truth labels (B,)
        topk: Tuple of k values for top-k accuracy
    
    Returns:
        Dictionary with top-k accuracy values
    """
    maxk = max(topk)
    batch_size = labels.size(0)
    
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    
    results = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results[f"top{k}"] = correct_k.mul_(100.0 / batch_size).item()
    
    return results


def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate prediction entropy.
    
    Args:
        logits: Model predictions (B, num_classes)
    
    Returns:
        Entropy values (B,)
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def calculate_confidence(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate prediction confidence.
    
    Args:
        logits: Model predictions (B, num_classes)
    
    Returns:
        max_probs: Maximum probability per sample (B,)
        predictions: Predicted class indices (B,)
    """
    probs = F.softmax(logits, dim=-1)
    max_probs, predictions = torch.max(probs, dim=-1)
    return max_probs, predictions


def calculate_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict:
    """
    Calculate F1, Sensitivity (Recall), and Precision (macro-averaged).
    
    Args:
        predictions: Predicted class indices (B,)
        labels: Ground truth labels (B,)
        num_classes: Total number of classes
    
    Returns:
        Dictionary with F1, Sensitivity, and Precision values
    """
    # Initialize per-class counters
    tp = torch.zeros(num_classes, device=predictions.device)
    fp = torch.zeros(num_classes, device=predictions.device)
    fn = torch.zeros(num_classes, device=predictions.device)
    
    for c in range(num_classes):
        pred_c = predictions == c
        label_c = labels == c
        tp[c] = (pred_c & label_c).sum().float()
        fp[c] = (pred_c & ~label_c).sum().float()
        fn[c] = (~pred_c & label_c).sum().float()
    
    # Compute per-class precision, recall, F1
    precision_per_class = tp / (tp + fp + 1e-8)
    recall_per_class = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-8)
    
    # Macro average (only for classes that appear in labels)
    classes_in_labels = torch.unique(labels)
    if len(classes_in_labels) > 0:
        precision = precision_per_class[classes_in_labels].mean().item()
        recall = recall_per_class[classes_in_labels].mean().item()
        f1 = f1_per_class[classes_in_labels].mean().item()
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    
    return {
        "precision": precision * 100.0,
        "sensitivity": recall * 100.0,  # Sensitivity = Recall
        "f1": f1 * 100.0,
    }

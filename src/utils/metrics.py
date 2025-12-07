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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Script for ViT-B/16 on Raabin-WBC Dataset
- Evaluate all checkpoints in a directory
- Auto-detect test classes (supports full 5-class or subset 2-class)
- Computes macro metrics with proper class mapping
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import timm
from timm.data import create_transform, resolve_data_config


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def compute_metrics_from_confmat(
    conf: np.ndarray, class_ids: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute accuracy, macro F1, sensitivity, precision from confusion matrix.
    conf: [C, C] where rows = true, cols = pred
    class_ids: if provided, compute macro metrics only on these classes.
               This handles the case where only some rows have non-zero values
               (e.g., subset evaluation with full logits).
    """
    C = conf.shape[0]
    if class_ids is None:
        class_ids = list(range(C))

    # Total samples: sum of rows for relevant classes only
    total = sum(conf[c, :].sum() for c in class_ids)
    # Correct predictions: diagonal entries for relevant classes
    correct = sum(conf[c, c] for c in class_ids)
    acc = float(correct / total) if total > 0 else 0.0

    precisions, recalls, f1s = [], [], []
    for c in class_ids:
        tp = conf[c, c]
        # FP: other classes (in class_ids) predicted as c
        fp = sum(conf[other, c] for other in class_ids if other != c)
        # FN: class c predicted as anything else (including classes outside class_ids)
        fn = conf[c, :].sum() - tp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_sens = float(np.mean(recalls)) if recalls else 0.0
    
    return {
        "accuracy": acc,
        "balanced_accuracy": macro_sens,  # balanced acc = mean of per-class recall
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "macro_sensitivity": macro_sens,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
    }


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate when test set has the same classes as training.
    Returns metrics and confusion matrix.
    """
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
            conf[int(t), int(p)] += 1

    metrics = compute_metrics_from_confmat(conf)
    return metrics, conf


@torch.no_grad()
def evaluate_subset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_class_to_idx: Dict[str, int],
    test_class_names: List[str],
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate when test set is a SUBSET of training classes.
    
    - Model outputs logits over all training classes
    - We restrict to the subset classes present in test set
    - Predictions are made only among these subset classes
    
    Returns metrics and confusion matrix (in subset space).
    """
    num_subset = len(test_class_names)
    
    # Map test class names to their indices in the full model output
    subset_indices = []
    for cn in test_class_names:
        if cn not in train_class_to_idx:
            raise ValueError(
                f"Test class '{cn}' not found in training classes: "
                f"{list(train_class_to_idx.keys())}"
            )
        subset_indices.append(train_class_to_idx[cn])
    
    print(f"  Subset mapping: {list(zip(test_class_names, subset_indices))}")
    
    model.eval()
    conf = np.zeros((num_subset, num_subset), dtype=np.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)  # [B, num_train_classes]
        
        # Restrict logits to subset classes
        logits_subset = logits[:, subset_indices]  # [B, num_subset]
        preds = torch.argmax(logits_subset, dim=1)  # predictions in subset space (0 to num_subset-1)

        # targets are already in subset space (0 to num_subset-1) from ImageFolder
        for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
            conf[int(t), int(p)] += 1

    metrics = compute_metrics_from_confmat(conf)
    return metrics, conf


@torch.no_grad()
def evaluate_subset_full_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_class_to_idx: Dict[str, int],
    test_class_names: List[str],
    num_train_classes: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate subset with FULL logits - predictions are made over ALL training classes.
    
    This is stricter than evaluate_subset: if model predicts a class not in test set, 
    it counts as an error. This tests whether the model has truly learned to distinguish
    the subset classes from all other classes.
    
    Returns metrics and confusion matrix (in full training class space).
    """
    # Map test class names to their indices in the full model output
    test_indices_in_train = []
    for cn in test_class_names:
        if cn not in train_class_to_idx:
            raise ValueError(
                f"Test class '{cn}' not found in training classes: "
                f"{list(train_class_to_idx.keys())}"
            )
        test_indices_in_train.append(train_class_to_idx[cn])
    
    print(f"  Subset mapping (full logits): {list(zip(test_class_names, test_indices_in_train))}")
    
    model.eval()
    # Full confusion matrix in training class space
    conf = np.zeros((num_train_classes, num_train_classes), dtype=np.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)  # [B, num_train_classes]
        preds = torch.argmax(logits, dim=1)  # predictions in full space (0 to num_train_classes-1)

        # targets are in subset space (0 to num_subset-1), need to map to full space
        for t_subset, p_full in zip(targets.cpu().numpy(), preds.cpu().numpy()):
            t_full = test_indices_in_train[int(t_subset)]
            conf[int(t_full), int(p_full)] += 1

    # Compute metrics only on test classes (but predictions can be any class)
    metrics = compute_metrics_from_confmat(conf, class_ids=test_indices_in_train)
    return metrics, conf


def load_checkpoint(ckpt_path: Path, model: nn.Module, device: torch.device):
    """Load model weights from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    
    # Extract metadata
    meta = {}
    if "step" in ckpt:
        meta["step"] = ckpt["step"]
    if "epoch" in ckpt:
        meta["epoch"] = ckpt["epoch"]
    if "best_val_f1" in ckpt:
        meta["best_val_f1"] = ckpt["best_val_f1"]
    
    return meta


def find_checkpoints(ckpt_dir: Path) -> List[Path]:
    """Find all .pt checkpoint files in directory."""
    ckpts = list(ckpt_dir.glob("*.pt"))
    ckpts.sort(key=lambda x: x.name)
    return ckpts


def load_config(ckpt_dir: Path) -> Dict:
    """Load training config from run directory."""
    # Try to find config.json in parent directory or current directory
    for config_path in [ckpt_dir.parent / "config.json", ckpt_dir / "config.json"]:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser("Evaluate ViT checkpoints on test data")

    parser.add_argument(
        "--ckpt_dir", type=str, required=True,
        help="Directory containing checkpoint files (.pt)"
    )
    parser.add_argument(
        "--test_dir", type=str, required=True,
        help="Path to test data directory (ImageFolder format)"
    )
    parser.add_argument(
        "--model", type=str, default="vit_base_patch16_224",
        help="Model architecture (must match training)"
    )
    parser.add_argument(
        "--num_classes", type=int, default=None,
        help="Number of classes model was trained on (auto-detect from config if not set)"
    )
    parser.add_argument(
        "--train_class_names", type=str, nargs="+", default=None,
        help="Training class names in order (auto-detect from config if not set)"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results (default: results.json in ckpt_dir)"
    )
    parser.add_argument(
        "--full_logits", action="store_true", default=False,
        help="When test set is a subset, predict over ALL training classes instead of only subset classes. "
             "This is stricter: predictions to classes outside the test set count as errors."
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)

    # -------------------------
    # Load training config
    # -------------------------
    config = load_config(ckpt_dir)
    
    if config:
        print(f"[Config] Loaded from {ckpt_dir.parent / 'config.json'}")
        if args.num_classes is None:
            args.num_classes = config["num_classes"]
        if args.train_class_names is None:
            args.train_class_names = config["class_names"]
        if "model" in config:
            args.model = config["model"]
    else:
        print("[Config] No config.json found, using command line args")
        if args.num_classes is None or args.train_class_names is None:
            raise ValueError(
                "Must provide --num_classes and --train_class_names "
                "when config.json is not available"
            )

    train_class_to_idx = {name: i for i, name in enumerate(args.train_class_names)}
    print(f"[Train classes] {args.train_class_names} (n={args.num_classes})")

    # -------------------------
    # Test dataset
    # -------------------------
    test_ds = ImageFolder(args.test_dir)
    test_class_names = test_ds.classes
    num_test_classes = len(test_class_names)
    print(f"[Test classes]  {test_class_names} (n={num_test_classes})")

    # Determine evaluation mode
    is_subset = num_test_classes < args.num_classes
    is_full = num_test_classes == args.num_classes
    
    if is_subset:
        # Verify all test classes exist in training classes
        for cn in test_class_names:
            if cn not in train_class_to_idx:
                raise ValueError(
                    f"Test class '{cn}' not in training classes {args.train_class_names}"
                )
        if args.full_logits:
            print(f"[Mode] SUBSET evaluation with FULL LOGITS ({num_test_classes}/{args.num_classes} classes)")
            print("       Predictions over all training classes; predicting absent classes = error")
        else:
            print(f"[Mode] SUBSET evaluation ({num_test_classes}/{args.num_classes} classes)")
            print("       Predictions restricted to subset classes only")
    elif is_full:
        # Verify class names match
        if set(test_class_names) != set(args.train_class_names):
            print(f"[Warn] Test classes differ from train classes!")
            print(f"  Train: {args.train_class_names}")
            print(f"  Test:  {test_class_names}")
        print(f"[Mode] FULL evaluation (all {args.num_classes} classes)")
    else:
        raise ValueError(
            f"Test has {num_test_classes} classes but model was trained on {args.num_classes}"
        )

    # -------------------------
    # Create model and get transforms
    # -------------------------
    model = timm.create_model(
        args.model, pretrained=False, num_classes=args.num_classes
    )
    model.to(device)

    # Get data config
    if config and "data_config" in config:
        data_cfg = config["data_config"]
        mean = tuple(data_cfg["mean"])
        std = tuple(data_cfg["std"])
        interpolation = data_cfg.get("interpolation", "bicubic")
    else:
        data_cfg = resolve_data_config({}, model=model)
        mean = data_cfg["mean"]
        std = data_cfg["std"]
        interpolation = data_cfg.get("interpolation", "bicubic")

    print(f"[Transform] mean={mean}, std={std}")

    eval_tf = create_transform(
        input_size=(3, args.input_size, args.input_size),
        is_training=False,
        interpolation=interpolation,
        mean=mean,
        std=std,
    )
    test_ds.transform = eval_tf

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -------------------------
    # Find and evaluate checkpoints
    # -------------------------
    ckpts = find_checkpoints(ckpt_dir)
    if not ckpts:
        raise FileNotFoundError(f"No .pt files found in {ckpt_dir}")
    
    print(f"\n[Checkpoints] Found {len(ckpts)} files")
    print("-" * 80)

    results = []

    for ckpt_path in ckpts:
        print(f"\n  Evaluating: {ckpt_path.name}")
        
        try:
            meta = load_checkpoint(ckpt_path, model, device)
        except Exception as e:
            print(f"    [Error] Failed to load: {e}")
            continue

        if is_subset:
            if args.full_logits:
                metrics, conf = evaluate_subset_full_logits(
                    model, test_loader, device, train_class_to_idx, 
                    test_class_names, args.num_classes
                )
            else:
                metrics, conf = evaluate_subset(
                    model, test_loader, device, train_class_to_idx, test_class_names
                )
        else:
            metrics, conf = evaluate_full(
                model, test_loader, device, args.num_classes
            )

        result = {
            "checkpoint": ckpt_path.name,
            **meta,
            "test_classes": test_class_names,
            "is_subset": is_subset,
            "full_logits": args.full_logits if is_subset else None,
            **metrics,
            "confusion_matrix": conf.tolist(),
        }
        results.append(result)

        print(
            f"    Acc={metrics['accuracy']:.4f}  "
            f"BalAcc={metrics['balanced_accuracy']:.4f}  "
            f"F1={metrics['macro_f1']:.4f}  "
            f"Sens={metrics['macro_sensitivity']:.4f}  "
            f"Prec={metrics['macro_precision']:.4f}"
        )

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results:
        # Find best by macro_f1
        best = max(results, key=lambda x: x["macro_f1"])
        print(f"\nBest checkpoint by macro_f1: {best['checkpoint']}")
        print(f"  Accuracy:         {best['accuracy']:.4f}")
        print(f"  Balanced Acc:     {best['balanced_accuracy']:.4f}")
        print(f"  Macro F1:         {best['macro_f1']:.4f}")
        print(f"  Sensitivity:      {best['macro_sensitivity']:.4f}")
        print(f"  Precision:        {best['macro_precision']:.4f}")

        # Print table
        print("\n{:<30} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
            "Checkpoint", "Acc", "BalAcc", "F1", "Sens", "Prec"
        ))
        print("-" * 80)
        for r in results:
            print("{:<30} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
                r["checkpoint"][:30],
                r["accuracy"],
                r["balanced_accuracy"],
                r["macro_f1"],
                r["macro_sensitivity"],
                r["macro_precision"],
            ))

    # Save results
    output_path = Path(args.output) if args.output else ckpt_dir / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_dir": str(args.test_dir),
            "test_classes": test_class_names,
            "train_classes": args.train_class_names,
            "is_subset": is_subset,
            "full_logits": args.full_logits if is_subset else None,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Saved] {output_path}")


if __name__ == "__main__":
    main()
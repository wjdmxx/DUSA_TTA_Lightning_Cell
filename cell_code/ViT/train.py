#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViT-B/16 Training on Raabin-WBC Dataset
- Finetune from ImageNet pretrained weights
- Save checkpoints at regular intervals
"""

import os
import json
import time
import uuid
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder

import timm
from timm.data import create_transform, resolve_data_config
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(output_root: str, prefix: str = "vitb16_raabin") -> Path:
    run_id = f"{prefix}_{now_str()}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class TransformSubset(torch.utils.data.Dataset):
    """Wrapper to apply transform to a Subset."""
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        x = self.transform(x)
        return x, y


@torch.no_grad()
def compute_metrics_from_confmat(conf: np.ndarray) -> Dict[str, float]:
    """
    Compute accuracy, macro F1, sensitivity, precision from confusion matrix.
    conf: [C, C] where rows = true, cols = pred
    """
    C = conf.shape[0]
    total = conf.sum()
    acc = float(np.trace(conf) / total) if total > 0 else 0.0

    precisions, recalls, f1s = [], [], []
    for c in range(C):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "macro_sensitivity": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> Dict[str, float]:
    """Evaluate model on loader, return metrics."""
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
            conf[int(t), int(p)] += 1

    return compute_metrics_from_confmat(conf)


def main():
    parser = argparse.ArgumentParser("ViT-B/16 Training on Raabin-WBC")

    # Data
    parser.add_argument(
        "--train_dir", type=str, required=True,
        help="Path to training data directory (ImageFolder format)"
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--pretrained", action="store_true", default=True)

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # Scheduler
    parser.add_argument("--sched", type=str, default="cosine")
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Augmentation
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)

    # Runtime
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_root", type=str, default="./runs")

    # Checkpointing
    parser.add_argument("--eval_every_steps", type=int, default=500)
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--save_every_epoch", action="store_true", default=True)

    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(args.output_root)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run] {run_dir}")

    # -------------------------
    # Dataset
    # -------------------------
    train_full = ImageFolder(args.train_dir)
    num_classes = len(train_full.classes)
    class_names = train_full.classes
    class_to_idx = train_full.class_to_idx
    print(f"[Classes] {class_names} (num_classes={num_classes})")

    # Train/Val split
    n_total = len(train_full)
    n_val = int(round(n_total * args.val_ratio))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(train_full, [n_train, n_val], generator=g)
    print(f"[Split] train={len(train_subset)}, val={len(val_subset)}")

    # -------------------------
    # Model (create first to get correct data config)
    # -------------------------
    model = timm.create_model(
        args.model, pretrained=args.pretrained, num_classes=num_classes
    )
    model.to(device)

    # Get data config from model instance
    data_cfg = resolve_data_config({}, model=model)
    data_cfg["input_size"] = (3, args.input_size, args.input_size)
    print(f"[Data config] mean={data_cfg['mean']}, std={data_cfg['std']}")

    # -------------------------
    # Transforms
    # -------------------------
    train_tf = create_transform(
        input_size=data_cfg["input_size"],
        is_training=True,
        auto_augment=args.aa,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        interpolation=data_cfg.get("interpolation", "bicubic"),
        mean=data_cfg["mean"],
        std=data_cfg["std"],
    )
    eval_tf = create_transform(
        input_size=data_cfg["input_size"],
        is_training=False,
        interpolation=data_cfg.get("interpolation", "bicubic"),
        mean=data_cfg["mean"],
        std=data_cfg["std"],
    )

    train_ds = TransformSubset(train_subset, train_tf)
    val_ds = TransformSubset(val_subset, eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # -------------------------
    # Optimizer & Scheduler
    # -------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = create_optimizer_v2(
        model, opt="adamw", lr=args.lr, weight_decay=args.weight_decay
    )

    # Use create_scheduler_v2 for better step-level control
    updates_per_epoch = len(train_loader)
    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        sched=args.sched,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        warmup_lr=1e-7,
        updates_per_epoch=updates_per_epoch,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # -------------------------
    # Save config
    # -------------------------
    config = {
        **vars(args),
        "num_classes": num_classes,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "data_config": {
            "mean": list(data_cfg["mean"]),
            "std": list(data_cfg["std"]),
            "input_size": list(data_cfg["input_size"]),
            "interpolation": data_cfg.get("interpolation", "bicubic"),
        },
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # -------------------------
    # Logging
    # -------------------------
    metrics_path = run_dir / "metrics.jsonl"

    def log_metrics(row: Dict):
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -------------------------
    # Training Loop
    # -------------------------
    global_step = 0
    best_val_f1 = -1.0

    interval_loss_sum = 0.0
    interval_correct = 0
    interval_count = 0

    def save_checkpoint(name: str, extra: Dict = None):
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "config": config,
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, ckpt_dir / name)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            global_step += 1
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if args.clip_grad and args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()

            # Step-level scheduler update (for warmup)
            if scheduler is not None:
                scheduler.step_update(global_step)

            # Accumulate interval stats
            interval_loss_sum += float(loss.item()) * images.size(0)
            preds = torch.argmax(logits, dim=1)
            interval_correct += int((preds == targets).sum().item())
            interval_count += int(images.size(0))

            # Periodic evaluation
            do_eval = global_step % args.eval_every_steps == 0
            do_save = global_step % args.save_every_steps == 0

            if do_eval or do_save:
                train_loss = interval_loss_sum / max(1, interval_count)
                train_acc = interval_correct / max(1, interval_count)
                interval_loss_sum, interval_correct, interval_count = 0.0, 0, 0

                val_metrics = evaluate(model, val_loader, device, num_classes)

                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                log_metrics(row)

                print(
                    f"[Step {global_step:>6}] "
                    f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.4f} "
                    f"val_f1={val_metrics['macro_f1']:.4f}"
                )

                if do_save:
                    save_checkpoint(f"step_{global_step:06d}.pt")

                # Track best
                if val_metrics["macro_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["macro_f1"]
                    save_checkpoint("best.pt", {"best_val_f1": best_val_f1})

                model.train()

        # Epoch-level scheduler step
        if scheduler is not None:
            scheduler.step(epoch)

        # Save epoch checkpoint
        if args.save_every_epoch:
            save_checkpoint(f"epoch_{epoch:03d}.pt")

        dur = time.time() - epoch_start
        print(
            f"[Epoch {epoch}/{args.epochs}] {dur/60:.1f} min, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    # Final checkpoint
    save_checkpoint("final.pt")

    print(f"\n[Done] {run_dir}")
    print(f"  - config:  {run_dir / 'config.json'}")
    print(f"  - metrics: {metrics_path}")
    print(f"  - best:    {ckpt_dir / 'best.pt'}")
    print(f"  - final:   {ckpt_dir / 'final.pt'}")


if __name__ == "__main__":
    main()

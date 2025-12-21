"""PyTorch Lightning module for DUSA TTA."""

import csv
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from ..models import CombinedModel
from ..utils import calculate_accuracy


class DUSATTAModule(pl.LightningModule):
    """
    Lightning module for DUSA Test-Time Adaptation.
    No training phase - only test-time adaptation on each task.

    Uses automatic optimization with gradient accumulation support.
    """

    def __init__(
        self,
        model: CombinedModel,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        continual: bool = False,
        update_auxiliary: bool = True,
        update_task_norm_only: bool = True,
        update_pixel_adapter: bool = True,
        forward_mode: str = "normed_logits_with_logits",
        log_aux_metrics: bool = False,
        num_classes: int = 1000,
        sample_log_dir: Optional[str] = None,
        enable_sample_logging: bool = True,
        scheduler_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model: Combined discriminative + generative model
            learning_rate: Learning rate for Adam optimizer
            weight_decay: Weight decay for optimizer
            scheduler_config: Optional learning rate scheduler configuration
            continual: If False, reset model between tasks (fully TTA)
            update_auxiliary: Whether to update auxiliary model parameters
            update_task_norm_only: If True, only update norm layers in task model
            update_pixel_adapter: If True, update pixel adapter parameters (if present)
            forward_mode: Forward mode for combined model
            log_aux_metrics: Whether to log auxiliary metrics (loss_top_i, auc, etc.)
            num_classes: Number of classes for accuracy metrics
            sample_log_dir: Directory to dump per-sample CSV logs (one file per task)
            enable_sample_logging: Toggle per-sample CSV logging
        """
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.continual = continual
        self.update_auxiliary = update_auxiliary
        self.update_task_norm_only = update_task_norm_only
        self.update_pixel_adapter = update_pixel_adapter
        self.forward_mode = forward_mode
        self.log_aux_metrics = log_aux_metrics
        self.enable_sample_logging = enable_sample_logging
        self.sample_log_dir = Path(sample_log_dir) if sample_log_dir else None
        self.scheduler_config = scheduler_config or {}
        
        # Current task info (set by callback)
        self.current_task_name = "unknown"
        self.current_task_idx = 0
        self.current_task_log_path: Optional[Path] = None
        self._task_log_header_written = False
        self._sample_counter = 0

        # TorchMetrics for cumulative accuracy (automatically handles accumulation)
        self.train_acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.train_acc_top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)

        # Save initial model state for reset
        self.save_hyperparameters(ignore=["model"])
        self.initial_model_state = None

        # Configure model training mode
        self._configure_model()

    @staticmethod
    def _split_aux_metrics(aux_metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """
        Split auxiliary metrics into scalar-for-logging and tensor-for-CSV parts.
        Scalars (dim 0 tensors or python numbers) are safe to send to loggers (e.g., W&B).
        Higher-dim tensors are kept for per-sample CSV logging.
        """
        scalar_metrics: Dict[str, Any] = {}
        tensor_metrics: Dict[str, torch.Tensor] = {}
        for k, v in aux_metrics.items():
            if torch.is_tensor(v):
                if v.dim() == 0:
                    scalar_metrics[k] = v.detach()
                else:
                    tensor_metrics[k] = v
            elif isinstance(v, (float, int)):
                scalar_metrics[k] = v
        return scalar_metrics, tensor_metrics

    def _reset_metrics(self):
        """Reset accumulated metrics for a new task."""
        self.train_acc_top1.reset()
        self.train_acc_top5.reset()

    def _configure_model(self):
        """Configure which parameters to update."""
        # Set task model training mode
        self.model.set_task_train_mode(
            update_norm_only=self.update_task_norm_only,
            update_pixel_adapter=self.update_pixel_adapter,
        )

        # Set auxiliary model training mode
        if self.update_auxiliary:
            self.model.set_auxiliary_train_mode(update_flow=True)

    def _prepare_task_logging(self):
        """Initialize per-task CSV logging state."""
        self._task_log_header_written = False
        self._sample_counter = 0
        if not self.enable_sample_logging or self.sample_log_dir is None:
            self.current_task_log_path = None
            return

        self.sample_log_dir.mkdir(parents=True, exist_ok=True)
        safe_task_name = self.current_task_name.replace(os.sep, "_")
        log_name = f"task_{self.current_task_idx:02d}_{safe_task_name}.csv"
        self.current_task_log_path = self.sample_log_dir / log_name

    def save_initial_state(self):
        """Manually save initial model state before TTA begins."""
        self.initial_model_state = deepcopy(self.model.get_model_state())

    def _write_task_log_header(self, k: int, log_time: bool = False, log_reward: bool = False, log_arm: bool = False):
        """Write CSV header for the current task."""
        if self.current_task_log_path is None or self._task_log_header_written:
            return

        header: List[str] = [
            "sample_index",
            "label",
            "prediction",
            "correct",
            "gap_norm",
            "kendall_tau",
            "entropy",
        ]
        if log_time:
            header.append("timestep")
        if log_reward:
            header.append("bandit_reward")
        if log_arm:
            header.append("bandit_arm")
        for i in range(k):
            header.extend(
                [
                    f"class_{i}",
                    f"ori_logit_{i}",
                    f"norm_logit_{i}",
                    f"gen_loss_{i}",
                ]
            )
        header.append("image_path")

        with self.current_task_log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

        self._task_log_header_written = True

    def _record_sample_batch(
        self,
        batch: Dict[str, Any],
        logits: torch.Tensor,
        aux_metrics: Dict[str, torch.Tensor],
    ):
        """Append per-sample details to CSV for the current task."""
        if self.current_task_log_path is None or not aux_metrics:
            return

        required_keys = [
            "forward_idx",
            "selected_ori_logits",
            "selected_norm_logits",
            "per_class_loss",
            "sample_gap_norm",
            "sample_kendall_tau",
            "sample_entropy",
        ]
        if any(key not in aux_metrics for key in required_keys):
            return

        forward_idx = aux_metrics["forward_idx"].detach().cpu()
        selected_ori_logits = aux_metrics["selected_ori_logits"].detach().cpu()
        selected_norm_logits = aux_metrics["selected_norm_logits"].detach().cpu()
        per_class_loss = aux_metrics["per_class_loss"].detach().cpu()
        gap_norm = aux_metrics["sample_gap_norm"].detach().cpu()
        kendall = aux_metrics["sample_kendall_tau"].detach().cpu()
        entropy = aux_metrics["sample_entropy"].detach().cpu()

        selected_timesteps = None
        if "selected_timestep" in aux_metrics:
            selected_timesteps = aux_metrics["selected_timestep"].detach().cpu()
        elif "selected_timesteps" in aux_metrics:
            selected_timesteps = aux_metrics["selected_timesteps"].detach().cpu()

        bandit_rewards = aux_metrics.get("bandit_reward")
        if torch.is_tensor(bandit_rewards):
            bandit_rewards = bandit_rewards.detach().cpu()
        bandit_arms = aux_metrics.get("selected_arm")
        if torch.is_tensor(bandit_arms):
            bandit_arms = bandit_arms.detach().cpu()

        batch_size, k = forward_idx.shape
        self._write_task_log_header(
            k,
            log_time=selected_timesteps is not None,
            log_reward=bandit_rewards is not None,
            log_arm=bandit_arms is not None,
        )

        with torch.no_grad():
            preds = torch.argmax(logits.detach(), dim=-1).cpu()
        labels = batch["labels"].detach().cpu()
        correct = preds.eq(labels)

        indices = batch.get("indices")
        if indices is None:
            indices_tensor = torch.arange(self._sample_counter, self._sample_counter + batch_size)
        else:
            indices_tensor = indices.detach().cpu() if torch.is_tensor(indices) else torch.tensor(indices)

        image_paths = batch.get("image_paths", [""] * batch_size)

        with self.current_task_log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            for i in range(batch_size):
                row = [int(indices_tensor[i]), int(labels[i]), int(preds[i]), int(correct[i]), float(gap_norm[i]), float(kendall[i]), float(entropy[i])]
                if selected_timesteps is not None:
                    row.append(float(selected_timesteps[i]))
                if bandit_rewards is not None:
                    row.append(float(bandit_rewards[i]))
                if bandit_arms is not None:
                    row.append(int(bandit_arms[i]))
                for cls_idx in range(k):
                    row.extend(
                        [
                            int(forward_idx[i, cls_idx]),
                            float(selected_ori_logits[i, cls_idx]),
                            float(selected_norm_logits[i, cls_idx]),
                            float(per_class_loss[i, cls_idx]),
                        ]
                    )
                row.append(image_paths[i] if i < len(image_paths) else "")
                writer.writerow(row)

        self._sample_counter += batch_size

    def on_fit_start(self):
        """Save initial model state before TTA (only if not already saved)."""
        if self.initial_model_state is None:
            self.save_initial_state()

    def reset_model(self):
        """Reset model to initial state (for fully TTA mode)."""
        if self.initial_model_state is not None:
            self.model.load_model_state(self.initial_model_state)
            self._configure_model()  # Re-configure training mode after reset
            if hasattr(self.model, "reset_time_selector"):
                self.model.reset_time_selector()
            self.print(f"Model reset to initial state")

    def set_task_info(self, task_name: str, task_idx: int):
        """Set current task information for logging."""
        self.current_task_name = task_name
        self.current_task_idx = task_idx

    def forward(self, batch: Dict[str, Any], batch_idx: int):
        """
        Forward pass for TTA.

        Args:
            batch: Dict with "images" (B, 3, 224, 224) in [0, 1], "labels"
            batch_idx: Batch index

        Returns:
            logits: Predictions (B, num_classes)
            loss: Auxiliary loss (scalar)
            metrics: Dict of metrics
            aux_metrics: Raw auxiliary metrics (may include tensors)
        """
        images = batch["images"]  # (B, 3, 224, 224) in [0, 1]
        labels = batch["labels"]

        # Prepare batch infos for auxiliary model
        batch_infos = {
            "task_name": self.current_task_name,
            "task_idx": self.current_task_idx,
            "step": batch_idx,
        }

        # Forward pass
        logits, features, loss_with_metrics = self.model(
            images=images,
            mode=self.forward_mode,
            batch_infos=batch_infos,
        )

        # Unpack auxiliary loss and metrics
        if loss_with_metrics is not None:
            auxiliary_loss, raw_aux_metrics = loss_with_metrics
        else:
            auxiliary_loss = None
            raw_aux_metrics = {}

        # Separate scalar (for logger) vs tensor (for CSV) aux metrics
        scalar_aux_metrics, tensor_aux_metrics = self._split_aux_metrics(raw_aux_metrics)

        # Calculate accuracy
        accuracy_metrics = calculate_accuracy(logits, labels, topk=(1, 5))

        # Combine metrics
        metrics = {
            "loss": auxiliary_loss.item() if auxiliary_loss is not None else 0.0,
            **accuracy_metrics,
        }

        if self.log_aux_metrics and scalar_aux_metrics:
            for k, v in scalar_aux_metrics.items():
                metrics[k] = v.item() if torch.is_tensor(v) else v

        return logits, auxiliary_loss, metrics, tensor_aux_metrics

    def training_step(self, batch, batch_idx):
        """
        TTA step using automatic optimization.
        Supports gradient accumulation via trainer config.
        """
        # Forward pass
        logits, loss, metrics, aux_metrics = self(batch, batch_idx)
        labels = batch["labels"]

        # Update TorchMetrics (automatically accumulates)
        self.train_acc_top1.update(logits, labels)
        self.train_acc_top5.update(logits, labels)

        # Get cumulative accuracy (TorchMetrics handles the math)
        avg_top1 = self.train_acc_top1.compute() * 100.0
        avg_top5 = self.train_acc_top5.compute() * 100.0

        # Log metrics with task-specific names to avoid overwriting
        # Progress bar: only show loss and avg_top1 (without task prefix for readability)
        self.log("loss", metrics["loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("avg_top1", avg_top1, on_step=True, on_epoch=False, prog_bar=True)

        # Logger: use task-specific prefix to avoid overwriting across tasks
        task_prefix = f"task_{self.current_task_idx:02d}"
        self.log(f"{task_prefix}/loss", metrics["loss"], on_step=True, on_epoch=False, prog_bar=False)
        self.log(f"{task_prefix}/avg_top1", avg_top1, on_step=True, on_epoch=False, prog_bar=False)
        self.log(f"{task_prefix}/avg_top5", avg_top5, on_step=True, on_epoch=False, prog_bar=False)

        # Log auxiliary metrics if enabled
        if self.log_aux_metrics:
            aux_keys = [k for k in metrics.keys() if k not in ["loss", "top1", "top5"]]
            for k in aux_keys:
                self.log(f"{task_prefix}/aux/{k}", metrics[k], on_step=True, on_epoch=False, prog_bar=False)

        # Per-sample logging (CSV)
        self._record_sample_batch(batch, logits, aux_metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step (no adaptation, just evaluation)."""
        images = batch["images"]  # (B, 3, 224, 224) in [0, 1]
        labels = batch["labels"]

        # Forward without auxiliary (no adaptation)
        with torch.no_grad():
            logits, _, _ = self.model(
                images=images,
                mode="logits",
            )

        # Calculate accuracy
        metrics = calculate_accuracy(logits, labels, topk=(1, 5))

        # Log metrics
        task_prefix = f"task_{self.current_task_idx:02d}"
        self.log(f"{task_prefix}/val_top1", metrics["top1"], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{task_prefix}/val_top5", metrics["top5"], on_step=False, on_epoch=True, prog_bar=False)

        return metrics

    def test_step(self, batch, batch_idx):
        """Test step (same as validation)."""
        images = batch["images"]  # (B, 3, 224, 224) in [0, 1]
        labels = batch["labels"]

        # Forward without auxiliary
        with torch.no_grad():
            logits, _, _ = self.model(
                images=images,
                mode="logits",
            )

        # Calculate accuracy
        metrics = calculate_accuracy(logits, labels, topk=(1, 5))

        # Log metrics
        task_prefix = f"task_{self.current_task_idx:02d}"
        self.log(f"{task_prefix}/test_top1", metrics["top1"], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{task_prefix}/test_top5", metrics["top5"], on_step=False, on_epoch=True, prog_bar=False)

        return metrics

    def configure_optimizers(self):
        """Configure optimizer for TTA."""
        # Collect trainable parameters
        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param)

        if len(params) == 0:
            raise RuntimeError("No trainable parameters found!")

        # Adam optimizer (standard for TTA)
        optimizer = torch.optim.Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = self._build_lr_scheduler(optimizer)
        if scheduler is None:
            return optimizer

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _build_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        """Build a learning rate scheduler from config."""
        if not self.scheduler_config:
            return None

        name = str(self.scheduler_config.get("name", "")).lower()
        if name in {"", "none", "null"}:
            return None

        interval = self.scheduler_config.get("interval", "step")
        frequency = int(self.scheduler_config.get("frequency", 1))
        monitor = self.scheduler_config.get("monitor")

        if name == "cosine":
            t_max = int(self.scheduler_config.get("t_max", 100))
            eta_min = float(self.scheduler_config.get("eta_min", 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min
            )
        elif name == "step":
            step_size = int(self.scheduler_config.get("step_size", 1))
            gamma = float(self.scheduler_config.get("gamma", 0.1))
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {name}")

        scheduler_config = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
        }
        if monitor:
            scheduler_config["monitor"] = monitor

        return scheduler_config
    
    def on_train_epoch_start(self):
        """Called at the start of each task (epoch)."""
        # Reset metrics for new task
        self._reset_metrics()
        self._prepare_task_logging()

    def on_train_epoch_end(self):
        """Called at the end of each task (epoch). Log final metrics."""
        # Get final cumulative accuracy from TorchMetrics
        final_avg_top1 = self.train_acc_top1.compute() * 100.0
        final_avg_top5 = self.train_acc_top5.compute() * 100.0

        # Log final task accuracy with task-specific prefix
        task_prefix = f"task_{self.current_task_idx:02d}"
        self.log(f"{task_prefix}/final_top1", final_avg_top1, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{task_prefix}/final_top5", final_avg_top5, on_step=False, on_epoch=True, prog_bar=False)

        # Print task summary
        self.print(f"\nTask '{self.current_task_name}' completed: " f"Avg Top-1={final_avg_top1:.2f}%, Avg Top-5={final_avg_top5:.2f}%")

    def get_final_accuracy(self) -> Dict[str, float]:
        """Get final accuracy for current task."""
        return {
            "top1": self.train_acc_top1.compute().item() * 100.0,
            "top5": self.train_acc_top5.compute().item() * 100.0,
        }

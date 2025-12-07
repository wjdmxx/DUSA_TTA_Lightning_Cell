"""PyTorch Lightning module for DUSA TTA."""
import torch
import pytorch_lightning as pl
from typing import Optional, Dict, Any
from copy import deepcopy

from ..models import CombinedModel
from ..utils import calculate_accuracy


class DUSATTAModule(pl.LightningModule):
    """
    Lightning module for DUSA Test-Time Adaptation.
    No training phase - only test-time adaptation on each task.
    """
    
    def __init__(
        self,
        model: CombinedModel,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        continual: bool = False,
        update_auxiliary: bool = True,
        update_task_norm_only: bool = True,
        tta_step: int = 1,
        forward_mode: str = "normed_logits_with_logits",
        log_aux_metrics: bool = True,
    ):
        """
        Args:
            model: Combined discriminative + generative model
            learning_rate: Learning rate for Adam optimizer
            weight_decay: Weight decay for optimizer
            continual: If False, reset model between tasks (fully TTA)
            update_auxiliary: Whether to update auxiliary model parameters
            update_task_norm_only: If True, only update norm layers in task model
            tta_step: Number of adaptation steps per batch
            forward_mode: Forward mode for combined model
            log_aux_metrics: Whether to log auxiliary metrics (loss_top_i, auc, etc.)
        """
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.continual = continual
        self.update_auxiliary = update_auxiliary
        self.update_task_norm_only = update_task_norm_only
        self.tta_step = tta_step
        self.forward_mode = forward_mode
        self.log_aux_metrics = log_aux_metrics
        
        # Save initial model state for reset
        self.save_hyperparameters(ignore=["model"])
        self.initial_model_state = None
        
        # Configure model training mode
        self._configure_model()
    
    def _configure_model(self):
        """Configure which parameters to update."""
        # Set task model training mode
        self.model.set_task_train_mode(
            update_all=False,
            update_norm_only=self.update_task_norm_only,
        )
        
        # Set auxiliary model training mode
        if self.update_auxiliary:
            self.model.set_auxiliary_train_mode(update_flow=True)
    
    def on_fit_start(self):
        """Save initial model state before TTA."""
        self.initial_model_state = deepcopy(self.model.get_model_state())
    
    def reset_model(self):
        """Reset model to initial state (for fully TTA mode)."""
        if self.initial_model_state is not None:
            self.model.load_model_state(self.initial_model_state)
            self.print(f"Model reset to initial state")
    
    def forward(self, batch: Dict[str, Any], batch_idx: int, task_info: Optional[Dict] = None):
        """
        Forward pass for TTA.
        
        Args:
            batch: Dict with "task_images", "raw_images", "labels"
            batch_idx: Batch index
            task_info: Optional dict with task metadata (task_name, step, all_steps)
        
        Returns:
            logits: Predictions (B, num_classes)
            loss: Auxiliary loss (scalar)
            metrics: Dict of metrics
        """
        task_images = batch["task_images"]
        raw_images = batch["raw_images"]
        labels = batch["labels"]
        
        # Prepare batch infos for auxiliary model
        batch_infos = task_info or {}
        batch_infos["step"] = batch_idx
        
        # Forward pass
        logits, features, loss_with_metrics = self.model(
            images=task_images,
            raw_images=raw_images,
            mode=self.forward_mode,
            batch_infos=batch_infos,
        )
        
        # Unpack auxiliary loss and metrics
        if loss_with_metrics is not None:
            auxiliary_loss, aux_metrics = loss_with_metrics
        else:
            auxiliary_loss = None
            aux_metrics = {}
        
        # Calculate accuracy
        accuracy_metrics = calculate_accuracy(logits, labels, topk=(1, 5))
        
        # Combine metrics
        metrics = {
            "loss": auxiliary_loss.item() if auxiliary_loss is not None else 0.0,
            **accuracy_metrics,
        }
        
        if self.log_aux_metrics and aux_metrics:
            metrics.update({k: v.item() if torch.is_tensor(v) else v for k, v in aux_metrics.items()})
        
        return logits, auxiliary_loss, metrics
    
    def training_step(self, batch, batch_idx):
        """
        TTA step (not traditional training).
        Note: We use training_step for TTA adaptation, even though it's test-time.
        """
        # Get task info from trainer's current task
        task_info = getattr(self.trainer, "current_task_info", {})
        
        # Multiple TTA steps per batch if configured
        for step_i in range(self.tta_step):
            logits, loss, metrics = self(batch, batch_idx, task_info)
            
            # Only optimize if we have auxiliary loss
            if loss is not None:
                # Manual optimization (Lightning automatic optimization disabled)
                if self.automatic_optimization:
                    return loss
                else:
                    # Manual backward and optimizer step
                    opt = self.optimizers()
                    opt.zero_grad()
                    self.manual_backward(loss)
                    opt.step()
        
        # Log metrics
        self.log_dict(
            {f"train/{k}": v for k, v in metrics.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step (no adaptation, just evaluation)."""
        task_images = batch["task_images"]
        labels = batch["labels"]
        
        # Forward without auxiliary (no adaptation)
        logits, _, _ = self.model(
            images=task_images,
            raw_images=None,
            mode="logits",
        )
        
        # Calculate accuracy
        metrics = calculate_accuracy(logits, labels, topk=(1, 5))
        
        # Log metrics
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        return metrics
    
    def test_step(self, batch, batch_idx):
        """Test step (same as validation)."""
        task_images = batch["task_images"]
        labels = batch["labels"]
        
        # Forward without auxiliary
        logits, _, _ = self.model(
            images=task_images,
            raw_images=None,
            mode="logits",
        )
        
        # Calculate accuracy
        metrics = calculate_accuracy(logits, labels, topk=(1, 5))
        
        # Log metrics
        self.log_dict(
            {f"test/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
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
        
        return optimizer
    
    def on_train_epoch_start(self):
        """Called at the start of each task (epoch in our case)."""
        # Reset model if not continual
        if not self.continual and self.current_epoch > 0:
            self.reset_model()
            # Reconfigure optimizer after reset
            self.trainer.strategy.setup_optimizers(self.trainer)

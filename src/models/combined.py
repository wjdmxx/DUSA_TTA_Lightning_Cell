"""Combined model wrapper for TTA."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

from .discriminative import TimmClassifier
from .generative import REPASiT


class CombinedModel(nn.Module):
    """
    Unified model combining discriminative classifier and generative auxiliary model.
    Implements multiple forward modes for DUSA TTA.
    """
    
    def __init__(
        self,
        discriminative_model: TimmClassifier,
        generative_model: Optional[REPASiT] = None,
    ):
        """
        Args:
            discriminative_model: Task classifier (e.g., ConvNeXt from timm)
            generative_model: Auxiliary model (e.g., REPA SiT), optional
        """
        super().__init__()
        
        self.task_model = discriminative_model
        self.auxiliary_model = generative_model
        
        # Get feature dimension for potential feature alignment
        self.feature_dim = self.task_model.get_feature_dim()
    
    def task_forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward through task model only.
        
        Args:
            images: Preprocessed images (B, 3, 224, 224)
            return_features: Whether to return pre-logits features
        
        Returns:
            logits: (B, num_classes)
            features: (B, feature_dim) or None
        """
        return self.task_model(images, return_features=return_features)
    
    def forward(
        self,
        images: torch.Tensor,
        raw_images: Optional[List[torch.Tensor]] = None,
        mode: str = "logits",
        batch_infos: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with multiple modes.
        
        Args:
            images: Preprocessed images for task model (B, 3, 224, 224)
            raw_images: Raw images for auxiliary model (list of (3, H, W) in BGR [0, 255])
            mode: Forward mode, one of:
                - "logits": Return logits only, no auxiliary
                - "normed_logits": Auxiliary uses L2-normalized logits
                - "normed_logits_with_logits": DUSA default, auxiliary uses both
            batch_infos: Optional dict for auxiliary model (step, all_steps, task_name)
        
        Returns:
            logits: Task model logits (B, num_classes)
            features: Task model features (B, feature_dim) or None
            auxiliary_loss: Auxiliary loss (scalar) or None
        """
        # Get logits and features from task model
        logits, features = self.task_model(images, return_features=True)
        
        # If no auxiliary model or mode is logits-only, return early
        if self.auxiliary_model is None or mode == "logits":
            return logits, features, None
        
        # Prepare inputs for auxiliary model
        if raw_images is None:
            raise ValueError("raw_images required for auxiliary model forward")
        
        # Compute auxiliary loss based on mode
        if mode == "normed_logits":
            # Normalize logits to unit sphere
            normed_logits = F.normalize(logits, p=2, dim=-1)
            auxiliary_loss, aux_metrics = self.auxiliary_model(
                images=raw_images,
                normed_logits=normed_logits,
                ori_logits=logits,
                batch_infos=batch_infos,
            )
        
        elif mode == "normed_logits_with_logits":
            # DUSA default: use both normalized and original logits
            normed_logits = F.normalize(logits, p=2, dim=-1)
            auxiliary_loss, aux_metrics = self.auxiliary_model(
                images=raw_images,
                normed_logits=normed_logits,
                ori_logits=logits,
                batch_infos=batch_infos,
            )
        
        else:
            raise ValueError(f"Unknown forward mode: {mode}")
        
        # Package loss with metrics
        loss_with_metrics = (auxiliary_loss, aux_metrics)
        
        return logits, features, loss_with_metrics
    
    def set_task_train_mode(
        self,
        update_all: bool = False,
        update_norm_only: bool = True,
    ):
        """Configure task model training mode."""
        self.task_model.set_train_mode(
            update_all=update_all,
            update_norm_only=update_norm_only,
        )
    
    def set_auxiliary_train_mode(self, update_flow: bool = True):
        """Configure auxiliary model training mode."""
        if self.auxiliary_model is not None:
            self.auxiliary_model.set_train_mode(update_flow=update_flow)
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current model state for reset."""
        state = {
            "task_model": self.task_model.state_dict(),
        }
        if self.auxiliary_model is not None:
            state["auxiliary_model"] = self.auxiliary_model.state_dict()
        return state
    
    def load_model_state(self, state: Dict[str, torch.Tensor]):
        """Load model state for reset."""
        self.task_model.load_state_dict(state["task_model"])
        if self.auxiliary_model is not None and "auxiliary_model" in state:
            self.auxiliary_model.load_state_dict(state["auxiliary_model"])


def create_combined_model(
    discriminative_config: Dict,
    generative_config: Optional[Dict] = None,
) -> CombinedModel:
    """
    Factory function to create combined model from configs.
    
    Args:
        discriminative_config: Config dict for discriminative model
            {
                "model_name": "convnext_large",
                "pretrained": True,
                "checkpoint_path": None,
                "num_classes": 1000,
            }
        generative_config: Config dict for generative model (optional)
            {
                "sit_model_name": "SiT-XL/2",
                "sit_checkpoint": "path/to/ckpt.pt",
                "num_classes": 1000,
                "topk": 4,
                "rand_budget": 2,
                "temperature": 1.0,
                ...
            }
    
    Returns:
        CombinedModel instance
    """
    from .discriminative import create_classifier
    from .generative import create_repa_sit
    
    # Build discriminative model
    disc_model = create_classifier(**discriminative_config)
    
    # Build generative model if config provided
    gen_model = None
    if generative_config is not None:
        gen_model = create_repa_sit(**generative_config)
    
    return CombinedModel(
        discriminative_model=disc_model,
        generative_model=gen_model,
    )

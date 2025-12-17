"""Combined model wrapper for TTA."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any

from .discriminative import TimmClassifier
from .generative import REPASiT
from .pixel_adapter import create_pixel_adapter


class CombinedModel(nn.Module):
    """
    Unified model combining discriminative classifier and generative auxiliary model.
    Implements multiple forward modes for DUSA TTA.
    
    Architecture:
    1. Input: images in [0, 1] range (B, 3, 224, 224)
    2. Pixel Adapter: learns perturbations, output clamped to [0, 1]
    3A. Discriminative branch: normalize + classify (gradients flow to adapter)
    3B. Generative branch: resize to 256, scale to [-1, 1], detach (no gradients to adapter)
    """
    
    def __init__(
        self,
        discriminative_model: TimmClassifier,
        generative_model: Optional[REPASiT] = None,
        pixel_adapter: Optional[nn.Module] = None,
    ):
        """
        Args:
            discriminative_model: Task classifier (e.g., ConvNeXt from timm)
            generative_model: Auxiliary model (e.g., REPA SiT), optional
            pixel_adapter: Pixel-level adapter for input perturbations, optional
        """
        super().__init__()
        
        self.task_model = discriminative_model
        self.auxiliary_model = generative_model
        self.pixel_adapter = pixel_adapter
        
        if self.pixel_adapter is not None:
            print(f"Pixel Adapter enabled in CombinedModel: {type(self.pixel_adapter).__name__}")
    
    def task_forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward through task model only (with adapter if present).
        
        Args:
            images: Images in [0, 1] range (B, 3, 224, 224)
            return_features: Whether to return pre-logits features
        
        Returns:
            logits: (B, num_classes)
            features: (B, feature_dim) or None
        """
        # Apply pixel adapter if present
        if self.pixel_adapter is not None:
            images = self.pixel_adapter(images)
            images = images.clamp(0, 1)
        
        return self.task_model(images, return_features=return_features)
    
    def forward(
        self,
        images: torch.Tensor,
        mode: str = "logits",
        batch_infos: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, Dict]]]:
        """
        Forward pass with multiple modes.
        
        Args:
            images: Images in [0, 1] range (B, 3, 224, 224)
            mode: Forward mode, one of:
                - "logits": Return logits only, no auxiliary
                - "normed_logits": Auxiliary uses L2-normalized logits
                - "normed_logits_with_logits": DUSA default, auxiliary uses both
            batch_infos: Optional dict for auxiliary model (step, all_steps, task_name)
        
        Returns:
            logits: Task model logits (B, num_classes)
            features: Task model features (B, feature_dim) or None
            loss_with_metrics: Tuple of (auxiliary_loss, aux_metrics) or None
        """
        # Step 1: Apply pixel adapter (if present)
        if self.pixel_adapter is not None:
            adapted_images = self.pixel_adapter(images)
            adapted_images = adapted_images.clamp(0, 1)
        else:
            adapted_images = images
        
        # Step 2: Discriminative branch (gradients flow through adapter)
        logits, features = self.task_model(adapted_images, return_features=False)
        
        # If no auxiliary model or mode is logits-only, return early
        if self.auxiliary_model is None or mode == "logits":
            return logits, features, None
        
        # Step 3: Generative branch (no gradients to adapter)
        # Detach adapted images for diffusion branch
        adapted_images_detached = adapted_images.detach()
        
        # Compute auxiliary loss based on mode
        if mode == "normed_logits":
            # Normalize logits to unit sphere
            normed_logits = F.normalize(logits, p=2, dim=-1)
            auxiliary_loss, aux_metrics = self.auxiliary_model(
                images=adapted_images_detached,
                normed_logits=normed_logits,
                ori_logits=logits,
                batch_infos=batch_infos,
            )
        
        elif mode == "normed_logits_with_logits":
            # DUSA default: use both normalized and original logits
            normed_logits = F.normalize(logits, p=2, dim=-1)
            auxiliary_loss, aux_metrics = self.auxiliary_model(
                images=adapted_images_detached,
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
        update_norm_only: bool = True,
        update_pixel_adapter: bool = True,
    ):
        """Configure task model and pixel adapter training mode."""
        self.task_model.set_train_mode(update_norm_only=update_norm_only)
        
        # Pixel adapter is trainable if enabled
        if self.pixel_adapter is not None:
            self.pixel_adapter.requires_grad_(update_pixel_adapter)
            if update_pixel_adapter:
                print(f"Pixel adapter parameters are trainable")
    
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
        if self.pixel_adapter is not None:
            state["pixel_adapter"] = self.pixel_adapter.state_dict()
        return state
    
    def load_model_state(self, state: Dict[str, torch.Tensor]):
        """Load model state for reset."""
        self.task_model.load_state_dict(state["task_model"])
        if self.auxiliary_model is not None and "auxiliary_model" in state:
            self.auxiliary_model.load_state_dict(state["auxiliary_model"])
        if self.pixel_adapter is not None and "pixel_adapter" in state:
            self.pixel_adapter.load_state_dict(state["pixel_adapter"])
    
    def get_pixel_adapter_stats(self, x: torch.Tensor) -> Optional[Dict[str, float]]:
        """Get pixel adapter perturbation statistics for monitoring."""
        if self.pixel_adapter is not None and hasattr(self.pixel_adapter, 'get_perturbation_stats'):
            return self.pixel_adapter.get_perturbation_stats(x)
        return None
    
    def get_pixel_adapter_scale(self) -> Optional[float]:
        """Get current effective scale of pixel adapter."""
        if self.pixel_adapter is not None and hasattr(self.pixel_adapter, 'get_effective_scale'):
            return self.pixel_adapter.get_effective_scale().item()
        return None

    def reset_time_selector(self):
        """Reset auxiliary timestep selector (bandit) if present."""
        if self.auxiliary_model is not None and hasattr(self.auxiliary_model, "reset_time_selector"):
            self.auxiliary_model.reset_time_selector()


def create_combined_model(
    discriminative_config: Dict,
    generative_config: Optional[Dict] = None,
    pixel_adapter_config: Optional[Dict[str, Any]] = None,
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
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
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
        pixel_adapter_config: Config dict for pixel adapter (optional)
            {
                "type": "standard",
                "in_channels": 3,
                "hidden_channels": 32,
                "num_blocks": 2,
                "max_scale": 0.15,
                "use_spatial_attention": True,
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
    
    # Build pixel adapter if config provided
    pixel_adapter = None
    if pixel_adapter_config is not None:
        pixel_adapter = create_pixel_adapter(
            adapter_type=pixel_adapter_config.get("type", "standard"),
            in_channels=pixel_adapter_config.get("in_channels", 3),
            hidden_channels=pixel_adapter_config.get("hidden_channels", 32),
            num_blocks=pixel_adapter_config.get("num_blocks", 2),
            max_scale=pixel_adapter_config.get("max_scale", 0.15),
            use_spatial_attention=pixel_adapter_config.get("use_spatial_attention", True),
        )
    
    return CombinedModel(
        discriminative_model=disc_model,
        generative_model=gen_model,
        pixel_adapter=pixel_adapter,
    )

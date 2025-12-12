"""Discriminative models using timm."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional, Dict, Any, List


class TimmClassifier(nn.Module):
    """Wrapper for timm models with feature extraction support.
    
    Now receives images in [0, 1] range and does normalization internally.
    """
    
    def __init__(
        self,
        model_name: str = "convnext_large",
        pretrained: bool = True,
        num_classes: int = 1000,
        checkpoint_path: Optional[str] = None,
        # Normalization config (ImageNet defaults)
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            model_name: timm model name (e.g., 'convnext_large', 'vit_base_patch16_224')
            pretrained: Load ImageNet pretrained weights
            num_classes: Number of output classes
            checkpoint_path: Path to custom checkpoint (overrides pretrained)
            normalize_mean: Mean for normalization (ImageNet default)
            normalize_std: Std for normalization (ImageNet default)
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Register normalization parameters as buffers (move with model to device)
        self.register_buffer(
            'normalize_mean', 
            torch.tensor(normalize_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'normalize_std', 
            torch.tensor(normalize_std).view(1, 3, 1, 1)
        )
        
        # Create model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained and checkpoint_path is None,
            num_classes=num_classes,
        )
        
        # Load custom checkpoint if provided
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # Handle different checkpoint formats
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # Remove 'model.' prefix if present (Lightning checkpoints)
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        # Get model info for feature extraction
        self.feature_info = self.model.feature_info if hasattr(self.model, "feature_info") else None
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input images.
        
        Args:
            x: Input tensor (B, 3, H, W) in [0, 1] range
            
        Returns:
            Normalized tensor with ImageNet mean/std
        """
        return (x - self.normalize_mean) / self.normalize_std
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional feature extraction.
        
        Args:
            x: Input tensor (B, 3, H, W) in [0, 1] range
            return_features: If True, return (logits, pre_logits_features)
        
        Returns:
            logits: (B, num_classes)
            features: (B, feature_dim) or None
        """
        # Normalize input (expects [0, 1] range)
        x = self.normalize(x)
        
        if return_features:
            # Extract pre-logits features (before final classifier)
            features = self.model.forward_features(x)
            
            # Handle different feature formats
            if isinstance(features, (tuple, list)):
                features = features[-1]  # Take last feature map
            
            # Global pool if needed (for conv models)
            if features.dim() == 4:  # (B, C, H, W)
                features = self.model.global_pool(features)
                if features.dim() == 4:  # Still 4D after pool
                    features = features.flatten(1)
            elif features.dim() == 3:  # (B, N, C) for ViT
                features = features[:, 0]  # Take CLS token
            
            # Get logits
            logits = self.model.head(features)
            
            return logits, features
        else:
            # Standard forward
            logits = self.model(x)
            return logits, None
    
    def get_feature_dim(self) -> int:
        """Get dimension of pre-logits features."""
        # Try to get from model head
        if hasattr(self.model, "head"):
            head = self.model.head
            if isinstance(head, nn.Linear):
                return head.in_features
            elif isinstance(head, nn.Sequential):
                # Find first Linear layer
                for module in head:
                    if isinstance(module, nn.Linear):
                        return module.in_features
        
        # Fallback: forward a dummy input
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            _, features = self.forward(dummy, return_features=True)
            return features.shape[1]
    
    def set_train_mode(
        self,
        update_norm_only: bool = True,
    ):
        """
        Configure which parameters to update during TTA.
        
        Args:
            update_norm_only: If True, only update normalization layers (BN, LN, GN) in backbone
        """
        if not update_norm_only:
            self.model.requires_grad_(True)
        else:
            # Freeze all parameters in backbone
            self.model.requires_grad_(False)
            
            # Only update normalization layers in backbone
            for name, module in self.model.named_modules():
                if isinstance(module, (
                    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                    nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                    nn.InstanceNorm2d, nn.InstanceNorm3d
                )):
                    module.requires_grad_(True)
                    # For BN: use only current batch statistics
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        if hasattr(module, "track_running_stats"):
                            module.track_running_stats = False
                            module.running_mean = None
                            module.running_var = None


def create_convnext_large(
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    num_classes: int = 1000,
) -> TimmClassifier:
    """Create ConvNeXt-Large model."""
    return TimmClassifier(
        model_name="convnext_large",
        pretrained=pretrained,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
    )


def create_classifier(
    model_name: str,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    num_classes: int = 1000,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    # Legacy params (ignored for backward compatibility)
    use_pixel_adapter: bool = False,
    pixel_adapter_config: Optional[Dict[str, Any]] = None,
) -> TimmClassifier:
    """Factory function to create any timm classifier."""
    return TimmClassifier(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

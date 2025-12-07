"""Discriminative models using timm."""
import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional


class TimmClassifier(nn.Module):
    """Wrapper for timm models with feature extraction support."""
    
    def __init__(
        self,
        model_name: str = "convnext_large",
        pretrained: bool = True,
        num_classes: int = 1000,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Args:
            model_name: timm model name (e.g., 'convnext_large', 'vit_base_patch16_224')
            pretrained: Load ImageNet pretrained weights
            num_classes: Number of output classes
            checkpoint_path: Path to custom checkpoint (overrides pretrained)
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
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
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional feature extraction.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_features: If True, return (logits, pre_logits_features)
        
        Returns:
            logits: (B, num_classes)
            features: (B, feature_dim) or None
        """
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
        update_all: bool = False,
        update_norm_only: bool = True,
    ):
        """
        Configure which parameters to update during TTA.
        
        Args:
            update_all: If True, make all parameters trainable
            update_norm_only: If True, only update normalization layers (BN, LN, GN)
        """
        if update_all:
            self.model.requires_grad_(True)
            return
        
        # Freeze all parameters
        self.model.requires_grad_(False)
        
        if update_norm_only:
            # Only update normalization layers
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
) -> TimmClassifier:
    """Factory function to create any timm classifier."""
    return TimmClassifier(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
    )

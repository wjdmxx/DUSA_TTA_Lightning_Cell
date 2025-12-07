"""VAE encoder for latent space."""
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from typing import Optional


class VAEEncoder(nn.Module):
    """Wrapper for Diffusers VAE with encoding only."""
    
    def __init__(
        self,
        pretrained_model: str = "stabilityai/sd-vae-ft-ema",
        scaling_factor: float = 0.18215,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            pretrained_model: Hugging Face model ID
            scaling_factor: VAE latent scaling factor
            device: Device to load model on (None = auto)
        """
        super().__init__()
        
        self.scaling_factor = scaling_factor
        self.vae = AutoencoderKL.from_pretrained(pretrained_model)
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        if device is not None:
            self.vae.to(device)
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space.
        
        Args:
            x: Image tensor (B, 3, H, W) in range [-1, 1]
        
        Returns:
            z: Latent tensor (B, 4, H//8, W//8)
        """
        latent_dist = self.vae.encode(x).latent_dist
        z = latent_dist.mean  # Use mean, not sample
        z = z * self.scaling_factor
        return z
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image space.
        
        Args:
            z: Latent tensor (B, 4, H//8, W//8)
        
        Returns:
            x: Image tensor (B, 3, H, W) in range [-1, 1]
        """
        z = z / self.scaling_factor
        x = self.vae.decode(z).sample
        return x


def create_vae_encoder(
    pretrained_model: str = "stabilityai/sd-vae-ft-ema",
    scaling_factor: float = 0.18215,
    device: Optional[torch.device] = None,
) -> VAEEncoder:
    """Factory function to create VAE encoder."""
    return VAEEncoder(
        pretrained_model=pretrained_model,
        scaling_factor=scaling_factor,
        device=device,
    )

"""Flow scheduler for REPA SiT."""
import torch
import torch.nn as nn
from typing import Tuple


class FlowScheduler(nn.Module):
    """
    Flow scheduler implementing different interpolation schemes.
    
    General form: x_t = alpha(t) * x0 + sigma(t) * z
    Velocity: v_t = alpha'(t) * x0 + sigma'(t) * z
    """
    
    def __init__(self, scheduler_type: str = "linear"):
        """
        Args:
            scheduler_type: One of ["linear", "reverse_linear"]
                - linear: alpha(t) = 1-t, sigma(t) = t (REPA default)
                - reverse_linear: alpha(t) = t, sigma(t) = 1-t
        """
        super().__init__()
        self.scheduler_type = scheduler_type
        
        if scheduler_type not in ["linear", "reverse_linear"]:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def get_coefficients(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get interpolation and velocity coefficients.
        
        Args:
            t: Time values (B,) or (B, 1, 1, 1)
        
        Returns:
            alpha: Coefficient for x0 in x_t
            sigma: Coefficient for z in x_t
            alpha_dot: Derivative of alpha (for velocity)
            sigma_dot: Derivative of sigma (for velocity)
        """
        if self.scheduler_type == "linear":
            # x_t = (1-t) * x0 + t * z
            alpha = 1.0 - t
            sigma = t
            alpha_dot = torch.full_like(t, -1.0)
            sigma_dot = torch.full_like(t, 1.0)
        
        elif self.scheduler_type == "reverse_linear":
            # x_t = t * x0 + (1-t) * z
            alpha = t
            sigma = 1.0 - t
            alpha_dot = torch.full_like(t, 1.0)
            sigma_dot = torch.full_like(t, -1.0)
        
        return alpha, sigma, alpha_dot, sigma_dot
    
    def get_xt(
        self,
        x0: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute x_t = alpha(t) * x0 + sigma(t) * z.
        
        Args:
            x0: Clean latent (B, C, H, W)
            z: Noise (B, C, H, W)
            t: Time (B,)
        
        Returns:
            x_t: Noised latent (B, C, H, W)
        """
        # Expand t to match x0 dimensions
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        
        alpha, sigma, _, _ = self.get_coefficients(t)
        x_t = alpha * x0 + sigma * z
        return x_t
    
    def get_vt(
        self,
        x0: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity v_t = alpha'(t) * x0 + sigma'(t) * z.
        
        This is the target for the flow model to learn.
        
        Args:
            x0: Clean latent (B, C, H, W)
            z: Noise (B, C, H, W)
            t: Time (B,)
        
        Returns:
            v_t: Target velocity (B, C, H, W)
        """
        # Expand t to match x0 dimensions
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        
        _, _, alpha_dot, sigma_dot = self.get_coefficients(t)
        v_t = alpha_dot * x0 + sigma_dot * z
        return v_t


class UniformTimeSampler:
    """Sample timesteps uniformly from a range."""
    
    def __init__(self, t_min: float = 0.0, t_max: float = 1.0):
        self.t_min = t_min
        self.t_max = t_max
    
    def __call__(self, batch_size: int) -> torch.Tensor:
        """Sample batch_size timesteps."""
        t = torch.rand(batch_size) * (self.t_max - self.t_min) + self.t_min
        return t


class LogitNormalTimeSampler:
    """Sample timesteps from logit-normal distribution."""
    
    def __init__(
        self,
        mean: float = 0.4,
        std: float = 1.0,
        clamp_min: float = 1e-5,
        clamp_max: float = 1.0 - 1e-5,
    ):
        self.mean = mean
        self.std = std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    
    def __call__(self, batch_size: int) -> torch.Tensor:
        """Sample batch_size timesteps."""
        # Sample from normal distribution
        normal_samples = torch.randn(batch_size) * self.std + self.mean
        # Apply sigmoid to get logit-normal
        t = torch.sigmoid(normal_samples)
        # Clamp to avoid numerical issues
        t = torch.clamp(t, self.clamp_min, self.clamp_max)
        return t


def create_time_sampler(sampler_type: str = "uniform", **kwargs):
    """Factory function for time samplers."""
    if sampler_type == "uniform":
        return UniformTimeSampler(**kwargs)
    elif sampler_type == "logit_normal":
        return LogitNormalTimeSampler(**kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

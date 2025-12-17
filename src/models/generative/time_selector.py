"""Contextual timestep selection with discounted LinUCB."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ImageFeatureExtractor:
    """Lightweight image statistics for contextual bandit."""

    def __init__(self, high_freq_ratio: float = 0.25, eps: float = 1e-6):
        self.high_freq_ratio = high_freq_ratio
        self.eps = eps
        self._mask_cache: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

    @property
    def dim(self) -> int:
        return 8  # mean (3) + std (3) + laplacian (1) + high-freq ratio (1)

    def _get_high_freq_mask(
        self, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Precompute a radial high-frequency mask for FFT energy."""
        cache_key = (height, width, device, dtype)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        y = torch.linspace(-0.5, 0.5, steps=height, device=device, dtype=dtype)
        x = torch.linspace(-0.5, 0.5, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx**2 + yy**2)
        mask = (radius >= self.high_freq_ratio).float()  # 1 for high-freq area
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self._mask_cache[cache_key] = mask
        return mask

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) in [0, 1] on any device
        Returns:
            features: (B, 8)
        """
        images = images.float()
        B, _, H, W = images.shape

        means = images.mean(dim=(2, 3))  # (B, 3)
        stds = images.std(dim=(2, 3), unbiased=False)  # (B, 3)

        # Laplacian energy on grayscale
        gray = images.mean(dim=1, keepdim=True)
        lap_kernel = images.new_tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]])
        lap_resp = F.conv2d(gray, lap_kernel, padding=1)
        lap_energy = (lap_resp ** 2).mean(dim=(1, 2, 3))  # (B,)

        # High-frequency energy ratio via FFT
        fft = torch.fft.fft2(gray)
        power = fft.abs() ** 2  # (B, 1, H, W)
        mask = self._get_high_freq_mask(H, W, images.device, images.dtype)
        high_energy = (power * mask).sum(dim=(2, 3))  # (B, 1)
        total_energy = power.sum(dim=(2, 3)) + self.eps  # (B, 1)
        high_ratio = (high_energy / total_energy).squeeze(1)  # (B,)

        features = torch.cat(
            [means, stds, lap_energy.unsqueeze(-1), high_ratio.unsqueeze(-1)], dim=1
        )  # (B, 8)
        return features


class _OutputFeatureExtractor:
    """Classifier output statistics."""

    def __init__(self, include_energy: bool = True, eps: float = 1e-6):
        self.include_energy = include_energy
        self.eps = eps

    @property
    def dim(self) -> int:
        return 5 + int(self.include_energy)  # entropy, p_max, margin, ||z||, (energy)

    @torch.no_grad()
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C)
        Returns:
            features: (B, d_out)
        """
        logits = logits.float()
        probs = torch.softmax(logits, dim=-1)

        entropy = -(probs * (probs.clamp_min(self.eps).log())).sum(dim=-1)  # (B,)
        p_max, _ = probs.max(dim=-1)  # (B,)
        top2 = torch.topk(probs, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]  # (B,)
        logit_norm = logits.norm(p=2, dim=-1)  # (B,)

        feats = [entropy, p_max, margin, logit_norm]
        if self.include_energy:
            energy = -torch.logsumexp(logits, dim=-1)  # (B,)
            feats.append(energy)

        return torch.stack(feats, dim=1)  # (B, d_out)


class DiscountedContextualLinUCB(nn.Module):
    """Disjoint discounted LinUCB with per-arm statistics."""

    def __init__(
        self,
        num_arms: int,
        context_dim: int,
        alpha: float = 2.0,
        lambda_reg: float = 1.0,
        gamma: float = 0.99,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        A0 = torch.eye(context_dim) * lambda_reg
        self.register_buffer("A", A0.unsqueeze(0).repeat(num_arms, 1, 1))  # (K, d, d)
        self.register_buffer("b", torch.zeros(num_arms, context_dim))  # (K, d)
        self.register_buffer("lambda_reg", torch.tensor(lambda_reg))

    @torch.no_grad()
    def select(self, contexts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            contexts: (B, d)
        Returns:
            arm_indices: (B,)
            ucb_scores: (B, K)
        """
        if contexts.dim() != 2 or contexts.size(1) != self.context_dim:
            raise ValueError(f"Context shape {contexts.shape} is invalid for dim={self.context_dim}")

        device = self.A.device
        contexts = contexts.to(device)

        jitter = self.eps * torch.eye(self.context_dim, device=device, dtype=self.A.dtype)
        A_inv = torch.linalg.inv(self.A + jitter)  # (K, d, d)
        theta = torch.bmm(A_inv, self.b.unsqueeze(-1)).squeeze(-1)  # (K, d)

        mean_term = torch.einsum("bd,kd->bk", contexts, theta)  # (B, K)
        proj = torch.einsum("bd,kdc->bkc", contexts, A_inv)  # (B, K, d)
        var_term = torch.einsum("bkd,bd->bk", proj, contexts)  # (B, K)

        ucb = mean_term + self.alpha * torch.sqrt(torch.clamp(var_term, min=0.0) + self.eps)
        arm_indices = torch.argmax(ucb, dim=1)  # (B,)
        return arm_indices, ucb

    @torch.no_grad()
    def update(self, contexts: torch.Tensor, rewards: torch.Tensor, arm_indices: torch.Tensor):
        """
        Discounted update for chosen arms.

        Args:
            contexts: (N, d) normalized contexts
            rewards: (N,) in [0, 1]
            arm_indices: (N,) chosen arms
        """
        if contexts.numel() == 0:
            return

        device = self.A.device
        contexts = contexts.to(device)
        rewards = rewards.to(device).view(-1)
        arm_indices = arm_indices.to(device).view(-1).long()

        unique_arms = torch.unique(arm_indices)
        for arm in unique_arms:
            mask = arm_indices == arm
            if not mask.any():
                continue
            ctx_arm = contexts[mask]  # (n_a, d)
            rew_arm = rewards[mask].unsqueeze(1)  # (n_a, 1)

            # Discount old stats then add new observations
            self.A[arm] = self.gamma * self.A[arm] + ctx_arm.t().mm(ctx_arm)
            self.b[arm] = self.gamma * self.b[arm] + (rew_arm * ctx_arm).sum(dim=0)


class ContextualTimeStepSelector(nn.Module):
    """Wrapper that extracts context and runs discounted LinUCB over candidate timesteps."""

    def __init__(
        self,
        time_candidates: List[float],
        alpha: float = 2.0,
        lambda_reg: float = 1.0,
        gamma: float = 0.995,
        update_interval: int = 1,
        high_freq_ratio: float = 0.25,
        include_energy: bool = True,
        feature_eps: float = 1e-6,
    ):
        super().__init__()
        if len(time_candidates) == 0:
            raise ValueError("time_candidates must contain at least one timestep")

        time_tensor = torch.tensor(time_candidates, dtype=torch.float32).clamp(0.0, 1.0)
        self.register_buffer("time_candidates", time_tensor)

        self.image_features = _ImageFeatureExtractor(high_freq_ratio=high_freq_ratio, eps=feature_eps)
        self.output_features = _OutputFeatureExtractor(include_energy=include_energy, eps=feature_eps)
        context_dim = self.image_features.dim + self.output_features.dim

        self.bandit = DiscountedContextualLinUCB(
            num_arms=len(time_candidates),
            context_dim=context_dim,
            alpha=alpha,
            lambda_reg=lambda_reg,
            gamma=gamma,
            eps=feature_eps,
        )
        self.update_interval = max(1, int(update_interval))
        self.feature_eps = feature_eps

        self._pending: Dict[str, List[torch.Tensor]] = {
            "contexts": [],
            "arms": [],
            "rewards": [],
        }

    @torch.no_grad()
    def _build_context(self, images: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_feat = self.image_features(images)  # (B, d_img)
        out_feat = self.output_features(logits)  # (B, d_out)
        context = torch.cat([img_feat, out_feat], dim=1)  # (B, d)

        mean = context.mean(dim=0, keepdim=True)
        std = context.std(dim=0, keepdim=True, unbiased=False)
        norm_context = (context - mean) / (std + self.feature_eps)

        return norm_context, {
            "phi_img": img_feat,
            "phi_out": out_feat,
            "context_mean": mean,
            "context_std": std,
        }

    @torch.no_grad()
    def select_timesteps(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            images: (B, 3, H, W) in [0, 1]
            logits: (B, C)
        Returns:
            timesteps: (B,)
            arm_indices: (B,)
            contexts: (B, d)
            extra_info: dict with diagnostics (ucb, features)
        """
        contexts, ctx_info = self._build_context(images, logits)
        arm_indices, ucb_scores = self.bandit.select(contexts)
        timesteps = self.time_candidates[arm_indices]

        ctx_info.update(
            {
                "ucb": ucb_scores,
                "selected_arms": arm_indices,
                "selected_timesteps": timesteps,
            }
        )
        return timesteps, arm_indices, contexts, ctx_info

    @torch.no_grad()
    def observe(
        self,
        contexts: torch.Tensor,
        arm_indices: torch.Tensor,
        rewards: torch.Tensor,
        force_update: bool = True,
    ) -> None:
        """Buffer observations then update the bandit."""
        self._pending["contexts"].append(contexts.detach())
        self._pending["arms"].append(arm_indices.detach())
        self._pending["rewards"].append(rewards.detach())

        should_update = force_update or len(self._pending["contexts"]) >= self.update_interval
        if not should_update:
            return

        ctx = torch.cat(self._pending["contexts"], dim=0)
        arms = torch.cat(self._pending["arms"], dim=0)
        rew = torch.cat(self._pending["rewards"], dim=0)
        self._pending = {"contexts": [], "arms": [], "rewards": []}

        self.bandit.update(ctx, rew, arms)


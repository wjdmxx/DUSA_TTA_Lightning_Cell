"""
Contextual timestep selection with discounted LinUCB.

- Disjoint LinUCB over candidate timesteps (arms)
- Exponential discounting for non-stationary test-time adaptation
- Stable context normalization via running mean/var (no per-batch z-score)

Public interface kept compatible:
  - ContextualTimeStepSelector.select_timesteps(images, logits)
  - ContextualTimeStepSelector.observe(contexts, arm_indices, rewards, force_update=...)
  - ContextualTimeStepSelector.reset()

Reward:
  - Assumed already normalized into [0, 1] (your normalized Kendall tau).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ImageFeatureExtractor:
    """Lightweight image statistics for contextual bandit.

    Input: images (B, 3, H, W) in [0, 1]
    Output: features (B, 8)
      - mean per channel (3)
      - std per channel (3)
      - Laplacian energy on grayscale (1)
      - high-frequency energy ratio via FFT (1)
    """

    def __init__(self, high_freq_ratio: float = 0.25, eps: float = 1e-6, fft_norm: str = "backward"):
        if not (0.0 < float(high_freq_ratio) < 1.0):
            raise ValueError(f"high_freq_ratio should be in (0,1), got {high_freq_ratio}")
        self.high_freq_ratio = float(high_freq_ratio)
        self.eps = float(eps)
        self.fft_norm = fft_norm
        self._mask_cache: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

    @property
    def dim(self) -> int:
        return 8

    def _get_high_freq_mask(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Radial high-frequency mask aligned with torch.fft.fft2 ordering.

        Use torch.fft.fftfreq to match FFT output convention (DC at index 0, then positive freqs, then negative freqs).
        """
        cache_key = (height, width, device, dtype)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(width, d=1.0, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")

        radius = torch.sqrt(xx * xx + yy * yy)
        max_r = (0.5**2 + 0.5**2) ** 0.5  # continuous limit
        thr = self.high_freq_ratio * max_r

        mask = (radius >= thr).to(dtype=dtype)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self._mask_cache[cache_key] = mask
        return mask

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        if images.dim() != 4 or images.size(1) != 3:
            raise ValueError(f"images must have shape (B, 3, H, W), got {tuple(images.shape)}")

        means = images.mean(dim=(2, 3))  # (B, 3)
        stds = images.std(dim=(2, 3), unbiased=False)  # (B, 3)

        gray = images.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        lap_kernel = images.new_tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]])
        lap_resp = F.conv2d(gray, lap_kernel, padding=1)
        lap_energy = (lap_resp.square()).mean(dim=(1, 2, 3))  # (B,)

        fft = torch.fft.fft2(gray, norm=self.fft_norm)
        power = fft.abs().square()  # (B, 1, H, W)
        mask = self._get_high_freq_mask(images.size(2), images.size(3), images.device, images.dtype)

        high_energy = (power * mask).sum(dim=(2, 3))  # (B, 1)
        total_energy = power.sum(dim=(2, 3)).clamp_min(self.eps)  # (B, 1)
        high_ratio = (high_energy / total_energy).squeeze(1)  # (B,)

        features = torch.cat(
            [means, stds, lap_energy.unsqueeze(-1), high_ratio.unsqueeze(-1)],
            dim=1,
        )
        return features


class _OutputFeatureExtractor:
    """Classifier output statistics from logits."""

    def __init__(self, include_energy: bool = True, eps: float = 1e-6):
        self.include_energy = bool(include_energy)
        self.eps = float(eps)

    @property
    def dim(self) -> int:
        return 4 + int(self.include_energy)

    @torch.no_grad()
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must have shape (B, C), got {tuple(logits.shape)}")

        logits = logits.float()
        probs = torch.softmax(logits, dim=-1)

        entropy = -(probs * probs.clamp_min(self.eps).log()).sum(dim=-1)  # (B,)
        p_max = probs.max(dim=-1).values  # (B,)
        top2 = torch.topk(probs, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]  # (B,)
        logit_norm = logits.norm(p=2, dim=-1)  # (B,)

        feats = [entropy, p_max, margin, logit_norm]
        if self.include_energy:
            energy = -torch.logsumexp(logits, dim=-1)  # (B,)
            feats.append(energy)

        return torch.stack(feats, dim=1)  # (B, d_out)


class DiscountedContextualLinUCB(nn.Module):
    """Disjoint discounted LinUCB with per-arm statistics.

    Uses the discounted ridge regression recursion that preserves base regularization:
      A <- gamma A + X^T X + (1-gamma)*lambda*I
      b <- gamma b + sum r x
    """

    def __init__(
        self,
        num_arms: int,
        context_dim: int,
        alpha: float = 2.0,
        lambda_reg: float = 1.0,
        gamma: float = 0.99,
        eps: float = 1e-6,
        tie_break_noise: float = 1e-8,
    ):
        super().__init__()
        if num_arms <= 0:
            raise ValueError(f"num_arms must be positive, got {num_arms}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if not (0.0 < float(gamma) <= 1.0):
            raise ValueError(f"gamma must be in (0,1], got {gamma}")

        self.num_arms = int(num_arms)
        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.tie_break_noise = float(tie_break_noise)

        A0 = torch.eye(context_dim) * float(lambda_reg)
        self.register_buffer("A", A0.unsqueeze(0).repeat(self.num_arms, 1, 1))  # (K, d, d)
        self.register_buffer("b", torch.zeros(self.num_arms, context_dim))  # (K, d)
        self.register_buffer("lambda_reg", torch.tensor(float(lambda_reg)))
        self.register_buffer("_eye", torch.eye(context_dim))

    @torch.no_grad()
    def reset_stats(self) -> None:
        eye = self._eye.to(device=self.A.device, dtype=self.A.dtype)
        A0 = eye * float(self.lambda_reg)
        self.A.copy_(A0.unsqueeze(0).repeat(self.num_arms, 1, 1))
        self.b.zero_()

    @torch.no_grad()
    def _discount_one_step(self) -> None:
        """Discount all arms one step, preserving lambda*I."""
        if self.gamma >= 1.0:
            return

        g = self.gamma
        eye = self._eye.to(device=self.A.device, dtype=self.A.dtype)
        lam = float(self.lambda_reg)

        self.A.mul_(g).add_(eye.unsqueeze(0) * ((1.0 - g) * lam))
        self.b.mul_(g)

    @torch.no_grad()
    def select(self, contexts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if contexts.dim() != 2 or contexts.size(1) != self.context_dim:
            raise ValueError(f"Context shape {tuple(contexts.shape)} is invalid for dim={self.context_dim}")

        device = self.A.device
        dtype = self.A.dtype
        contexts = contexts.to(device=device, dtype=dtype)

        eye = self._eye.to(device=device, dtype=dtype)
        A_reg = self.A + eye.unsqueeze(0) * self.eps

        L = torch.linalg.cholesky(A_reg)  # (K, d, d)
        A_inv = torch.cholesky_inverse(L)  # (K, d, d)
        theta = torch.bmm(A_inv, self.b.unsqueeze(-1)).squeeze(-1)  # (K, d)

        mean_term = torch.einsum("bd,kd->bk", contexts, theta)  # (B, K)
        proj = torch.einsum("bd,kdc->bkc", contexts, A_inv)  # (B, K, d)
        var_term = torch.einsum("bkd,bd->bk", proj, contexts)  # (B, K)

        ucb = mean_term + self.alpha * torch.sqrt(torch.clamp(var_term, min=0.0) + self.eps)
        if self.tie_break_noise > 0.0:
            ucb = ucb + self.tie_break_noise * torch.randn_like(ucb)

        arm_indices = torch.argmax(ucb, dim=1)
        return arm_indices, ucb

    @torch.no_grad()
    def update(self, contexts: torch.Tensor, rewards: torch.Tensor, arm_indices: torch.Tensor) -> None:
        """One-step discounted update for a batch of observations in the same step."""
        if contexts.numel() == 0:
            return

        device = self.A.device
        dtype = self.A.dtype
        contexts = contexts.to(device=device, dtype=dtype)
        rewards = rewards.to(device=device, dtype=dtype).view(-1)
        arm_indices = arm_indices.to(device=device).view(-1).long()

        if contexts.dim() != 2 or contexts.size(1) != self.context_dim:
            raise ValueError(f"contexts must have shape (N, {self.context_dim}), got {tuple(contexts.shape)}")
        if rewards.numel() != contexts.size(0) or arm_indices.numel() != contexts.size(0):
            raise ValueError("contexts, rewards, arm_indices must have matching first dimension")

        finite = torch.isfinite(contexts).all(dim=1) & torch.isfinite(rewards)
        if not finite.all():
            contexts = contexts[finite]
            rewards = rewards[finite]
            arm_indices = arm_indices[finite]
            if contexts.numel() == 0:
                return

        # reward already normalized Kendall tau in [0,1]
        rewards = rewards.clamp(0.0, 1.0)

        # discount for one step
        self._discount_one_step()

        unique_arms = torch.unique(arm_indices)
        for arm in unique_arms.tolist():
            mask = arm_indices == arm
            if not mask.any():
                continue
            ctx_arm = contexts[mask]  # (n_a, d)
            rew_arm = rewards[mask].unsqueeze(1)  # (n_a, 1)

            self.A[arm].add_(ctx_arm.t().mm(ctx_arm))
            self.b[arm].add_((rew_arm * ctx_arm).sum(dim=0))


class ContextualTimeStepSelector(nn.Module):
    """Extract context and run discounted LinUCB over candidate timesteps."""

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
        # optional knobs (safe defaults)
        normalize_context: bool = True,
        context_stats_momentum: Optional[float] = None,
        tie_break_noise: float = 1e-8,
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
            tie_break_noise=tie_break_noise,
        )

        self.update_interval = max(1, int(update_interval))
        self.feature_eps = float(feature_eps)

        # Running normalization stats (fixes B=1 degeneracy & avoids per-batch coordinate drift)
        self.normalize_context = bool(normalize_context)
        self.context_stats_momentum = context_stats_momentum
        self.register_buffer("_ctx_count", torch.tensor(0.0))
        self.register_buffer("_ctx_mean", torch.zeros(context_dim))
        self.register_buffer("_ctx_var", torch.ones(context_dim))

        # Pending buffers: one entry per observe() call (supports cross-batch accumulation)
        self._pending: Dict[str, List[torch.Tensor]] = {"contexts": [], "arms": [], "rewards": []}

    @torch.no_grad()
    def _get_running_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self._ctx_mean.unsqueeze(0)
        std = torch.sqrt(self._ctx_var.clamp_min(0.0) + self.feature_eps).unsqueeze(0)
        return mean, std

    @torch.no_grad()
    def _update_running_stats(self, x: torch.Tensor) -> None:
        """Update running mean/var with batch x using either exact merge or EMA."""
        if x.numel() == 0:
            return

        x = x.detach()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        if self._ctx_count.item() <= 0.0:
            self._ctx_mean.copy_(batch_mean)
            self._ctx_var.copy_(batch_var.clamp_min(self.feature_eps))
            self._ctx_count.fill_(float(x.size(0)))
            return

        m = self.context_stats_momentum
        if m is not None:
            m = float(m)
            if not (0.0 < m < 1.0):
                raise ValueError(f"context_stats_momentum must be in (0,1), got {m}")
            self._ctx_mean.mul_(1.0 - m).add_(batch_mean * m)
            self._ctx_var.mul_(1.0 - m).add_(batch_var * m).clamp_min_(self.feature_eps)
            self._ctx_count.add_(float(x.size(0)))
            return

        # Exact running merge (population variance)
        n1 = self._ctx_count.item()
        n2 = float(x.size(0))
        mean1 = self._ctx_mean
        var1 = self._ctx_var.clamp_min(self.feature_eps)
        mean2 = batch_mean
        var2 = batch_var.clamp_min(self.feature_eps)

        n = n1 + n2
        delta = mean2 - mean1
        mean = mean1 + delta * (n2 / n)

        M2_1 = var1 * n1
        M2_2 = var2 * n2
        M2 = M2_1 + M2_2 + delta.square() * (n1 * n2 / n)
        var = (M2 / n).clamp_min(self.feature_eps)

        self._ctx_mean.copy_(mean)
        self._ctx_var.copy_(var)
        self._ctx_count.fill_(n)

    @torch.no_grad()
    def _build_context(self, images: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_feat = self.image_features(images)
        out_feat = self.output_features(logits)
        raw_context = torch.cat([img_feat, out_feat], dim=1)

        if not self.normalize_context:
            return raw_context, {
                "phi_img": img_feat,
                "phi_out": out_feat,
                "context_mean": torch.zeros(1, raw_context.size(1), device=raw_context.device, dtype=raw_context.dtype),
                "context_std": torch.ones(1, raw_context.size(1), device=raw_context.device, dtype=raw_context.dtype),
                "context_count": self._ctx_count.detach().clone(),
            }

        mean, std = self._get_running_stats()
        mean = mean.to(device=raw_context.device, dtype=raw_context.dtype)
        std = std.to(device=raw_context.device, dtype=raw_context.dtype)

        norm_context = (raw_context - mean) / std

        # update running stats after producing normalized context
        self._update_running_stats(raw_context)

        return norm_context, {
            "phi_img": img_feat,
            "phi_out": out_feat,
            "context_mean": mean,
            "context_std": std,
            "context_count": self._ctx_count.detach().clone(),
        }

    @torch.no_grad()
    def select_timesteps(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        contexts, ctx_info = self._build_context(images, logits)
        arm_indices, ucb_scores = self.bandit.select(contexts)
        timesteps = self.time_candidates[arm_indices]

        ctx_info.update({"ucb": ucb_scores, "selected_arms": arm_indices, "selected_timesteps": timesteps})
        return timesteps, arm_indices, contexts, ctx_info

    @torch.no_grad()
    def observe(
        self,
        contexts: torch.Tensor,
        arm_indices: torch.Tensor,
        rewards: torch.Tensor,
        force_update: bool = False,
    ) -> None:
        """Buffer observations, then update bandit.

        update_interval is counted in number of observe() calls (i.e., batches / steps).

        - update_interval=1 (your current setting): update every call (same as your old behavior).
        - update_interval>1: automatically accumulates across calls, and updates only when enough calls arrive.
          This is the cross-batch update you asked for.
        - force_update=True flushes immediately (useful at epoch end / test end).
        """
        self._pending["contexts"].append(contexts.detach())
        self._pending["arms"].append(arm_indices.detach())
        self._pending["rewards"].append(rewards.detach())

        should_update = force_update or (len(self._pending["contexts"]) >= self.update_interval)
        if not should_update:
            return

        # Process pending calls as sequential "steps"
        for ctx, arms, rew in zip(self._pending["contexts"], self._pending["arms"], self._pending["rewards"]):
            self.bandit.update(ctx, rew, arms)

        self._pending = {"contexts": [], "arms": [], "rewards": []}

    @torch.no_grad()
    def reset(self) -> None:
        """Reset bandit stats, running normalization, and pending buffers."""
        self.bandit.reset_stats()
        self._pending = {"contexts": [], "arms": [], "rewards": []}
        self._ctx_count.zero_()
        self._ctx_mean.zero_()
        self._ctx_var.fill_(1.0)

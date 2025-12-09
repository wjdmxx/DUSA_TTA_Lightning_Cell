"""
REPA SiT model implementation.
Adapted from the original DUSA codebase with simplifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, List, Dict
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange

from .scheduler import FlowScheduler, create_time_sampler
from .vae import VAEEncoder


def modulate(x, shift, scale):
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.positional_embedding(t, self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class SiTBlock(nn.Module):
    """A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
        fused_attn: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm
        )
        if hasattr(self.attn, "fused_attn"):
            self.attn.fused_attn = fused_attn

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """The final layer of SiT."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    """Scalable Interpolant Transformer."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        decoder_hidden_size: int = 1152,
        encoder_depth: int = 8,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        use_cfg: bool = False,
        qk_norm: bool = False,
        fused_attn: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                SiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    fused_attn=fused_attn,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(
            decoder_hidden_size, patch_size, self.out_channels
        )
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize transformer layers."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """x: (N, T, patch_size**2 * C) -> imgs: (N, C, H, W)"""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, y):
        """
        Args:
            x: (N, C, H, W) spatial inputs (latent representations)
            t: (N,) diffusion timesteps
            y: (N,) class labels

        Returns:
            output: (N, C, H, W) predicted velocity
            middle_features: (N, T, D) intermediate features from encoder
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        N, T, D = x.shape

        # Timestep and class embedding
        t_embed = self.t_embedder(t)  # (N, D)
        y_embed = self.y_embedder(y, self.training)  # (N, D)
        c = t_embed + y_embed  # (N, D)

        # Transformer blocks
        middle_features = None
        for i, block in enumerate(self.blocks):
            x = block(x, c)  # (N, T, D)
            if (i + 1) == self.encoder_depth:
                middle_features = x  # Save intermediate features

        x = self.final_layer(x, c)  # (N, T, patch_size**2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x, middle_features


# ================= Positional Embedding Functions =================


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """Generate 2D sin-cos positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# ================= Model Configs =================


def SiT_XL_2(**kwargs):
    return SiT(
        depth=28,
        hidden_size=1152,
        decoder_hidden_size=1152,
        patch_size=2,
        num_heads=16,
        **kwargs,
    )


def SiT_B_2(**kwargs):
    return SiT(
        depth=12,
        hidden_size=768,
        decoder_hidden_size=768,
        patch_size=2,
        num_heads=12,
        **kwargs,
    )


# ================= REPA SiT Module =================


class REPASiT(nn.Module):
    """
    REPA (auxiliary) module wrapping SiT + VAE + preprocessing.
    Implements the DUSA TTA loss with top-k + random sampling.
    """

    def __init__(
        self,
        # SiT config
        sit_model_name: str = "SiT-XL/2",
        sit_checkpoint: Optional[str] = None,
        num_classes: int = 1000,
        # VAE config
        vae_pretrained: str = "stabilityai/sd-vae-ft-ema",
        vae_scaling_factor: float = 0.18215,
        # Loss config
        topk: int = 4,
        rand_budget: int = 2,
        temperature: float = 1.0,
        sample_reverse_logits: bool = False,
        # Scheduler config
        scheduler_type: str = "linear",
        time_sampler_type: str = "uniform",
        time_sampler_kwargs: Optional[Dict] = None,
        # Device
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.topk = topk
        self.rand_budget = rand_budget
        self.temperature = temperature
        self.sample_reverse_logits = sample_reverse_logits
        self.num_classes = num_classes

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Build SiT model
        if sit_model_name == "SiT-XL/2":
            self.flow_model = SiT_XL_2(
                input_size=32,
                num_classes=num_classes,
                use_cfg=True,
                encoder_depth=8,
                fused_attn=True,
                qk_norm=False,
            )
        elif sit_model_name == "SiT-B/2":
            self.flow_model = SiT_B_2(
                input_size=32,
                num_classes=num_classes,
                use_cfg=True,
                encoder_depth=6,
                fused_attn=True,
                qk_norm=False,
            )
        else:
            raise ValueError(f"Unknown SiT model: {sit_model_name}")

        # Load checkpoint
        if sit_checkpoint is not None:
            state_dict = torch.load(sit_checkpoint, map_location=self.device)
            self.flow_model.load_state_dict(state_dict, strict=False)
            print(f"Loaded SiT checkpoint from {sit_checkpoint}")

        self.flow_model.to(self.device)
        self.flow_model.eval()

        # Build VAE
        self.vae = VAEEncoder(
            pretrained_model=vae_pretrained,
            scaling_factor=vae_scaling_factor,
            device=self.device,
        )

        # Build scheduler and time sampler
        self.scheduler = FlowScheduler(scheduler_type=scheduler_type)
        time_sampler_kwargs = time_sampler_kwargs or {}
        self.time_sampler = create_time_sampler(
            time_sampler_type, **time_sampler_kwargs
        )

    def set_train_mode(self, update_flow: bool = True):
        """Configure which parameters to update."""
        if update_flow:
            self.flow_model.requires_grad_(True)
            # Freeze class embeddings
            if hasattr(self.flow_model, "y_embedder"):
                self.flow_model.y_embedder.requires_grad_(False)
        else:
            self.flow_model.requires_grad_(False)

        # VAE always frozen
        self.vae.requires_grad_(False)

    def preprocess_images(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Preprocess images for diffusion model.

        Args:
            images: List of (C, H, W) tensors in BGR, range [0, 255]

        Returns:
            Preprocessed tensor (B, 3, 256, 256) in RGB, range [-1, 1]
        """
        # BGR to RGB
        images_rgb = [img[[2, 1, 0], :, :] for img in images]

        # Stack and resize
        images_batch = torch.stack(images_rgb, dim=0).float()  # (B, 3, H, W)
        if images_batch.shape[-2:] != (256, 256):
            images_batch = F.interpolate(
                images_batch, size=(256, 256), mode="bilinear", align_corners=True
            )

        # Normalize to [-1, 1]
        images_batch = images_batch / 255.0
        images_batch = images_batch * 2.0 - 1.0

        return images_batch.to(self.device)

    def forward(
        self,
        images: List[torch.Tensor],
        normed_logits: torch.Tensor,
        ori_logits: torch.Tensor,
        batch_infos: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for REPA loss computation.

        Args:
            images: List of input images (C, H, W) in BGR [0, 255]
            normed_logits: L2-normalized logits (B, num_classes)
            ori_logits: Original logits (B, num_classes)
            batch_infos: Optional dict with step, all_steps, task_name

        Returns:
            loss: Scalar loss
            aux_losses: Dict of auxiliary metrics (loss_top_i, loss_rand_i, auc, gap_norm)
        """
        batch_infos = batch_infos or {}

        # Preprocess images
        x_img = self.preprocess_images(images)  # (B, 3, 256, 256)
        B = x_img.size(0)

        # VAE encode
        x0 = self.vae.encode(x_img)  # (B, 4, 32, 32)

        # Top-k + random sampling
        topk_logits, topk_idx = torch.topk(
            normed_logits, self.topk, dim=-1
        )  # (B, topk)

        if self.rand_budget > 0:
            # Get non-topk indices
            _, non_topk_idx = torch.topk(
                normed_logits, normed_logits.shape[1] - self.topk, dim=-1, largest=False
            )

            # Sample from non-topk using temperature-scaled original logits
            if self.sample_reverse_logits:
                # Sample from low-confidence classes
                rand_logits = (
                    -torch.gather(ori_logits, 1, non_topk_idx) / self.temperature
                )
            else:
                rand_logits = (
                    torch.gather(ori_logits, 1, non_topk_idx) / self.temperature
                )

            rand_idx = torch.multinomial(
                F.softmax(rand_logits, dim=1), self.rand_budget, replacement=False
            )
            rand_idx = torch.gather(non_topk_idx, 1, rand_idx)  # (B, rand_budget)

            # Combine topk + random
            forward_idx = torch.cat([topk_idx, rand_idx], dim=-1)  # (B, K)

            # Weighting coefficients
            if self.sample_reverse_logits:
                prob_as_coeff = F.softmax(
                    topk_logits, dim=-1
                )  # Only use topk for weighting
            else:
                prob_as_coeff = F.softmax(
                    torch.gather(normed_logits, 1, forward_idx), dim=-1
                )

            K = self.topk + self.rand_budget
        else:
            forward_idx = topk_idx
            prob_as_coeff = F.softmax(topk_logits, dim=-1)
            K = self.topk

        # Sample noise and timestep
        z = torch.randn_like(x0, device=x0.device)
        t = self.time_sampler(B).to(self.device)

        # Compute x_t
        x_t = self.scheduler.get_xt(x0, z, t)

        # Expand for all selected classes
        t_rep = t.repeat_interleave(K, dim=0)  # (B*K,)
        x_t_rep = x_t.repeat_interleave(K, dim=0)  # (B*K, 4, 32, 32)
        y = forward_idx.reshape(-1).long().to(x0.device)  # (B*K,)

        # Flow model forward
        model_out, _ = self.flow_model(x=x_t_rep, t=t_rep, y=y)
        model_out = rearrange(
            model_out, "(b k) c h w -> b k c h w", b=B
        )  # (B, K, 4, 32, 32)

        # Separate topk and random outputs
        topk_out = model_out[:, : self.topk, ...]  # (B, topk, 4, 32, 32)
        rand_out = model_out[:, self.topk :, ...]  # (B, rand_budget, 4, 32, 32)

        # Weighted aggregation
        if self.sample_reverse_logits:
            weighted_out = torch.einsum("bk,bkchw->bchw", prob_as_coeff, topk_out)
        else:
            weighted_out = torch.einsum("bk,bkchw->bchw", prob_as_coeff, model_out)

        # Target velocity
        target = self.scheduler.get_vt(x0, z, t)

        # REPA loss: norm_l2_loss + cosine_loss
        loss = self._compute_repa_loss(weighted_out, target)

        # Compute auxiliary metrics
        with torch.no_grad():
            aux_losses = self._compute_aux_metrics(model_out, target, prob_as_coeff, K)

        return loss, aux_losses

    def _compute_repa_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute REPA loss: norm_l2_loss + cosine_loss."""
        # Norm L2 loss with outlier resistance
        e = torch.mean((pred - target) ** 2, dim=(1, 2, 3), keepdim=False)
        p, c = 0.5, 1e-3
        norm_l2 = (e / (e + c).pow(p).detach()).mean()

        # Cosine loss
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-8)
        cos_loss = (1.0 - cos_sim).mean()

        return norm_l2

    def _compute_aux_metrics(
        self,
        model_out: torch.Tensor,
        target: torch.Tensor,
        prob_as_coeff: torch.Tensor,
        K: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary metrics (per-class losses, AUC, gap norm)."""
        # Per-class MSE
        per_pixel_mse = F.mse_loss(
            model_out,
            target.unsqueeze(1),  # (B, 1, C, H, W)
            reduction="none",
        )
        per_class_loss = per_pixel_mse.flatten(2).mean(dim=-1)  # (B, K)

        loss_pos = per_class_loss[:, : self.topk]  # (B, topk)
        loss_neg = per_class_loss[:, self.topk :]  # (B, rand_budget)

        aux_losses = {}

        # Per-class losses
        for i in range(self.topk):
            aux_losses[f"loss_top_{i}"] = loss_pos[:, i].mean()
        for i in range(K - self.topk):
            aux_losses[f"loss_rand_{i}"] = loss_neg[:, i].mean()

        # AUC (ranking metric)
        if loss_neg.numel() > 0:
            pos_score = -loss_pos.unsqueeze(-1)  # (B, topk, 1)
            neg_score = -loss_neg.unsqueeze(1)  # (B, 1, rand_budget)
            pairwise = (pos_score > neg_score).float() + 0.5 * (
                pos_score == neg_score
            ).float()
            auc = pairwise.mean()
        else:
            auc = torch.tensor(1.0, device=loss_pos.device)
        aux_losses["auc"] = auc

        # Gap norm (normalized separation)
        mean_pos = loss_pos.mean(dim=1)  # (B,)
        mean_neg = loss_neg.mean(dim=1)  # (B,)
        std_all = per_class_loss.std(dim=1, unbiased=False)  # (B,)
        gap_norm = ((mean_neg - mean_pos) / (std_all + 1e-12)).mean()
        aux_losses["gap_norm"] = gap_norm

        return aux_losses


def create_repa_sit(
    sit_model_name: str = "SiT-XL/2",
    sit_checkpoint: Optional[str] = None,
    num_classes: int = 1000,
    **kwargs,
) -> REPASiT:
    """Factory function to create REPA SiT."""
    return REPASiT(
        sit_model_name=sit_model_name,
        sit_checkpoint=sit_checkpoint,
        num_classes=num_classes,
        **kwargs,
    )

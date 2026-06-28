"""
Step Chart Neural Network Model (v8 — arrow heads + structure awareness).

v8 adds on top of v7:
    - Four per-arrow logit heads (L/D/U/R) trained with focal BCE only at
      onset frames, used by inference to bias the FootStateArrowAssigner.
    - Section-boundary conditioning: a Fourier-encoded `section_position`
      scalar (0=start of section, 0.5=middle, 1=end) plus an extra
      `near_boundary` flag join the FiLM conditioner so the model can vary
      its predictions across song sections.

Core v7 heads retained:
    feats[T,88] -> AudioEncoder (CNN) -> Dilated TCN -> early FiLM
        (difficulty + density + start_seconds + remaining_seconds
         + section_position + near_boundary)
        -> RoPE Transformer -> LayerNorm
        -> onset head      [T, 1] sigmoid: any-onset prob (Gaussian-smoothed BCE)
        -> sustain head    [T, 1] sigmoid: in-any-hold prob (dense BCE)
        -> intensity head  [T, 1] linear:  jump-vs-tap intensity (MSE regression)
        -> arrow_heads     [T, 4] sigmoid: per-arrow onset prob (focal BCE, onset-masked)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# v8: arrow prediction heads + structure-aware FiLM conditioning.
ARCH_VERSION = 8


class AudioEncoder(nn.Module):
    """CNN encoder for the v8 88-channel audio feature tensor.

    Input is `feats[B, T, n_in_channels]` (mel ⊕ onset_strength ⊕ spec_contrast)
    rather than the v6 mel-only input — the conv stack widens the channel
    count from `n_in_channels` to `hidden_dim` in the first conv.
    """

    def __init__(self, n_in_channels: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2)   # [B, n_in_channels, T]
        x = self.net(x)             # [B, H, T]
        return x.transpose(1, 2)    # [B, T, H]


# ---------------------------------------------------------------------------
# Dilated temporal conv stack (widens receptive field before attention)
# ---------------------------------------------------------------------------
class DilatedTCNBlock(nn.Module):
    """Residual dilated-conv block: LN -> Conv(dilated) -> GELU -> Dropout."""

    def __init__(self, hidden_dim: int, dilation: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.norm = nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T]
        y = self.norm(x)
        y = self.conv(y)
        y = F.gelu(y)
        y = self.drop(y)
        return x + y


class DilatedTCN(nn.Module):
    """Stack of dilated residual conv blocks. [B, T, H] -> [B, T, H]."""

    def __init__(
        self,
        hidden_dim: int,
        dilations: Tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DilatedTCNBlock(hidden_dim, d, dropout=dropout) for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, H] -> [B, H, T] -> blocks -> [B, H, T] -> [B, T, H]
        x = x.transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# FiLM conditioning (difficulty + density + song position + structure)
# ---------------------------------------------------------------------------
class FourierFeatures(nn.Module):
    """Random Fourier features for a scalar input."""

    def __init__(self, n_features: int = 16, sigma: float = 1.0):
        super().__init__()
        self.register_buffer('freqs', torch.randn(n_features) * sigma, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B] -> [B, 2*n_features]
        proj = x.unsqueeze(-1) * self.freqs * (2 * math.pi)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class FiLMConditioner(nn.Module):
    """
    FiLM conditioner for (difficulty, density, start_seconds, remaining_seconds,
    section_position, near_boundary).

    v8 adds section_position (0=start … 1=end of the detected section) and
    near_boundary (0/1 flag for frames close to a section boundary) so the
    model can adapt its predictions to song structure.
    """

    def __init__(
        self,
        n_difficulties: int,
        hidden_dim: int,
        n_fourier: int = 16,
        n_fourier_position: int = 16,
        position_sigma: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.diff_emb = nn.Embedding(n_difficulties, hidden_dim)
        self.fourier = FourierFeatures(n_fourier)
        self.fourier_start = FourierFeatures(n_fourier_position, sigma=position_sigma)
        self.fourier_remaining = FourierFeatures(n_fourier_position, sigma=position_sigma)
        self.fourier_section_pos = FourierFeatures(n_fourier_position, sigma=position_sigma)
        # near_boundary is a 0/1 flag — small embedding + Fourier.
        self.boundary_emb = nn.Embedding(2, 8)
        cond_dim = (
            hidden_dim                          # difficulty embedding
            + 2 * n_fourier                     # density
            + 2 * (2 * n_fourier_position)      # start + remaining
            + 2 * n_fourier_position            # section_position
            + 8                                  # near_boundary
        )
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)

        nn.init.normal_(self.diff_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.boundary_emb.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.bias.zero_()
            self.proj.bias[:hidden_dim] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        difficulty: torch.Tensor,
        density: torch.Tensor,
        start_seconds: torch.Tensor,
        remaining_seconds: torch.Tensor,
        section_position: torch.Tensor,
        near_boundary: torch.Tensor,
    ) -> torch.Tensor:
        d = self.diff_emb(difficulty)
        f = self.fourier(density)
        fs = self.fourier_start(start_seconds)
        fr = self.fourier_remaining(remaining_seconds)
        fsec = self.fourier_section_pos(section_position)
        bdry = self.boundary_emb(near_boundary.long())
        cond = torch.cat([d, f, fs, fr, fsec, bdry], dim=-1)
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


# ---------------------------------------------------------------------------
# RoPE (Rotary Positional Embedding) transformer
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """Cacheable rotary positional embedding for head-dim vectors."""

    def __init__(self, head_dim: int, max_len: int = 16384, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-2)
        cos = self.cos[:T].to(dtype=x.dtype)
        sin = self.sin[:T].to(dtype=x.dtype)
        return x * cos + _rotate_half(x) * sin


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rope(q)
        k = self.rope(k)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(B, T, D)
        return self.out(attn)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Full step chart model (v8)
# ---------------------------------------------------------------------------

class StepChartModel(nn.Module):
    """
    feats -> (onset_logits, sustain_logits, intensity_pred, arrow_logits).

    onset_logits:    [B, T, 1]  any-onset pre-sigmoid
    sustain_logits:  [B, T, 1]  "any arrow currently held" pre-sigmoid
    intensity_pred:  [B, T, 1]  continuous jump-vs-tap intensity (linear out)
    arrow_logits:    [B, T, 4]  per-arrow onset pre-sigmoid (L/D/U/R)
    """

    N_ARROWS = 4  # left, down, up, right

    def __init__(
        self,
        n_in_channels: int = 88,
        hidden_dim: int = 160,
        n_heads: int = 4,
        n_transformer_layers: int = 3,
        n_difficulties: int = 5,
        dropout: float = 0.1,
        tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8, 16),
        head_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_in_channels = int(n_in_channels)

        self.audio_encoder = AudioEncoder(n_in_channels, hidden_dim, dropout)
        self.tcn = DilatedTCN(hidden_dim, dilations=tcn_dilations, dropout=dropout)
        self.film = FiLMConditioner(n_difficulties, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, ff_mult=4, dropout=dropout)
            for _ in range(n_transformer_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        def _mlp_head(out_dim: int) -> nn.Sequential:
            layers: list = []
            for _ in range(max(1, head_layers)):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        self.onset_head = _mlp_head(1)
        self.sustain_head = _mlp_head(1)
        self.intensity_head = _mlp_head(1)
        self.arrow_heads = _mlp_head(self.N_ARROWS)  # [B, T, 4]

    def set_onset_prior(self, p: float) -> None:
        """Bias-init the onset head's final layer to logit(p)."""
        p = float(max(min(p, 0.999), 1e-4))
        bias = math.log(p / (1.0 - p))
        final = self.onset_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.constant_(final.bias, bias)

    def set_sustain_prior(self, p: float) -> None:
        """Bias-init the sustain head to logit(p) (any-arrow in-hold rate)."""
        p = float(max(min(p, 0.999), 1e-4))
        bias = math.log(p / (1.0 - p))
        final = self.sustain_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.constant_(final.bias, bias)

    def encode(
        self,
        feats: torch.Tensor,
        difficulty: torch.Tensor,
        density: torch.Tensor,
        start_seconds: torch.Tensor,
        remaining_seconds: torch.Tensor,
        section_position: torch.Tensor,
        near_boundary: torch.Tensor,
    ) -> torch.Tensor:
        """Backbone forward. [B, T, n_in_channels] -> [B, T, H]."""
        x = self.audio_encoder(feats)
        x = self.tcn(x)
        x = self.film(
            x, difficulty, density, start_seconds, remaining_seconds,
            section_position, near_boundary,
        )
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(
        self,
        feats: torch.Tensor,
        difficulty: torch.Tensor,
        density: torch.Tensor,
        start_seconds: torch.Tensor,
        remaining_seconds: torch.Tensor,
        section_position: Optional[torch.Tensor] = None,
        near_boundary: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feats: [B, T, n_in_channels]
            difficulty: [B] long
            density: [B] float (normalized)
            start_seconds: [B] float
            remaining_seconds: [B] float
            section_position: [B] float in [0, 1] — position within section (optional)
            near_boundary: [B] float 0/1 — near a section boundary (optional)

        Returns:
            onset_logits:    [B, T, 1] pre-sigmoid
            sustain_logits:  [B, T, 1] pre-sigmoid
            intensity_pred:  [B, T, 1] linear regression output
            arrow_logits:    [B, T, 4] per-arrow pre-sigmoid
        """
        if section_position is None:
            section_position = torch.zeros(
                feats.size(0), dtype=torch.float32, device=feats.device,
            )
        if near_boundary is None:
            near_boundary = torch.zeros(
                feats.size(0), dtype=torch.float32, device=feats.device,
            )
        features = self.encode(
            feats, difficulty, density, start_seconds, remaining_seconds,
            section_position, near_boundary,
        )
        onset_logits = self.onset_head(features)
        sustain_logits = self.sustain_head(features)
        intensity_pred = self.intensity_head(features)
        arrow_logits = self.arrow_heads(features)
        return onset_logits, sustain_logits, intensity_pred, arrow_logits


# ---------------------------------------------------------------------------
# Loss (v8 — adds arrow BCE loss)
# ---------------------------------------------------------------------------

class StepChartLoss(nn.Module):
    """
    Combined v8 loss:
        - focal BCE on onset_logits against Gaussian-smoothed any-onset target
        - focal BCE on sustain_logits against dense in-any-hold target
        - MSE on intensity_pred against jump-intensity target (smeared)
        - focal BCE on arrow_logits against per-arrow onset labels,
          masked to onset frames only (where any arrow fires)
    """

    def __init__(
        self,
        onset_pos_weight: float = 10.0,
        sustain_pos_weight: float = 5.0,
        focal_gamma: float = 2.0,
        sustain_weight: float = 1.0,
        intensity_weight: float = 0.5,
        arrow_weight: float = 0.3,
        arrow_pos_weight: float = 3.0,
    ):
        super().__init__()
        self.register_buffer(
            'onset_pos_weight',
            torch.tensor(float(onset_pos_weight), dtype=torch.float32),
        )
        self.register_buffer(
            'sustain_pos_weight',
            torch.tensor(float(sustain_pos_weight), dtype=torch.float32),
        )
        self.register_buffer(
            'arrow_pos_weight',
            torch.tensor(float(arrow_pos_weight), dtype=torch.float32),
        )
        self.focal_gamma = float(focal_gamma)
        self.sustain_weight = float(sustain_weight)
        self.intensity_weight = float(intensity_weight)
        self.arrow_weight = float(arrow_weight)

    def forward(
        self,
        onset_logits: torch.Tensor,      # [B, T, 1]
        sustain_logits: torch.Tensor,    # [B, T, 1]
        intensity_pred: torch.Tensor,    # [B, T, 1]
        arrow_logits: torch.Tensor,      # [B, T, 4]
        onset_soft: torch.Tensor,        # [B, T, 1]
        sustain_target: torch.Tensor,    # [B, T, 1]
        intensity_target: torch.Tensor,  # [B, T, 1]
        arrow_target: torch.Tensor,      # [B, T, 4] uint8
        onset_mask: Optional[torch.Tensor] = None,  # [B, T, 1] float — arrow loss mask
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Focal BCE on onset.
        if self.focal_gamma > 0.0:
            bce = F.binary_cross_entropy_with_logits(
                onset_logits, onset_soft,
                pos_weight=self.onset_pos_weight,
                reduction='none',
            )
            with torch.no_grad():
                p = torch.sigmoid(onset_logits)
                p_t = onset_soft * p + (1.0 - onset_soft) * (1.0 - p)
                focal_w = (1.0 - p_t).clamp_(min=0.0, max=1.0).pow(self.focal_gamma)
            onset_loss = (focal_w * bce).mean()
        else:
            onset_loss = F.binary_cross_entropy_with_logits(
                onset_logits, onset_soft,
                pos_weight=self.onset_pos_weight,
            )

        if self.focal_gamma > 0.0:
            bce_s = F.binary_cross_entropy_with_logits(
                sustain_logits, sustain_target,
                pos_weight=self.sustain_pos_weight,
                reduction='none',
            )
            with torch.no_grad():
                p_s = torch.sigmoid(sustain_logits)
                p_t_s = sustain_target * p_s + (1.0 - sustain_target) * (1.0 - p_s)
                focal_w_s = (1.0 - p_t_s).clamp_(min=0.0, max=1.0).pow(self.focal_gamma)
            sustain_loss = (focal_w_s * bce_s).mean()
        else:
            sustain_loss = F.binary_cross_entropy_with_logits(
                sustain_logits, sustain_target,
                pos_weight=self.sustain_pos_weight,
            )

        intensity_loss = F.mse_loss(intensity_pred, intensity_target)

        # Arrow loss: focal BCE, masked to onset frames only.
        if onset_mask is None:
            onset_mask = onset_soft.clamp(0.0, 1.0)
        # onset_mask is [B, T, 1] — expand to [B, T, 4]
        mask_4 = onset_mask.expand(-1, -1, 4)
        if self.focal_gamma > 0.0:
            bce_a = F.binary_cross_entropy_with_logits(
                arrow_logits, arrow_target,
                pos_weight=self.arrow_pos_weight,
                reduction='none',
            )
            with torch.no_grad():
                p_a = torch.sigmoid(arrow_logits)
                p_t_a = arrow_target * p_a + (1.0 - arrow_target) * (1.0 - p_a)
                focal_w_a = (1.0 - p_t_a).clamp_(min=0.0, max=1.0).pow(self.focal_gamma)
            arrow_loss = (focal_w_a * bce_a * mask_4).sum() / mask_4.sum().clamp(min=1.0)
        else:
            arrow_loss = F.binary_cross_entropy_with_logits(
                arrow_logits, arrow_target,
                pos_weight=self.arrow_pos_weight,
                reduction='none',
            )
            arrow_loss = (arrow_loss * mask_4).sum() / mask_4.sum().clamp(min=1.0)

        total = (
            onset_loss
            + self.sustain_weight * sustain_loss
            + self.intensity_weight * intensity_loss
            + self.arrow_weight * arrow_loss
        )
        return (
            total,
            onset_loss.detach(),
            sustain_loss.detach(),
            intensity_loss.detach(),
            arrow_loss.detach(),
        )

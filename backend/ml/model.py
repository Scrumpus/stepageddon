"""
Step Chart Neural Network Model.

Two-head onset-style architecture (Dance Dance Convolution formulation):
    mel -> AudioEncoder (CNN) -> Dilated TCN -> early FiLM (difficulty+density)
        -> RoPE Transformer -> LayerNorm
        -> onset head  (per-arrow sigmoid, "is there a note here?")
        -> type  head  (per-arrow 3-way: tap / hold_start / hold_end)

The type head is only supervised where onset target is positive.

Conditioning uses a single early FiLM layer. Difficulty is an embedding,
density is encoded with random Fourier features and concatenated before
projection, making the density signal harder to ignore than a zero-init
scalar linear.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Audio encoder (CNN)
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """CNN encoder for mel spectrogram features. [B, T, n_mels] -> [B, T, H]."""

    def __init__(self, n_mels: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1),
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

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.transpose(1, 2)   # [B, n_mels, T]
        x = self.net(x)           # [B, H, T]
        return x.transpose(1, 2)  # [B, T, H]


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

    def __init__(self, hidden_dim: int, dilations=(1, 2, 4, 8), dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DilatedTCNBlock(hidden_dim, d, dropout=dropout) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, H, T]
        for b in self.blocks:
            x = b(x)
        return x.transpose(1, 2)  # [B, T, H]


# ---------------------------------------------------------------------------
# Conditioning: difficulty embedding + Fourier features on density -> FiLM
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """
    Random Fourier features for a scalar input.

    sin/cos of random-frequency projections of the (normalized) density scalar.
    Output dim = 2 * n_features.
    """

    def __init__(self, n_features: int = 16, sigma: float = 1.0):
        super().__init__()
        self.register_buffer('freqs', torch.randn(n_features) * sigma, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B] -> [B, 2*n_features]
        proj = x.unsqueeze(-1) * self.freqs * (2 * math.pi)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class FiLMConditioner(nn.Module):
    """
    Single FiLM conditioner for (difficulty, density).

    Builds (gamma, beta) from a difficulty embedding concatenated with random
    Fourier features of density, and applies `gamma * x + beta` to features.

    Proj is init so that at step 0 the layer is an identity (gamma=1, beta=0)
    regardless of input, so training isn't destabilized.
    """

    def __init__(
        self,
        n_difficulties: int,
        hidden_dim: int,
        n_fourier: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.diff_emb = nn.Embedding(n_difficulties, hidden_dim)
        self.fourier = FourierFeatures(n_fourier)
        cond_dim = hidden_dim + 2 * n_fourier
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)

        nn.init.normal_(self.diff_emb.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            # bias: gamma init = 1, beta init = 0
            self.proj.bias.zero_()
            self.proj.bias[:hidden_dim] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        difficulty: torch.Tensor,
        density: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, T, H], difficulty: [B], density: [B]
        d = self.diff_emb(difficulty)           # [B, H]
        f = self.fourier(density)                # [B, 2*n_fourier]
        cond = torch.cat([d, f], dim=-1)         # [B, H + 2*n_fourier]
        params = self.proj(cond)                 # [B, 2H]
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
        freqs = torch.outer(t, inv_freq)                    # [max_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)              # [max_len, head_dim]
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, Dh]
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
        qkv = qkv.permute(2, 0, 3, 1, 4)                       # [3, B, H, T, Dh]
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
# Full step chart model
# ---------------------------------------------------------------------------

class StepChartModel(nn.Module):
    """
    Audio -> (onset_logits, type_logits).

    onset_logits: [B, T, 4]      per-arrow "note present" pre-sigmoid
    type_logits:  [B, T, 4, 3]   per-arrow 3-way {tap, hold_start, hold_end}
    """

    N_ARROWS = 4
    N_TYPES = 3  # tap=0, hold_start=1, hold_end=2

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_transformer_layers: int = 4,
        n_difficulties: int = 5,
        dropout: float = 0.1,
        tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.audio_encoder = AudioEncoder(n_mels, hidden_dim, dropout)
        self.tcn = DilatedTCN(hidden_dim, dilations=tcn_dilations, dropout=dropout)
        self.film = FiLMConditioner(n_difficulties, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, ff_mult=4, dropout=dropout)
            for _ in range(n_transformer_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        self.onset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.N_ARROWS),
        )
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.N_ARROWS * self.N_TYPES),
        )

    def set_onset_prior(self, p: float) -> None:
        """Initialize the onset head's final bias to logit(p).

        Call once after construction with the empirical per-frame per-arrow
        onset rate. This alone typically removes the need for focal loss.
        """
        p = float(max(min(p, 0.999), 1e-4))
        bias = math.log(p / (1.0 - p))
        final = self.onset_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.constant_(final.bias, bias)

    def forward(
        self,
        mel: torch.Tensor,
        difficulty: torch.Tensor,
        density: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: [B, T, n_mels]
            difficulty: [B] long
            density: [B] float (normalized)

        Returns:
            onset_logits: [B, T, 4]
            type_logits:  [B, T, 4, 3]
        """
        x = self.audio_encoder(mel)                 # [B, T, H]
        x = self.tcn(x)                              # [B, T, H]
        x = self.film(x, difficulty, density)        # [B, T, H]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                             # [B, T, H]

        onset_logits = self.onset_head(x)            # [B, T, 4]
        type_flat = self.type_head(x)                # [B, T, 4*3]
        B, T, _ = type_flat.shape
        type_logits = type_flat.view(B, T, self.N_ARROWS, self.N_TYPES)
        return onset_logits, type_logits


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class StepChartLoss(nn.Module):
    """
    Combined loss:
        - onset BCEWithLogits with pos_weight on soft (Gaussian-smoothed) targets
        - 3-way cross-entropy on type head, masked to frames with an onset
    """

    def __init__(self, pos_weight: float = 10.0, type_weight: float = 1.0):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))
        self.type_weight = type_weight

    def forward(
        self,
        onset_logits: torch.Tensor,   # [B, T, 4]
        type_logits: torch.Tensor,    # [B, T, 4, 3]
        onset_soft: torch.Tensor,     # [B, T, 4] float in [0,1]
        type_target: torch.Tensor,    # [B, T, 4] long, -100 where no note
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        onset_loss = F.binary_cross_entropy_with_logits(
            onset_logits, onset_soft, pos_weight=self.pos_weight
        )
        B, T, A, C = type_logits.shape
        type_flat = type_logits.reshape(-1, C)
        target_flat = type_target.reshape(-1)
        type_loss = F.cross_entropy(type_flat, target_flat, ignore_index=-100)
        total = onset_loss + self.type_weight * type_loss
        return total, onset_loss.detach(), type_loss.detach()

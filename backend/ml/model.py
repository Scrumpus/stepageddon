"""
Step Chart Neural Network Model.

CNN audio encoder + Transformer with FiLM difficulty conditioning.
Predicts per-frame note states for 4 arrows.
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(seq_len: int, d_head: int, device, base: float = 10000.0):
    """Precompute rotary cos/sin tables for a given sequence length."""
    half = d_head // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.einsum('t,f->tf', t, freqs)  # [T, half]
    return angles.cos(), angles.sin()            # [T, half], [T, half]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to q or k tensors.

    Args:
        x: [B, T, H, d_head]
        cos, sin: [T, d_head/2]
    """
    # Split last dim into two halves: (x_even, x_odd) interleaved.
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    out = torch.stack([rot_even, rot_odd], dim=-1)
    return out.reshape(*x.shape)


class RoPEMultiheadAttention(nn.Module):
    """Minimal self-attention block with rotary position embeddings.

    Uses ``scaled_dot_product_attention`` so FlashAttention is picked
    automatically when available. Replaces the sinusoidal PE + standard
    nn.TransformerEncoder path so the model extrapolates to longer
    sequences than it was trained on.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, rope_cos, rope_sin) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, d_head]
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        q = q.transpose(1, 2)  # [B, H, T, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(B, T, D)
        return self.out(attn)


class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiheadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.dropout(self.attn(self.norm1(x), rope_cos, rope_sin))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class RoPETransformerEncoder(nn.Module):
    """Stack of RoPE encoder layers sharing one cos/sin cache per call."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.d_head = d_model // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        cos, sin = build_rope_cache(T, self.d_head, x.device)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


class AudioEncoder(nn.Module):
    """CNN encoder for mel spectrogram features."""

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
        """
        Args:
            mel: [B, T, n_mels]
        Returns:
            [B, T, hidden_dim]
        """
        x = mel.transpose(1, 2)   # [B, n_mels, T]
        x = self.net(x)           # [B, hidden_dim, T]
        return x.transpose(1, 2)  # [B, T, hidden_dim]


class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation for difficulty + density conditioning.

    The base modulation comes from a per-difficulty embedding. A learned linear
    projection of the (normalized) density scalar is *added* to those FiLM
    parameters, so the model can shift gamma/beta along a continuous axis
    rather than only picking from 5 discrete points.

    The density projection is zero-initialized so that at step 0 this layer is
    numerically identical to the difficulty-only version — the model learns to
    use the density signal via gradients rather than being destabilized by it.
    """

    def __init__(self, n_conditions: int, hidden_dim: int):
        super().__init__()
        self.film = nn.Embedding(n_conditions, hidden_dim * 2)
        self.density_proj = nn.Linear(1, hidden_dim * 2)
        # Initialize gamma near 1 and beta near 0 with small random perturbation
        # so each difficulty embedding starts distinct (gradients can separate them).
        nn.init.normal_(self.film.weight[:, :hidden_dim], mean=1.0, std=0.05)
        nn.init.normal_(self.film.weight[:, hidden_dim:], mean=0.0, std=0.05)
        # Zero-init density projection: layer starts as a no-op contribution.
        nn.init.zeros_(self.density_proj.weight)
        nn.init.zeros_(self.density_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        density: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, hidden_dim]
            condition: [B] int indices
            density: [B] float, normalized density (see dataset.DENSITY_MEAN/STD)
        Returns:
            [B, T, hidden_dim]
        """
        params = self.film(condition) + self.density_proj(density.unsqueeze(-1))
        gamma = params[:, :x.size(2)].unsqueeze(1)   # [B, 1, hidden_dim]
        beta = params[:, x.size(2):].unsqueeze(1)    # [B, 1, hidden_dim]
        return gamma * x + beta


class StepChartModel(nn.Module):
    """
    Full model: audio → step chart predictions.

    Architecture:
        1. CNN encodes mel spectrogram into frame features
        2. FiLM conditions on difficulty level
        3. Transformer captures temporal context
        4. Linear head predicts note states per arrow per frame

    Output: dict of three independent sigmoid heads per arrow
        - 'tap'        [B, T, 4]: probability an arrow is a tap at this frame
        - 'hold_state' [B, T, 4]: probability arrow is *inside* a hold (incl. start)
        - 'hold_end'   [B, T, 4]: probability this is the final frame of a hold

    This replaces the previous 4-way softmax per arrow. The softmax
    formulation capped tap probability near 0.3 (since it competed with
    'none') and made holds brittle because hold_start/hold_end had to be
    paired by a fragile decoder. Per-frame hold *state* is easy to learn
    and decoded by contiguous runs.
    """

    N_ARROWS = 4
    N_OUTPUTS_PER_ARROW = 3  # tap, hold_state, hold_end (all sigmoid)

    def __init__(
        self,
        # Despite the name, this is the input feature dimension; we keep
        # the argument called ``n_mels`` for historical compatibility.
        # Default is 80 mel bins + 2 rhythm channels (onset_strength,
        # beat_phase) — see prepare_data.N_INPUT_CHANNELS.
        n_mels: int = 82,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_transformer_layers: int = 4,
        n_difficulties: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.audio_encoder = AudioEncoder(n_mels, hidden_dim, dropout)
        self.film = FiLMConditioning(n_difficulties, hidden_dim)
        self.film_post = FiLMConditioning(n_difficulties, hidden_dim)

        # Rotary-position self-attention replaces the old sinusoidal PE +
        # nn.TransformerEncoder combo so the model can extrapolate cleanly
        # to chunks longer than those it was trained on (e.g. full-song
        # inference) without retraining.
        self.transformer = RoPETransformerEncoder(
            d_model=hidden_dim,
            n_heads=n_heads,
            n_layers=n_transformer_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.N_ARROWS * self.N_OUTPUTS_PER_ARROW),
        )

        # Density regression head: pools the final transformer features
        # over time and regresses the chunk's normalized step density. This
        # is a supervised check on the FiLM density input — the model is
        # forced to reconstruct the very scalar it was conditioned on from
        # audio, which prevents it from ignoring the density signal and
        # turning the FiLM path into dead weight.
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Auxiliary arrow-agnostic placement head (à la DDC). Trained with
        # BCE against "any arrow has a step here" labels. At inference we
        # combine it with the per-arrow tap head as
        #     p(tap_i) ≈ sigmoid(placement) * sigmoid(tap_i)
        # which gives an absolute confidence signal and lets us drop the
        # post-hoc density top-N capping hack.
        self.placement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        mel: torch.Tensor,
        difficulty: torch.Tensor,
        density: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel: [B, T, n_mels] mel spectrogram
            difficulty: [B] integer difficulty indices (0-4)
            density: [B] float, normalized step density (see dataset.DENSITY_MEAN/STD).
                At training this is the empirical per-chunk density; at inference
                it's the density the caller wants the chart to have.

        Returns:
            dict with logit tensors:
                'tap'        [B, T, 4]
                'hold_state' [B, T, 4]
                'hold_end'   [B, T, 4]
                'placement'  [B, T]      arrow-agnostic "is there a step here?"
        """
        x = self.audio_encoder(mel)                     # [B, T, hidden]
        x = self.film(x, difficulty, density)           # [B, T, hidden]  early conditioning
        x = self.transformer(x)                         # [B, T, hidden]  (RoPE inside)
        x = self.film_post(x, difficulty, density)      # [B, T, hidden]  late conditioning
        logits = self.output_head(x)                    # [B, T, 12]

        B, T, _ = logits.shape
        logits = logits.view(B, T, self.N_ARROWS, self.N_OUTPUTS_PER_ARROW)
        placement_logit = self.placement_head(x).squeeze(-1)  # [B, T]
        density_pred = self.density_head(x.mean(dim=1)).squeeze(-1)  # [B]
        return {
            'tap':        logits[..., 0],
            'hold_state': logits[..., 1],
            'hold_end':   logits[..., 2],
            'placement':  placement_logit,
            'density':    density_pred,
        }


class MultiHeadFocalLoss(nn.Module):
    """
    Focal BCE loss summed across the three sigmoid heads.

    Each head is a heavy class imbalance (most frames are negative), so we
    use per-head `pos_weight` (≈ neg/pos ratio) plus a focal modulator that
    down-weights easy examples. The three head losses are averaged so the
    loss magnitude is comparable to the old single-softmax loss.
    """

    HEADS = ('tap', 'hold_state', 'hold_end')  # per-arrow heads ([B,T,4])
    AUX_HEADS = ('placement',)                 # arrow-agnostic heads ([B,T])

    def __init__(
        self,
        pos_weights: torch.Tensor = None,
        gamma: float = 2.0,
        head_weights: Dict[str, float] = None,
        placement_pos_weight: float = None,
        placement_loss_weight: float = 0.5,
        density_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.gamma = gamma
        if pos_weights is not None:
            assert pos_weights.numel() == len(self.HEADS), \
                f"pos_weights must have {len(self.HEADS)} entries (tap, hold_state, hold_end)"
            self.register_buffer('pos_weights', pos_weights.float())
        else:
            self.pos_weights = None
        if placement_pos_weight is not None:
            self.register_buffer(
                'placement_pos_weight',
                torch.tensor(float(placement_pos_weight)),
            )
        else:
            self.placement_pos_weight = None
        self.placement_loss_weight = placement_loss_weight
        self.density_loss_weight = density_loss_weight
        self.head_weights = head_weights or {h: 1.0 for h in self.HEADS}

    def _focal_bce(self, logit, target, pos_weight):
        bce = F.binary_cross_entropy_with_logits(
            logit, target, reduction='none', pos_weight=pos_weight,
        )
        with torch.no_grad():
            p = torch.sigmoid(logit.detach())
            pt = target * p + (1.0 - target) * (1.0 - p)
            focal = (1.0 - pt).clamp_min(1e-6) ** self.gamma
        return (focal * bce).mean()

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        placement_target: torch.Tensor = None,
        density_target: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: dict containing 'tap', 'hold_state', 'hold_end' each
                [B, T, 4], optionally 'placement' [B, T], and optionally
                'density' [B].
            targets: [B, T, 4, 3] float in {0, 1}; last dim aligned with HEADS
            placement_target: optional [B, T] float in {0, 1}
            density_target: optional [B] float (normalized per-chunk
                density; same scale as the FiLM density input)
        """
        total = 0.0
        total_w = 0.0
        for i, head in enumerate(self.HEADS):
            logit = logits[head]                 # [B, T, 4]
            target = targets[..., i].float()     # [B, T, 4]
            pw = self.pos_weights[i] if self.pos_weights is not None else None
            hw = self.head_weights.get(head, 1.0)
            total = total + hw * self._focal_bce(logit, target, pw)
            total_w += hw

        if placement_target is not None and 'placement' in logits:
            pl_logit = logits['placement']               # [B, T]
            pl_target = placement_target.float()
            pw = self.placement_pos_weight
            hw = self.placement_loss_weight
            total = total + hw * self._focal_bce(pl_logit, pl_target, pw)
            total_w += hw

        if (
            density_target is not None
            and 'density' in logits
            and self.density_loss_weight > 0
        ):
            dpred = logits['density']  # [B]
            mse = F.mse_loss(dpred, density_target.float())
            total = total + self.density_loss_weight * mse
            total_w += self.density_loss_weight

        return total / max(total_w, 1e-8)


# Backwards-compat alias — old imports won't break while training scripts
# transition over. New code should import MultiHeadFocalLoss directly.
FocalLoss = MultiHeadFocalLoss

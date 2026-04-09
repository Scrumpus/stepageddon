"""
Step Chart Neural Network Model.

CNN audio encoder + Transformer with FiLM difficulty conditioning.
Predicts per-frame note states for 4 arrows.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 8000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


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

    Output shape: [B, T, 4, 4]
        - 4 arrows (left, down, up, right)
        - 4 states per arrow (none, tap, hold_start, hold_end)
    """

    N_ARROWS = 4
    N_NOTE_TYPES = 4  # none=0, tap=1, hold_start=2, hold_end=3

    def __init__(
        self,
        n_mels: int = 80,
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
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.N_ARROWS * self.N_NOTE_TYPES),
        )

    def forward(
        self,
        mel: torch.Tensor,
        difficulty: torch.Tensor,
        density: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mel: [B, T, n_mels] mel spectrogram
            difficulty: [B] integer difficulty indices (0-4)
            density: [B] float, normalized step density (see dataset.DENSITY_MEAN/STD).
                At training this is the empirical per-chunk density; at inference
                it's the density the caller wants the chart to have.

        Returns:
            [B, T, 4, 4] logits for each arrow × note type
        """
        x = self.audio_encoder(mel)                     # [B, T, hidden]
        x = self.film(x, difficulty, density)           # [B, T, hidden]  early conditioning
        x = self.pos_encoding(x)                        # [B, T, hidden]
        x = self.transformer(x)                         # [B, T, hidden]
        x = self.film_post(x, difficulty, density)      # [B, T, hidden]  late conditioning
        logits = self.output_head(x)                    # [B, T, 16]

        B, T, _ = logits.shape
        return logits.view(B, T, self.N_ARROWS, self.N_NOTE_TYPES)


class FocalLoss(nn.Module):
    """
    Focal loss for handling severe class imbalance.

    Most frames have no notes, so standard CE would overfit to predicting 'none'.
    Focal loss down-weights easy (correct) predictions and focuses on hard examples.
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, T, 4, 4] raw predictions
            targets: [B, T, 4] integer class labels (0-3)
        """
        B, T, A, C = logits.shape
        logits_flat = logits.reshape(-1, C)      # [B*T*4, 4]
        targets_flat = targets.reshape(-1)        # [B*T*4]

        ce = nn.functional.cross_entropy(logits_flat, targets_flat, reduction='none')
        pt = torch.exp(-ce)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets_flat]
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce).mean()

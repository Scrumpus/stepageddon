"""
Step Chart Neural Network Model.

Four-head architecture (arrow-aware variant of Dance Dance Convolution):
    mel -> AudioEncoder (CNN) -> Dilated TCN -> early FiLM (difficulty+density)
        -> RoPE Transformer -> LayerNorm
        -> arrow head     (4 sigmoids, per-arrow "does this arrow fire?",
                           conditioned on the previous emitted arrow vector)
        -> type  head     (3-way: tap / jump / hold_start)
        -> duration head  (linear scalar in log-space; decode_duration() to seconds)
        -> beat head      (auxiliary: "is this a beat?")

Onset detection at inference time = `arrow_logits.sigmoid().max(-1)`. The
type head is supervised where any arrow fires; the duration head where the
type target is hold_start (plus a ±2-frame window).

The duration head emits a raw scalar interpreted as log(seconds + DURATION_OFFSET).
This converts the wide raw-seconds dynamic range (~0.2–5s) into a much
better-conditioned ~[-1.6, 1.6] target range and lets the loss treat
relative error symmetrically. Inference must call `decode_duration` to
get back to seconds.

Conditioning uses a single early FiLM layer. Difficulty is an embedding,
density is encoded with random Fourier features and concatenated before
projection, making the density signal harder to ignore than a zero-init
scalar linear.

The arrow head consumes a 4-dim binary `prev_arrow` vector (the arrow set
emitted at the most recent prior onset, or zeros pre-onset). This makes the
head a per-frame Markov chain over arrows, P(arrow_t | audio_t, arrow_{prev}),
which is what fixes the L-R-L-R mode-collapse seen with independent Bernoullis.
"""

import math
from typing import Optional, Tuple

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
    Single FiLM conditioner for (difficulty, density, song-position).

    Builds (gamma, beta) from a difficulty embedding concatenated with random
    Fourier features of density and song-position scalars (start_seconds,
    remaining_seconds), and applies `gamma * x + beta` to features.

    The position scalars let the model learn empirical edge-of-song patterns
    (e.g., charts typically have several seconds of silence before the first
    arrow, and often taper near the end). Per-chunk conditioning is
    sufficient because the transformer's RoPE positions resolve intra-chunk
    location; the FiLM signal only has to disambiguate "this chunk is at
    song start" from "this chunk is mid-song".

    Proj is init so that at step 0 the layer is an identity (gamma=1, beta=0)
    regardless of input, so training isn't destabilized.
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
        # Separate Fourier modules for the two position scalars. Sigma is
        # tuned for the seconds scale: with sigma=0.1, the random frequencies
        # have periods ranging from ~10s up to a few hundred seconds, which
        # spans both intro silence (sub-10s) and song-section structure.
        self.fourier_start = FourierFeatures(n_fourier_position, sigma=position_sigma)
        self.fourier_remaining = FourierFeatures(n_fourier_position, sigma=position_sigma)
        cond_dim = hidden_dim + 2 * n_fourier + 2 * (2 * n_fourier_position)
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
        start_seconds: torch.Tensor,
        remaining_seconds: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, T, H]; difficulty: [B]; density/start_seconds/remaining_seconds: [B]
        d = self.diff_emb(difficulty)                      # [B, H]
        f = self.fourier(density)                          # [B, 2*n_fourier]
        fs = self.fourier_start(start_seconds)             # [B, 2*n_fourier_position]
        fr = self.fourier_remaining(remaining_seconds)     # [B, 2*n_fourier_position]
        cond = torch.cat([d, f, fs, fr], dim=-1)
        params = self.proj(cond)                            # [B, 2H]
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
    Audio + prev_arrow -> (arrow_logits, type_logits, duration_pred, beat_logits).

    Arrow-aware: the model predicts WHICH arrows fire (4 independent Bernoullis
    per frame, conditioned on the previous emitted arrow vector), WHAT type
    of note it is (tap / jump / hold_start), and HOW LONG holds last.

    arrow_logits:  [B, T, 4]  per-arrow {L, D, U, R} pre-sigmoid
    type_logits:   [B, T, 3]  {tap, jump, hold_start}
    duration_pred: [B, T, 1]  predicted hold duration in log-space
    beat_logits:   [B, T, 1]  auxiliary "is this a beat?" pre-sigmoid
    """

    N_NOTE_TYPES = 3
    TYPE_TAP = 0
    TYPE_JUMP = 1
    TYPE_HOLD_START = 2
    N_ARROWS = 4
    PREV_ARROW_DIM = 4  # 4-dim binary vector input to the arrow head

    # Offset added inside log() to keep the duration target away from -inf
    # while preserving relative ordering at the small end of the range.
    # Used by both `encode_duration` (training target) and `decode_duration`
    # (inference). Must stay in lockstep across model + dataset + inference.
    DURATION_OFFSET: float = 0.05

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_transformer_layers: int = 4,
        n_difficulties: int = 5,
        dropout: float = 0.1,
        tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        onset_head_layers: int = 3,
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

        # Deeper arrow head: arrow prediction is the harder head, so give it
        # extra capacity. `onset_head_layers` counts hidden Linear→GELU blocks
        # before the final projection (default 3).
        def _build_mlp_head(out_dim: int, n_hidden: int) -> nn.Sequential:
            layers: list = []
            for _ in range(max(1, n_hidden)):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        # Arrow head: backbone feats + 4-dim prev_arrow vector → per-arrow
        # logit. The first Linear absorbs the +4 conditioning dims; the rest
        # of the head matches the legacy onset head shape.
        cond_layers: list = [
            nn.Linear(hidden_dim + self.PREV_ARROW_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(1, onset_head_layers) - 1):
            cond_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        cond_layers.append(nn.Linear(hidden_dim, self.N_ARROWS))
        self.arrow_head = nn.Sequential(*cond_layers)
        # Small uniform init on the prev-arrow conditioning columns of the
        # first Linear. Zero-init starves the conditioning of gradient signal
        # in early epochs and the head settles into a marginal-matching local
        # minimum that ignores prev_arrow; a small non-zero init kicks the
        # gradients alive from step 1 while still being small enough not to
        # dominate the audio features.
        with torch.no_grad():
            first_lin = self.arrow_head[0]
            assert isinstance(first_lin, nn.Linear)
            first_lin.weight[:, hidden_dim:].uniform_(-0.02, 0.02)

        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.N_NOTE_TYPES),
        )
        # Duration head emits a raw scalar interpreted as
        # log(duration_seconds + DURATION_OFFSET). No Softplus — the log
        # transform itself maps the natural [0.05, ~5]s range into roughly
        # [-3, 1.6], which is a well-conditioned regression target.
        # Inference must apply `decode_duration` to recover seconds.
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Auxiliary beat head: predicts per-frame "is this a beat?" from the
        # same backbone features. Same depth/shape as the legacy onset head —
        # beats and onsets share the rhythmic structure the backbone is
        # learning. Loss-weighted at ~0.3 so beat learning helps the backbone
        # without crowding out the primary onset task.
        self.beat_head = _build_mlp_head(1, onset_head_layers)

    def set_arrow_priors(self, priors) -> None:
        """Initialize the arrow head's final bias to logit(p_a) per arrow.

        priors: iterable of length N_ARROWS giving the per-arrow positive
            rate P(arrow_a fires at any given frame). Anchors the head at
            the empirical marginal so step-0 predictions match base rates
            instead of starting from random logits — same idea as the old
            `set_onset_prior` but per-arrow.
        """
        priors = torch.as_tensor(list(priors), dtype=torch.float32)
        assert priors.numel() == self.N_ARROWS, (
            f"expected {self.N_ARROWS} priors, got {priors.numel()}"
        )
        priors = priors.clamp(min=1e-4, max=0.999)
        bias = torch.log(priors / (1.0 - priors))
        final = self.arrow_head[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            final.bias.copy_(bias)

    def set_beat_prior(self, p: float) -> None:
        """Initialize the beat head's final bias to logit(p).

        Beats are denser than onsets (~120-180 BPM × songs) — typically 1-3%
        of frames at 100 fps depending on tempo. Anchoring the head at the
        empirical rate gives the auxiliary task a sane starting point.
        """
        p = float(max(min(p, 0.999), 1e-4))
        bias = math.log(p / (1.0 - p))
        final = self.beat_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.constant_(final.bias, bias)

    @classmethod
    def encode_duration(cls, seconds: torch.Tensor) -> torch.Tensor:
        """Convert a (batch of) duration in seconds to the log-space target."""
        return torch.log(seconds.clamp_min(0.0) + cls.DURATION_OFFSET)

    @classmethod
    def decode_duration(cls, log_value):
        """Inverse of `encode_duration`. Accepts torch tensor or numpy array."""
        if isinstance(log_value, torch.Tensor):
            return torch.exp(log_value) - cls.DURATION_OFFSET
        import numpy as _np
        return _np.exp(log_value) - cls.DURATION_OFFSET

    def set_duration_prior(self, median_seconds: float) -> None:
        """Initialize the duration head's final bias to log(median+offset).

        Call once after construction with the empirical median hold duration.
        Without this, the freshly-built model emits ~0 (= 1s after decode),
        which is a fine starting guess but converges slowly. Anchoring to
        the corpus median saves several epochs of "find the right scale".
        """
        median_seconds = float(max(median_seconds, 0.0))
        bias = math.log(median_seconds + self.DURATION_OFFSET)
        final = self.duration_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.constant_(final.bias, bias)

    def set_type_prior(self, priors) -> None:
        """Initialize the type head's final bias to log(prior) per class.

        priors: iterable of length N_NOTE_TYPES giving P(class | onset)
            for {tap, jump, hold_start}. With this init, a freshly built
            model produces softmax probabilities matching the empirical
            class distribution at step 0, which is the right starting
            point on a heavily imbalanced 3-way head.
        """
        priors = torch.as_tensor(list(priors), dtype=torch.float32)
        assert priors.numel() == self.N_NOTE_TYPES, (
            f"expected {self.N_NOTE_TYPES} priors, got {priors.numel()}"
        )
        priors = priors.clamp_min(1e-6)
        priors = priors / priors.sum()
        log_priors = torch.log(priors)
        final = self.type_head[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            final.bias.copy_(log_priors)

    def encode(
        self,
        mel: torch.Tensor,
        difficulty: torch.Tensor,
        density: torch.Tensor,
        start_seconds: torch.Tensor,
        remaining_seconds: torch.Tensor,
    ) -> torch.Tensor:
        """Backbone-only forward. Returns per-frame features [B, T, H].

        Inference uses this to share backbone work across calls that vary only
        the prev_arrow conditioning input (sequential arrow decoding). Training
        still goes through the full `forward`.
        """
        x = self.audio_encoder(mel)
        x = self.tcn(x)
        x = self.film(x, difficulty, density, start_seconds, remaining_seconds)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def apply_arrow_head(
        self, features: torch.Tensor, prev_arrow: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Arrow head, given backbone features. [B, T, H] -> [B, T, 4]."""
        if prev_arrow is None:
            prev_arrow = features.new_zeros(
                features.size(0), features.size(1), self.PREV_ARROW_DIM,
            )
        arrow_in = torch.cat([features, prev_arrow], dim=-1)
        return self.arrow_head(arrow_in)

    def apply_type_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.type_head(features)

    def apply_duration_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.duration_head(features)

    def apply_beat_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.beat_head(features)

    def forward(
        self,
        mel: torch.Tensor,
        difficulty: torch.Tensor,
        density: torch.Tensor,
        start_seconds: torch.Tensor,
        remaining_seconds: torch.Tensor,
        prev_arrow: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: [B, T, n_mels]
            difficulty: [B] long
            density: [B] float (normalized)
            start_seconds: [B] float — chunk's start time from song start (seconds)
            remaining_seconds: [B] float — song length minus chunk end time (seconds)
            prev_arrow: [B, T, 4] float — per-frame "previous emitted arrow"
                vector for the arrow head's transition conditioning. If None
                (legacy callers / one-shot inference), zeros are used —
                equivalent to "no prior context".

        Returns:
            arrow_logits:  [B, T, 4]  per-arrow {L, D, U, R} pre-sigmoid
            type_logits:   [B, T, 3]  {tap, jump, hold_start}
            duration_pred: [B, T, 1]  log-space scalar; decode via `decode_duration`
            beat_logits:   [B, T, 1]  auxiliary beat-frame prediction (pre-sigmoid)
        """
        features = self.encode(
            mel, difficulty, density, start_seconds, remaining_seconds,
        )                                                       # [B, T, H]
        arrow_logits = self.apply_arrow_head(features, prev_arrow)
        type_logits = self.apply_type_head(features)
        duration_pred = self.apply_duration_head(features)
        beat_logits = self.apply_beat_head(features)
        return arrow_logits, type_logits, duration_pred, beat_logits


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class StepChartLoss(nn.Module):
    """
    Combined loss:
        - per-arrow BCEWithLogits with per-arrow pos_weight on soft
          (Gaussian-smoothed) targets
            arrow_logits: [B, T, 4], arrow_soft: [B, T, 4]
        - diversity KL: per-chunk arrow distribution should match the target
          arrow distribution; punishes mode collapse to one arrow
        - 3-way cross-entropy on type head, masked to frames with an onset
            type_logits: [B, T, 3], type_target: [B, T] long (-100 where no note)
        - Smooth-L1 on duration head, masked to hold_start frames only
            duration_pred: [B, T, 1], duration_target: [B, T] float (0 where no hold)
    """

    HOLD_START_CLASS = 2  # type target class index for hold_start
    # Duration loss is supervised on hold_start frames AND a small ±N-frame
    # window around them (using the same target). This 5×s the gradient
    # signal on a head that otherwise sees ~0.09% of frames per batch.
    DURATION_WINDOW = 2  # frames either side of hold_start

    def __init__(
        self,
        pos_weight=10.0,
        type_weight: float = 1.0,
        duration_weight: float = 1.0,
        type_class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 0.0,
        type_focal_gamma: float = 0.0,
        beat_weight: float = 0.0,
        beat_pos_weight: float = 10.0,
        diversity_weight: float = 0.2,
        commit_weight: float = 0.5,
        jack_weight: float = 0.3,
        jack_stride: int = 3,
    ):
        super().__init__()
        # Per-arrow pos_weight: scalar broadcasts across the 4 arrows; a
        # length-4 tensor sets a distinct weight per arrow (recommended —
        # outer arrows L/R fire ~2× as often as D/U in DDR data, so a uniform
        # scalar over-weights D/U positives).
        pw = torch.as_tensor(pos_weight, dtype=torch.float32)
        if pw.ndim == 0:
            pw = pw.expand(4).clone()
        assert pw.numel() == 4, f"pos_weight must be scalar or length-4, got {pw.shape}"
        self.register_buffer('pos_weight', pw)
        self.register_buffer('beat_pos_weight', torch.tensor(float(beat_pos_weight)))
        self.type_weight = type_weight
        self.duration_weight = duration_weight
        self.beat_weight = float(beat_weight)
        self.diversity_weight = float(diversity_weight)
        self.commit_weight = float(commit_weight)
        # Anti-jack penalty: discourages same-arrow argmax on pairs of co-active
        # onset frames separated by `jack_stride` frames. Stride is chosen just
        # outside the Gaussian onset-smoothing radius (~2 frames at sigma=1.5)
        # so we penalize distinct predicted events, not the same event spread
        # across adjacent frames. Pushes stream coherence toward 0.7+ in dense
        # sections where per-arrow BCE alone is satisfied by marginal-matching.
        self.jack_weight = float(jack_weight)
        self.jack_stride = int(jack_stride)
        self.focal_gamma = float(focal_gamma)
        # Focal modulation on the type CE: down-weights confidently-correct
        # frames so the rare jump/hold classes contribute relatively more
        # gradient. With type_class_weights alone, easy taps still dominate
        # the loss because there are 90× more of them than rare classes;
        # focal weighting fixes that without changing the static weights.
        self.type_focal_gamma = float(type_focal_gamma)
        # Per-class CE weights for the type head ({tap, jump, hold_start}).
        # Without these, the head collapses to "always predict tap" because
        # taps dominate the imbalanced supervision. Registered as a buffer
        # so it moves with .to(device) and survives state-dict round-trips.
        if type_class_weights is None:
            type_class_weights = torch.ones(3, dtype=torch.float32)
        else:
            type_class_weights = torch.as_tensor(
                type_class_weights, dtype=torch.float32
            )
            assert type_class_weights.numel() == 3
        self.register_buffer('type_class_weights', type_class_weights)

    def forward(
        self,
        arrow_logits: torch.Tensor,    # [B, T, 4]
        type_logits: torch.Tensor,     # [B, T, 3]
        duration_pred: torch.Tensor,   # [B, T, 1]
        arrow_soft: torch.Tensor,      # [B, T, 4] float in [0,1]
        type_target: torch.Tensor,     # [B, T] long, -100 where no note
        duration_target: torch.Tensor, # [B, T] float, seconds at hold_start, 0 elsewhere
        beat_logits: Optional[torch.Tensor] = None,  # [B, T, 1] pre-sigmoid
        beat_soft: Optional[torch.Tensor] = None,    # [B, T, 1] float in [0,1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Per-arrow BCE. pos_weight is broadcast over [B, T, 4] from its [4]
        # buffer — F.bce_with_logits handles the broadcast.
        if self.focal_gamma > 0.0:
            bce = F.binary_cross_entropy_with_logits(
                arrow_logits, arrow_soft, pos_weight=self.pos_weight,
                reduction='none',
            )
            with torch.no_grad():
                p = torch.sigmoid(arrow_logits)
                p_t = arrow_soft * p + (1.0 - arrow_soft) * (1.0 - p)
                focal_w = (1.0 - p_t).clamp_(min=0.0, max=1.0).pow(self.focal_gamma)
            arrow_loss = (focal_w * bce).mean()
        else:
            arrow_loss = F.binary_cross_entropy_with_logits(
                arrow_logits, arrow_soft, pos_weight=self.pos_weight,
            )

        # Per-frame commit CE: on frames where exactly one arrow fires, treat
        # the 4 arrow logits as a 4-way softmax and CE against the firing
        # arrow index. Per-arrow BCE alone is satisfied by predicting the
        # marginal at every onset (no per-frame commitment required), which
        # is the local minimum that pins stream coherence. Softmax CE forces
        # the head to *pick* an arrow per onset frame.
        if self.commit_weight > 0.0:
            with torch.no_grad():
                arrow_hard = (arrow_soft >= 0.5).float()
                n_fire = arrow_hard.sum(dim=-1)
                single_mask = (n_fire == 1.0)
            if single_mask.any():
                target_idx = arrow_hard.argmax(dim=-1)
                commit_ce = F.cross_entropy(
                    arrow_logits[single_mask],
                    target_idx[single_mask],
                )
            else:
                commit_ce = arrow_logits.sum() * 0.0
            arrow_loss = arrow_loss + self.commit_weight * commit_ce

        # Anti-jack penalty: at pairs of co-active onset frames separated by
        # `jack_stride`, penalize the inner product of softmax(arrow_logits) —
        # i.e. the probability that the model picks the *same* arrow at both
        # frames. Anchored on ground-truth onsets (smoothed peak >= 0.5) and
        # gated by the model's predicted onset confidence so we only penalize
        # frames the model itself is committing to as events.
        #
        # When the labels actually contain a jack (L→L), the per-frame commit
        # CE above already pulls the model toward L at both frames, so the
        # jack penalty trades off against — but does not override — real
        # supervision. Empirically this only suppresses *unsupervised* jacks
        # that emerge from marginal-matching on the per-arrow head.
        if self.jack_weight > 0.0 and arrow_logits.size(1) > self.jack_stride:
            stride = self.jack_stride
            with torch.no_grad():
                soft_any = arrow_soft.max(dim=-1).values             # [B, T]
                gt_pair = (
                    (soft_any[:, :-stride] >= 0.5)
                    & (soft_any[:, stride:] >= 0.5)
                ).float()                                            # [B, T-s]
                pred_any = torch.sigmoid(arrow_logits).max(dim=-1).values
                pred_gate = pred_any[:, :-stride] * pred_any[:, stride:]
                gate = gt_pair * pred_gate
                gate_norm = gate.sum().clamp_min(1.0)
            prev_probs = F.softmax(arrow_logits[:, :-stride], dim=-1)
            cur_probs = F.softmax(arrow_logits[:, stride:], dim=-1)
            same_arrow_prob = (prev_probs * cur_probs).sum(dim=-1)   # [B, T-s]
            jack_loss = (gate * same_arrow_prob).sum() / gate_norm
            arrow_loss = arrow_loss + self.jack_weight * jack_loss

        # Diversity regularizer: KL(target_dist || pred_dist) on per-chunk
        # average arrow probabilities. Punishes "always L/R" collapse —
        # if the target chunk is balanced across 4 arrows but predictions
        # concentrate on two, the KL is high. Skipped on chunks where the
        # ground truth has no onsets at all (denominator is zero).
        with torch.no_grad():
            true_norm = arrow_soft.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            true_dist = arrow_soft.sum(dim=1, keepdim=True) / true_norm   # [B, 1, 4]
            target_total = arrow_soft.sum(dim=(1, 2))                     # [B]
        pred_probs = torch.sigmoid(arrow_logits)                          # [B, T, 4]
        pred_norm = pred_probs.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        pred_dist = pred_probs.sum(dim=1, keepdim=True) / pred_norm       # [B, 1, 4]
        eps = 1e-6
        kl_per_chunk = (
            true_dist * (true_dist.clamp_min(eps).log() - pred_dist.clamp_min(eps).log())
        ).sum(dim=-1).squeeze(1)                                          # [B]
        valid_chunks = (target_total > 0).float()
        if valid_chunks.sum() > 0:
            diversity_loss = (kl_per_chunk * valid_chunks).sum() / valid_chunks.sum()
        else:
            diversity_loss = torch.zeros((), device=arrow_logits.device)

        B, T, C = type_logits.shape
        type_flat = type_logits.reshape(-1, C)
        target_flat = type_target.reshape(-1)
        if self.type_focal_gamma > 0.0:
            # Per-frame CE so we can reweight by (1 - p_t)^gamma. The
            # static class weights are already baked into ce_per_frame.
            ce_per_frame = F.cross_entropy(
                type_flat, target_flat,
                weight=self.type_class_weights,
                ignore_index=-100,
                reduction='none',
            )
            valid = target_flat != -100
            if valid.any():
                with torch.no_grad():
                    log_p = F.log_softmax(type_flat[valid], dim=-1)
                    p_t = log_p.gather(
                        1, target_flat[valid].unsqueeze(-1)
                    ).squeeze(-1).exp()
                    focal_w = (1.0 - p_t).clamp_(min=0.0, max=1.0).pow(
                        self.type_focal_gamma
                    )
                # `cross_entropy(reduction='none', ignore_index=-100)` returns 0
                # at ignored positions, so the unmasked mean is correct.
                ce_valid = ce_per_frame[valid]
                type_loss = (focal_w * ce_valid).mean()
            else:
                type_loss = ce_per_frame.sum() * 0.0
        else:
            # Guard against fully-masked batches. MixUp sets every frame's
            # type target to -100, and `F.cross_entropy(reduction='mean',
            # ignore_index=-100)` returns NaN when zero frames are valid
            # (sum / count = 0 / 0). The focal branch above handles this
            # explicitly; we need the same guard here. Without it, the
            # mean-over-epoch type loss reports NaN and GradScaler silently
            # skips the affected batches.
            valid = target_flat != -100
            if valid.any():
                type_loss = F.cross_entropy(
                    type_flat, target_flat,
                    weight=self.type_class_weights,
                    ignore_index=-100,
                )
            else:
                type_loss = type_flat.sum() * 0.0

        # Duration loss: log-space smooth-L1 on a ±DURATION_WINDOW window
        # around each hold_start. Inflating the window from a single frame
        # to (2W+1) frames keeps targets accurate (the duration is constant
        # within a hold) while giving the head 5× more gradient per batch.
        # Targets are converted with `encode_duration` so the head learns
        # in log-seconds (well-conditioned dynamic range).
        hold_start_mask = (type_target == self.HOLD_START_CLASS)  # [B, T]
        if hold_start_mask.any():
            # Dilate the mask in time and propagate the duration target so
            # each window-frame gets the duration of the nearest hold_start.
            # We do this via a 1-D max-style dispatch implemented with
            # cumulative directional fills, keeping the op fully vectorized.
            B, T = hold_start_mask.shape
            window = self.DURATION_WINDOW
            # Build a dense [B, T] target tensor that is 0 outside windows
            # and equal to the hold_start's duration inside its ±W window.
            base = duration_target * hold_start_mask.float()
            window_mask = hold_start_mask.clone()
            window_target = base.clone()
            for offset in range(1, window + 1):
                # Right shift: copy hold_start contributions to t+offset
                window_mask[:, offset:] |= hold_start_mask[:, :-offset]
                window_target[:, offset:] = torch.where(
                    hold_start_mask[:, :-offset],
                    base[:, :-offset],
                    window_target[:, offset:],
                )
                # Left shift: copy to t-offset
                window_mask[:, :-offset] |= hold_start_mask[:, offset:]
                window_target[:, :-offset] = torch.where(
                    hold_start_mask[:, offset:],
                    base[:, offset:],
                    window_target[:, :-offset],
                )

            pred_log = duration_pred[:, :, 0][window_mask]                     # [N]
            true_log = StepChartModel.encode_duration(window_target[window_mask])
            dur_loss = F.smooth_l1_loss(pred_log, true_log)
        else:
            dur_loss = torch.tensor(0.0, device=arrow_logits.device)

        # Auxiliary beat-frame BCE. The beat target is dense and structured
        # (1-3% of frames depending on tempo), so a moderate pos_weight is
        # sufficient — no focal needed. Skipped if beat targets weren't
        # supplied (e.g., legacy datasets without `beats`).
        if (
            self.beat_weight > 0.0
            and beat_logits is not None
            and beat_soft is not None
        ):
            beat_loss = F.binary_cross_entropy_with_logits(
                beat_logits, beat_soft, pos_weight=self.beat_pos_weight,
            )
        else:
            beat_loss = torch.zeros((), device=arrow_logits.device)

        total = (arrow_loss
                 + self.type_weight * type_loss
                 + self.duration_weight * dur_loss
                 + self.beat_weight * beat_loss
                 + self.diversity_weight * diversity_loss)
        return (
            total,
            arrow_loss.detach(),
            type_loss.detach(),
            dur_loss.detach(),
            beat_loss.detach(),
            diversity_loss.detach(),
        )

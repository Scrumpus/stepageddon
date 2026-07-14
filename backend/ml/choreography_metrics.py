"""
Choreography metrics — quantify how *human-like* and *playable* a generated
arrow sequence is, measured against a reference human chart.

The rhythm side of the pipeline (onset / sustain / intensity) is already scored
in training. Nothing scored *which arrows* were chosen. This module fills that
gap: given a generated arrow sequence and the human `arrow_labels[T, 4]` for the
same song+difficulty, it reports how far the generated choreography is from the
human one.

Everything is computed on the canonical `[T, 4]` arrow-label representation
(columns L, D, U, R — the ordering used throughout the codebase). Generated
charts are rasterized back onto that grid via `steps_to_arrow_labels`, so the
style-rate stats reuse the *exact* definitions from `ml.style_profiles` and
therefore agree with the k-means clustering that drives inference.

Metric groups
-------------
- **Distribution divergence**: Jensen-Shannon + L1 between the panel-set uni-
  and bi-gram histograms of generated vs human. Captures "does the model pick
  the same *kinds* of steps and transitions humans do."
- **Style-rate deltas**: jump / jack / crossover / stream rates vs human,
  reusing `ml.style_profiles` so they match the profile clustering.
- **Playability**: foot-travel cost and same-foot double-step rate — the direct
  "is this awkward to play" signal.
- **Catalog-match rate**: fraction of steps covered by a named pattern from
  `ml.patterns` (streams, staircases, candles, spins, drills), vs the human rate.

`compare(generated_arrow_labels, human_arrow_labels)` returns a flat dict of
scalars suitable for logging next to the existing train/val metrics.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ml.dataset import FRAMES_PER_SECOND
from ml.patterns import (
    CANDLE_PATTERNS,
    DRILL_PATTERNS,
    SPIN_PATTERNS,
    STAIRCASE_PATTERNS,
    STREAM_PATTERNS,
)
from ml.style_profiles import (
    LEFT_REGION,
    RIGHT_REGION,
    _density_stats,
    _jack_and_crossover_rates,
)

# Spatial layout of the four panels on the dance pad, used for foot-travel.
# L/R on the horizontal axis, D/U on the vertical. Center is the neutral stance.
_PANEL_XY: Dict[int, Tuple[float, float]] = {
    0: (-1.0, 0.0),   # LEFT
    1: (0.0, -1.0),   # DOWN
    2: (0.0, 1.0),    # UP
    3: (1.0, 0.0),    # RIGHT
}


# ---------------------------------------------------------------------------
# Representation helpers
# ---------------------------------------------------------------------------
def steps_to_arrow_labels(
    steps: Sequence,
    n_frames: int,
    fps: float = FRAMES_PER_SECOND,
) -> np.ndarray:
    """Rasterize a generated chart's steps back onto the `[T, 4]` frame grid.

    `steps` is any sequence of objects exposing `.time` (seconds) and `.arrows`
    (iterable of `Direction`, whose `.value` is one of left/down/up/right) — i.e.
    the `Step` schema, but duck-typed so callers can pass lightweight stand-ins.
    Times are snapped to the nearest frame; collisions OR-together.
    """
    name_to_idx = {'left': 0, 'down': 1, 'up': 2, 'right': 3}
    out = np.zeros((int(n_frames), 4), dtype=np.uint8)
    if n_frames <= 0:
        return out
    for step in steps:
        frame = int(round(float(step.time) * fps))
        if frame < 0 or frame >= n_frames:
            continue
        for arrow in step.arrows:
            val = getattr(arrow, 'value', arrow)
            idx = name_to_idx.get(str(val).lower())
            if idx is not None:
                out[frame, idx] = 1
    return out


def panel_set_sequence(arrow_labels: np.ndarray) -> List[int]:
    """Onset frames → sequence of 4-bit panel-set codes (bit i set ⇒ arrow i).

    Empty frames are dropped, so consecutive entries are consecutive *steps*,
    which is what the n-gram / pattern metrics operate on. Codes are 1..15.
    """
    onset_mask = arrow_labels.any(axis=1)
    codes: List[int] = []
    for frame in np.flatnonzero(onset_mask):
        row = arrow_labels[frame]
        code = int(row[0]) | (int(row[1]) << 1) | (int(row[2]) << 2) | (int(row[3]) << 3)
        if code:
            codes.append(code)
    return codes


def _single_panel_sequence(codes: Sequence[int]) -> List[int]:
    """From panel-set codes, keep only single-arrow steps as their arrow index.

    Jumps (>1 bit set) break the run — they don't participate in the stream /
    staircase / drill catalog, which is defined over single-panel sequences.
    Returns a flat list where jumps are encoded as -1 separators.
    """
    out: List[int] = []
    for code in codes:
        if code and (code & (code - 1)) == 0:      # exactly one bit set
            out.append(int(code).bit_length() - 1)  # 1->0, 2->1, 4->2, 8->3
        else:
            out.append(-1)
    return out


# ---------------------------------------------------------------------------
# Distribution divergence
# ---------------------------------------------------------------------------
def _ngram_hist(codes: Sequence[int], n: int, vocab: int) -> np.ndarray:
    """Normalized histogram of length-`n` panel-set n-grams over a fixed vocab.

    Panel-set codes are 1..15, mapped to 0..14. An n-gram is the base-`vocab`
    number formed by its codes, giving a dense `vocab**n` histogram.
    """
    size = vocab ** n
    hist = np.zeros(size, dtype=np.float64)
    if len(codes) < n:
        return hist
    mapped = [c - 1 for c in codes]  # 1..15 -> 0..14
    for i in range(len(mapped) - n + 1):
        key = 0
        for j in range(n):
            key = key * vocab + mapped[i + j]
        hist[key] += 1.0
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two distributions."""
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    # Where m==0 both p and q are 0, so those terms vanish; guard the divide.
    m_safe = np.where(m > 0, m, 1.0)
    return 0.5 * _kl(p, m_safe) + 0.5 * _kl(q, m_safe)


def _distribution_divergence(
    gen_codes: Sequence[int],
    ref_codes: Sequence[int],
) -> Dict[str, float]:
    vocab = 15  # panel-set codes 1..15
    out: Dict[str, float] = {}
    for n, tag in ((1, 'unigram'), (2, 'bigram')):
        gp = _ngram_hist(gen_codes, n, vocab)
        rp = _ngram_hist(ref_codes, n, vocab)
        out[f'{tag}_js'] = _js_divergence(gp, rp)
        out[f'{tag}_l1'] = float(np.abs(gp - rp).sum())
    return out


# ---------------------------------------------------------------------------
# Playability
# ---------------------------------------------------------------------------
def _foot_travel(codes: Sequence[int]) -> float:
    """Mean euclidean distance between consecutive step centroids on the pad.

    A step's position is the centroid of its active panels; higher mean travel
    means more physical movement (a rough proxy for how frantic a chart feels).
    """
    pts: List[Tuple[float, float]] = []
    for code in codes:
        active = [i for i in range(4) if code & (1 << i)]
        if not active:
            continue
        xs = [_PANEL_XY[i][0] for i in active]
        ys = [_PANEL_XY[i][1] for i in active]
        pts.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    if len(pts) < 2:
        return 0.0
    dists = [
        ((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
        for i in range(1, len(pts))
    ]
    return float(np.mean(dists))


def _same_foot_double_rate(codes: Sequence[int]) -> float:
    """Fraction of consecutive single-panel steps that force a same-foot double.

    Using the two-region home model shared with `ml.style_profiles`
    (LEFT_REGION = {L, D}, RIGHT_REGION = {U, R}), two consecutive single-panel
    steps on *different* panels within the *same* region require the same foot to
    move twice — a genuine double-step, harder to play than a region-alternating
    move. Same-panel repeats (jacks) are excluded; they're scored separately.
    """
    singles = _single_panel_sequence(codes)
    pairs = 0
    doubles = 0
    prev = None
    for a in singles:
        if a < 0:            # jump breaks the chain
            prev = None
            continue
        if prev is not None:
            pairs += 1
            if a != prev:
                same_region = (
                    (prev in LEFT_REGION and a in LEFT_REGION)
                    or (prev in RIGHT_REGION and a in RIGHT_REGION)
                )
                if same_region:
                    doubles += 1
        prev = a
    return doubles / pairs if pairs else 0.0


# ---------------------------------------------------------------------------
# Catalog-match rate
# ---------------------------------------------------------------------------
def _catalog_windows() -> List[Tuple[int, ...]]:
    """All named single-panel patterns from ml.patterns as index tuples."""
    windows: List[Tuple[int, ...]] = []
    windows.extend(tuple(p) for p in STREAM_PATTERNS)
    windows.extend(tuple(p) for p in STAIRCASE_PATTERNS.values())
    windows.extend(tuple(p) for p in CANDLE_PATTERNS)
    windows.extend(tuple(p) for p in SPIN_PATTERNS)
    # Drills are 2-note frames repeated; expand to a 4-note repeat so they match
    # the same sliding-window machinery as the other catalogs.
    for a, b in DRILL_PATTERNS:
        windows.append((a, b, a, b))
    return windows


_CATALOG = _catalog_windows()
_CATALOG_BY_LEN: Dict[int, List[Tuple[int, ...]]] = {}
for _w in _CATALOG:
    _CATALOG_BY_LEN.setdefault(len(_w), []).append(_w)


def _catalog_match_rate(codes: Sequence[int]) -> float:
    """Fraction of single-panel steps covered by at least one catalog pattern.

    Slides every catalog window over the single-panel sequence; any step inside
    a matched window is marked covered. Jumps (-1) never match.
    """
    singles = _single_panel_sequence(codes)
    n = len(singles)
    if n == 0:
        return 0.0
    covered = np.zeros(n, dtype=bool)
    for length, patterns in _CATALOG_BY_LEN.items():
        if length > n:
            continue
        pattern_set = set(patterns)
        for i in range(n - length + 1):
            window = tuple(singles[i:i + length])
            if -1 in window:
                continue
            if window in pattern_set:
                covered[i:i + length] = True
    return float(covered.mean())


# ---------------------------------------------------------------------------
# Per-sequence stats (reuses ml.style_profiles definitions)
# ---------------------------------------------------------------------------
def sequence_stats(arrow_labels: np.ndarray) -> Dict[str, float]:
    """Style + playability stats for a single `[T, 4]` arrow-label array."""
    codes = panel_set_sequence(arrow_labels)
    n_steps = len(codes)
    jumps = sum(1 for c in codes if c and (c & (c - 1)) != 0)
    jump_rate = jumps / n_steps if n_steps else 0.0

    jack_rate, crossover_rate = _jack_and_crossover_rates(arrow_labels)
    mean_density, density_var, stream_density = _density_stats(arrow_labels)

    return {
        'n_steps': float(n_steps),
        'jump_rate': jump_rate,
        'jack_rate': jack_rate,
        'crossover_rate': crossover_rate,
        'stream_density': stream_density,
        'mean_density': mean_density,
        'density_var': density_var,
        'foot_travel': _foot_travel(codes),
        'same_foot_double_rate': _same_foot_double_rate(codes),
        'catalog_match_rate': _catalog_match_rate(codes),
    }


# Style/playability fields for which we report a generated-vs-human delta.
_DELTA_FIELDS = (
    'jump_rate',
    'jack_rate',
    'crossover_rate',
    'stream_density',
    'mean_density',
    'foot_travel',
    'same_foot_double_rate',
    'catalog_match_rate',
)


def compare(
    generated_arrow_labels: np.ndarray,
    human_arrow_labels: np.ndarray,
) -> Dict[str, float]:
    """Full choreography comparison of a generated chart against a human one.

    Both inputs are `[T, 4]` uint8 arrays on the same frame grid. Returns a flat
    dict: distribution divergences (`*_js`, `*_l1`), signed per-stat deltas
    (`<stat>_delta` = generated − human), and the raw generated/human stats
    (`gen_<stat>`, `ref_<stat>`) for context.
    """
    gen_codes = panel_set_sequence(generated_arrow_labels)
    ref_codes = panel_set_sequence(human_arrow_labels)

    metrics: Dict[str, float] = {}
    metrics.update(_distribution_divergence(gen_codes, ref_codes))

    gen_stats = sequence_stats(generated_arrow_labels)
    ref_stats = sequence_stats(human_arrow_labels)
    for field in _DELTA_FIELDS:
        metrics[f'{field}_delta'] = gen_stats[field] - ref_stats[field]
        metrics[f'gen_{field}'] = gen_stats[field]
        metrics[f'ref_{field}'] = ref_stats[field]

    # A single scalar summarizing choreography distance, for ranking/logging:
    # bigram JS (distribution shape) + normalized playability drift.
    metrics['choreography_distance'] = (
        metrics['bigram_js']
        + abs(metrics['same_foot_double_rate_delta'])
        + abs(metrics['catalog_match_rate_delta'])
    )
    return metrics

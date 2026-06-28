"""
Style profile clustering for the v7 hybrid step chart pipeline.

Each chart in the manifest is reduced to an 8-dim stat vector summarizing its
rhythmic / structural style. Vectors are standardized and k-means-clustered
(default k=6); each cluster is auto-named by its dominant trait. The result
is written to `style_profiles.json` and consumed at inference time to
parameterize the algorithmic post-processing (jump threshold, hold rate,
crossover/jack/stream preferences, etc.).

Run:
    python -m ml.style_profiles --data-dir <path>/training_data --k 6

Output schema:
    {
        "k": 6,
        "stat_fields": [...],
        "standardize_mean": [...8...],
        "standardize_std":  [...8...],
        "profiles": [
            {
                "name": "Stream-Heavy",
                "centroid":   [...8 standardized...],
                "raw_means":  {field: float, ...},
                "member_count": 412
            },
            ...
        ],
        "assignment": {"<entry_idx>": <cluster_id>, ...}
    }
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ml.dataset import (
    FRAMES_PER_SECOND,
    _build_sustain_target,
    load_manifest,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


STAT_FIELDS: Tuple[str, ...] = (
    'jump_rate',          # P(onset frame is a jump)
    'hold_rate',          # P(onset frame is a hold_start)
    'avg_hold_beats',     # mean hold duration, in beats
    'jack_rate',          # fraction of consecutive same-arrow onsets
    'crossover_rate',     # fraction of L→R-style crosses (foot logic; see below)
    'stream_density',     # fraction of 1-second windows that are dense (>=4 onsets)
    'mean_density',       # steps/sec over the song
    'density_var',        # variance of per-second step density
)


# Foot-region grouping for the crossover heuristic. Mirrors the
# FootStateArrowAssigner in inference.py: L/D belong to the left foot, U/R to
# the right. Two consecutive onsets that switch *foot region* in the
# anti-natural direction count as a crossover.
LEFT_REGION = {0, 1}   # L, D
RIGHT_REGION = {2, 3}  # U, R


def _consecutive_onset_pairs(arrow_labels: np.ndarray) -> List[Tuple[int, int]]:
    """Return (prev_arrow_idx, cur_arrow_idx) for adjacent onset frames where
    each side fires exactly one arrow. Jumps are skipped — they don't have a
    well-defined "single arrow" that pairs cleanly into a jack/crossover.
    """
    onset_mask = arrow_labels.any(axis=1)
    onset_frames = np.flatnonzero(onset_mask)
    pairs: List[Tuple[int, int]] = []
    prev_arrow = None
    for f in onset_frames:
        active = np.flatnonzero(arrow_labels[f])
        if active.size != 1:
            prev_arrow = None  # a jump or empty frame breaks the chain
            continue
        cur = int(active[0])
        if prev_arrow is not None:
            pairs.append((prev_arrow, cur))
        prev_arrow = cur
    return pairs


def _jack_and_crossover_rates(arrow_labels: np.ndarray) -> Tuple[float, float]:
    """Compute (jack_rate, crossover_rate) over consecutive single-arrow onsets.

    jack_rate: fraction of pairs with prev == cur (same arrow twice in a row).
    crossover_rate: fraction of pairs where the current arrow is in the *other*
        foot's region from the previous arrow, but the side-foot alternation
        *would have* required the same region — i.e. anti-natural crosses.
        Approximated here as "any L/D ↔ U/R region switch", since the strict
        foot-state needs FootStateArrowAssigner's tracking.
    """
    pairs = _consecutive_onset_pairs(arrow_labels)
    if not pairs:
        return 0.0, 0.0
    n = len(pairs)
    jacks = sum(1 for a, b in pairs if a == b)
    crosses = 0
    for a, b in pairs:
        a_left = a in LEFT_REGION
        b_left = b in LEFT_REGION
        if a_left != b_left:
            # Region switches that are not the natural same-foot move (which
            # stays inside one region). Counted as crossover if the absolute
            # arrow distance is at least 2 — picking up the inner panel of
            # the other region (e.g., L→U) rather than the symmetric far one.
            if abs(a - b) >= 2:
                crosses += 1
    return jacks / n, crosses / n


def _density_stats(
    labels: np.ndarray,
    fps: float = FRAMES_PER_SECOND,
) -> Tuple[float, float, float]:
    """(mean_density, density_var, stream_density)

    mean_density: steps/sec over the whole song.
    density_var:  variance of per-second step counts (in steps²/sec²).
    stream_density: fraction of 1-second windows containing ≥ 4 onsets.
    """
    T = labels.shape[0]
    if T == 0:
        return 0.0, 0.0, 0.0
    onset = (labels >= 1).astype(np.float32)
    n_onsets = int(onset.sum())
    duration_s = T / fps
    mean_density = n_onsets / max(duration_s, 1.0)

    win = max(1, int(round(fps)))
    n_windows = max(1, T // win)
    counts = np.array([
        int(onset[i * win:(i + 1) * win].sum()) for i in range(n_windows)
    ], dtype=np.float32)
    if counts.size > 1:
        density_var = float(counts.var())
    else:
        density_var = 0.0
    if counts.size > 0:
        stream_density = float((counts >= 4).mean())
    else:
        stream_density = 0.0
    return mean_density, density_var, stream_density


def per_chart_stats(entry: dict, data_dir: Path) -> Dict[str, float]:
    """Compute the 8 style-profile stats for a single chart entry."""
    archive = np.load(data_dir / entry['filename'])
    labels = archive[entry['labels_key']]                 # [T] uint8
    durations = archive[entry['durations_key']]           # [T] float32 seconds
    arrow_labels = archive[entry['arrow_labels_key']]     # [T, 4] uint8

    onset_mask = labels >= 1
    n_onsets = int(onset_mask.sum())
    if n_onsets > 0:
        jump_rate = float((labels == 2).sum() / n_onsets)
        hold_rate = float((labels == 3).sum() / n_onsets)
    else:
        jump_rate = 0.0
        hold_rate = 0.0

    # Average hold duration in beats. Convert seconds → beats with the song
    # tempo (default 120 BPM if missing) so the stat is BPM-invariant.
    tempo = float(entry.get('tempo', 120.0)) or 120.0
    hs_mask = labels == 3
    if hs_mask.any():
        hold_seconds = durations[hs_mask]
        hold_seconds = hold_seconds[hold_seconds > 0]
        if hold_seconds.size > 0:
            avg_hold_beats = float(hold_seconds.mean()) * (tempo / 60.0)
        else:
            avg_hold_beats = 0.0
    else:
        avg_hold_beats = 0.0

    jack_rate, crossover_rate = _jack_and_crossover_rates(arrow_labels)
    mean_density, density_var, stream_density = _density_stats(labels)

    return {
        'jump_rate': jump_rate,
        'hold_rate': hold_rate,
        'avg_hold_beats': avg_hold_beats,
        'jack_rate': jack_rate,
        'crossover_rate': crossover_rate,
        'stream_density': stream_density,
        'mean_density': mean_density,
        'density_var': density_var,
    }


def _autoname(raw_means: Dict[str, float]) -> str:
    """Pick a human-readable name based on the dominant trait of a centroid.

    The check order is intentional: rare but defining traits (jump, hold,
    technical) are checked before generic density/flow ones so a profile with
    e.g. high hold_rate AND high stream_density gets named "Hold-Heavy" rather
    than "Stream-Heavy".
    """
    j = raw_means['jump_rate']
    h = raw_means['hold_rate']
    s = raw_means['stream_density']
    md = raw_means['mean_density']
    jck = raw_means['jack_rate']
    cx = raw_means['crossover_rate']

    # Thresholds calibrated against rough corpus averages — these are the
    # rules of thumb for "this profile is unusually X." They don't need to be
    # exact: clusters that fall through pick up the generic "Flowing" name.
    if j >= 0.12:
        return 'Jump-Heavy'
    if h >= 0.06:
        return 'Hold-Heavy'
    if (jck + cx) >= 0.18:
        return 'Technical'
    if s >= 0.20 and md >= 3.0:
        return 'Stream-Heavy'
    if md >= 4.0:
        return 'Dense'
    return 'Flowing'


def _disambiguate_names(names: List[str]) -> List[str]:
    """Append " 2", " 3", ... to repeated names so each profile is uniquely
    addressable. Order is preserved so the first occurrence keeps the bare name.
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        if seen[n] == 1:
            out.append(n)
        else:
            out.append(f"{n} {seen[n]}")
    return out


def cluster_profiles(
    stats: List[Dict[str, float]],
    k: int = 6,
    seed: int = 42,
):
    """Standardize the per-chart stat vectors and run k-means.

    Returns:
        profiles: list of dicts, one per cluster, in cluster_id order.
            Each dict has {name, centroid (standardized), raw_means, member_count}.
        labels: int array [n_charts] of cluster assignments.
        mean: [8] standardize mean.
        std:  [8] standardize std.
    """
    if not stats:
        raise ValueError("No chart stats supplied to cluster_profiles().")
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            "sklearn is required for style profile clustering. "
            "Install with: pip install scikit-learn"
        ) from e

    X = np.array(
        [[s[f] for f in STAT_FIELDS] for s in stats],
        dtype=np.float64,
    )
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    Xz = (X - mean) / std

    k = min(int(k), Xz.shape[0])
    km = KMeans(n_clusters=k, n_init=10, random_state=int(seed))
    labels = km.fit_predict(Xz)
    centroids = km.cluster_centers_

    profiles = []
    raw_names: List[str] = []
    for c in range(k):
        members = X[labels == c]
        if members.size == 0:
            raw_means = {f: float(mean[i]) for i, f in enumerate(STAT_FIELDS)}
        else:
            raw_means = {
                f: float(members[:, i].mean())
                for i, f in enumerate(STAT_FIELDS)
            }
        raw_names.append(_autoname(raw_means))
        profiles.append({
            'centroid': centroids[c].tolist(),
            'raw_means': raw_means,
            'member_count': int((labels == c).sum()),
        })

    for prof, name in zip(profiles, _disambiguate_names(raw_names)):
        prof['name'] = name

    # Reorder dicts so 'name' shows up first in the JSON for readability.
    profiles = [
        {'name': p['name'], 'centroid': p['centroid'],
         'raw_means': p['raw_means'], 'member_count': p['member_count']}
        for p in profiles
    ]
    return profiles, labels.astype(int), mean, std


def main():
    parser = argparse.ArgumentParser(description='Cluster style profiles from prepared data.')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing manifest.json + per-song .npz files.')
    parser.add_argument('--k', type=int, default=12,
                        help='Number of clusters (default: 12).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Output path. Defaults to <data-dir>/style_profiles.json.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / 'manifest.json'
    manifest = load_manifest(str(manifest_path))
    entries = manifest['entries']
    logger.info(f"Computing per-chart style stats over {len(entries)} entries...")

    stats: List[Dict[str, float]] = []
    valid_indices: List[int] = []
    for i, entry in enumerate(entries):
        try:
            s = per_chart_stats(entry, data_dir)
        except Exception as e:
            logger.warning(f"Skipping entry {i} ({entry.get('song_title')}): {e}")
            continue
        stats.append(s)
        valid_indices.append(i)
        if (len(stats) % 200) == 0:
            logger.info(f"  {len(stats)} / {len(entries)} processed")

    logger.info(f"Clustering {len(stats)} chart stat vectors into k={args.k}...")
    profiles, labels, mean, std = cluster_profiles(stats, k=args.k, seed=args.seed)

    assignment: Dict[str, int] = {
        str(orig_idx): int(cluster_id)
        for orig_idx, cluster_id in zip(valid_indices, labels.tolist())
    }

    payload = {
        'k': len(profiles),
        'stat_fields': list(STAT_FIELDS),
        'standardize_mean': mean.tolist(),
        'standardize_std': std.tolist(),
        'profiles': profiles,
        'assignment': assignment,
    }

    output_path = Path(args.output) if args.output else (data_dir / 'style_profiles.json')
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote {output_path}")
    for c, p in enumerate(profiles):
        rm = p['raw_means']
        logger.info(
            f"  cluster {c} → {p['name']:>14s}  n={p['member_count']:5d}  "
            f"jump={rm['jump_rate']:.3f} hold={rm['hold_rate']:.3f} "
            f"jck={rm['jack_rate']:.3f} cx={rm['crossover_rate']:.3f} "
            f"stream={rm['stream_density']:.3f} dens={rm['mean_density']:.2f}"
        )


if __name__ == '__main__':
    main()

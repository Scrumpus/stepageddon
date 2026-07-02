"""
Baseline choreography evaluation — how human-like is the *current* (RNG) arrow
assigner, holding rhythm fixed?

For each sampled chart we take the human chart's own onset timing and arity
(tap / jump / hold, straight from the prepared `labels`/`durations`) and run only
`FootStateArrowAssigner` to choose *which* panels light up — exactly the arrow
loop from `MLChartGenerator._events_to_steps` (inference.py:1522-1602). The
assigner's arrows are rasterized back to `[T, 4]` and compared to the human
`arrow_labels` via `ml.choreography_metrics.compare`.

Because onset timing/arity are identical to the human chart, every reported
delta is attributable to *arrow selection*, isolating the thing this project is
trying to improve. This establishes the baseline that Phase 2's learned
step-selection must beat.

Run:
    python -m ml.eval_choreography --data-dir ml/data/training_data --n 100
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from ml.choreography_metrics import compare, steps_to_arrow_labels
from ml.dataset import FRAMES_PER_SECOND
from ml.inference import DEFAULT_STYLE_KNOBS, FootStateArrowAssigner
from src.charts.schemas import Direction, Step, StepType
from src.generation.constants import get_difficulty_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_IDX_TO_DIR = [Direction.LEFT, Direction.DOWN, Direction.UP, Direction.RIGHT]


def _energy_brightness(feats: np.ndarray, n_mels: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Derive per-frame energy + brightness proxies from the mel sub-block.

    energy    = mean mel magnitude per frame, min-max normalized to [0, 1].
    brightness= mel spectral centroid (energy-weighted bin index), normalized.
    Mirrors the *kind* of scalar curves the real pipeline feeds the assigner so
    the baseline reflects realistic assigner behavior, not a constant 0.5.
    """
    mel = feats[:, :n_mels].astype(np.float32)
    energy = mel.mean(axis=1)
    e_min, e_max = float(energy.min()), float(energy.max())
    energy = (energy - e_min) / (e_max - e_min) if e_max > e_min else np.full_like(energy, 0.5)

    bins = np.arange(n_mels, dtype=np.float32)
    denom = mel.sum(axis=1)
    centroid = np.where(denom > 0, (mel * bins).sum(axis=1) / np.maximum(denom, 1e-6), n_mels / 2)
    brightness = centroid / max(1.0, n_mels - 1)
    return energy.astype(np.float32), np.clip(brightness, 0.0, 1.0).astype(np.float32)


def assigner_steps_for_chart(
    labels: np.ndarray,
    durations: np.ndarray,
    arrow_labels: np.ndarray,
    difficulty_name: str,
    energy_curve: np.ndarray,
    brightness_curve: np.ndarray,
    seed: int,
) -> List[Step]:
    """Run FootStateArrowAssigner over the human onsets, returning its Steps.

    This is the emission loop from MLChartGenerator._events_to_steps, driven by
    the human `labels` (1=tap, 2=jump, 3=hold_start) instead of model peaks.
    """
    diff_config = get_difficulty_config(difficulty_name.lower())
    knobs = DEFAULT_STYLE_KNOBS
    assigner = FootStateArrowAssigner(
        seed=seed,
        allowed_patterns=diff_config.allowed_patterns,
        jack_rate=knobs['jack_rate'],
        crossover_rate=knobs['crossover_rate'],
        stream_preference=knobs['stream_preference'],
        candle_rate=knobs['candle_rate'],
        spin_rate=knobs['spin_rate'],
    )
    max_stream_length = max(1, getattr(diff_config, 'max_stream_length', 0))
    T = labels.shape[0]

    onset_frames = np.flatnonzero(labels >= 1)
    steps: List[Step] = []
    for i, frame in enumerate(onset_frames):
        t = round(float(frame) / FRAMES_PER_SECOND, 3)
        energy = float(energy_curve[min(frame, len(energy_curve) - 1)])
        brightness = float(brightness_curve[min(frame, len(brightness_curve) - 1)])
        num_arrows = int(arrow_labels[frame].sum())
        target_arity = 2 if num_arrows >= 2 else 1
        is_hold = int(labels[frame]) == 3

        if getattr(diff_config, 'max_stream_length', 0) > 0:
            assigner.maybe_start_stream(
                energy=energy,
                remaining_events=len(onset_frames) - i,
                max_stream_length=max_stream_length,
                time=t,
            )

        if is_hold:
            hold_duration = float(durations[frame]) or diff_config.min_hold_duration
            if target_arity >= 2:
                arrows = assigner.assign_jump(t, energy, brightness)
                for a in arrows:
                    assigner.active_holds[a] = t + hold_duration
            else:
                arrows = [assigner.start_hold(t, hold_duration, energy, brightness)]
            steps.append(Step(
                time=t, arrows=arrows, step_type=StepType.HOLD,
                hold_duration=round(hold_duration, 3),
            ))
        else:
            if target_arity >= 2:
                arrows = assigner.assign_jump(t, energy, brightness)
            else:
                arrows = [assigner.assign_single(t, energy, brightness)]
            held_now = assigner._held_arrows(t)
            if any(a in held_now for a in arrows):
                continue
            steps.append(Step(time=t, arrows=arrows, step_type=StepType.TAP))
    return steps


def evaluate(data_dir: Path, n: int, seed: int = 0) -> Dict[str, float]:
    # Read the manifest directly (not via dataset.load_manifest, which pins an
    # older format_version); only the per-entry keys below are needed.
    with open(data_dir / 'manifest.json') as f:
        manifest = json.load(f)
    entries = manifest['entries']

    # Deterministic sample spread across the corpus (no Date/random in scope).
    stride = max(1, len(entries) // n)
    sampled = entries[::stride][:n]
    logger.info(f"Evaluating {len(sampled)} charts (of {len(entries)}) with the RNG assigner...")

    per_metric: Dict[str, List[float]] = {}
    for idx, entry in enumerate(sampled):
        try:
            archive = np.load(data_dir / entry['filename'])
            feats = archive['feats']
            labels = archive[entry['labels_key']]
            durations = archive[entry['durations_key']]
            human_arrows = archive[entry['arrow_labels_key']]
        except Exception as e:
            logger.warning(f"Skipping {entry.get('filename')}: {e}")
            continue

        if int((labels >= 1).sum()) < 4:
            continue  # too few notes to say anything meaningful

        energy_curve, brightness_curve = _energy_brightness(feats)
        # Seed per-chart so runs are reproducible but charts aren't correlated.
        steps = assigner_steps_for_chart(
            labels, durations, human_arrows, entry['difficulty'],
            energy_curve, brightness_curve, seed=seed + idx,
        )
        gen_arrows = steps_to_arrow_labels(steps, n_frames=labels.shape[0])
        metrics = compare(gen_arrows, human_arrows.astype(np.uint8))
        for k, v in metrics.items():
            per_metric.setdefault(k, []).append(v)

    summary = {k: float(np.mean(v)) for k, v in per_metric.items() if v}
    return summary


def main():
    parser = argparse.ArgumentParser(description='Baseline choreography eval of the RNG assigner.')
    parser.add_argument('--data-dir', type=str, default='ml/data/training_data')
    parser.add_argument('--n', type=int, default=100, help='Number of charts to sample.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    summary = evaluate(Path(args.data_dir), args.n, args.seed)

    logger.info("=== Baseline choreography metrics (RNG assigner vs human) ===")
    # Headline divergences first, then style/playability gen-vs-ref pairs.
    order = ['choreography_distance', 'unigram_js', 'bigram_js', 'unigram_l1', 'bigram_l1']
    for k in order:
        if k in summary:
            logger.info(f"  {k:>28s} = {summary[k]:.4f}")
    for field in ('jump_rate', 'jack_rate', 'crossover_rate', 'stream_density',
                  'mean_density', 'foot_travel', 'same_foot_double_rate', 'catalog_match_rate'):
        g, r = summary.get(f'gen_{field}'), summary.get(f'ref_{field}')
        if g is not None and r is not None:
            logger.info(f"  {field:>28s} = gen {g:.4f} | human {r:.4f} | Δ {g - r:+.4f}")


if __name__ == '__main__':
    main()

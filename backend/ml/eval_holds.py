"""Offline evaluation for algorithmic hold derivation.

Holds are no longer predicted by the model — :meth:`MLChartGenerator._derive_holds`
promotes taps to holds from the harmonic-sustain envelope, bounded by what the
free foot can cover. Judging that by regenerating charts and listening is slow,
so this runs the derivation directly against prepared training data and scores it
against the human-charted holds.

The onset timeline comes from the **ground-truth** labels, not the model, so the
numbers isolate the derivation logic from onset-head error. Tune the knobs it
reports on: ``HOLD_DECAY_FLOOR``, ``HOLD_TAIL_LEAD_BEATS``, ``HOLD_BEAT_CELLS``,
and each difficulty's ``max_notes_under_hold``.

Usage::

    cd backend && source .venv/bin/activate
    python -m ml.eval_holds                          # 200 charts, all difficulties
    python -m ml.eval_holds --limit 500 --difficulty hard
    python -m ml.eval_holds --decay-floor 0.05       # sweep a knob

Caveat: the harmonic envelope is reconstructed from the stored mel spectrogram
(see :func:`harmonic_curve_from_feats`), not from raw audio via STFT+HPSS the way
inference does. The two correlate but are not identical, so treat absolute
numbers as a tuning signal and relative changes between runs as the real output.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml.inference import MLChartGenerator
from ml.prepare_data import FRAMES_PER_SECOND, N_MELS, DB_MIN, DB_MAX
from src.generation.constants import get_difficulty_config

logger = logging.getLogger(__name__)

DIFFICULTY_BY_ID = {
    0: 'beginner', 1: 'easy', 2: 'medium', 3: 'hard', 4: 'challenge',
}

# A derived hold head counts as matching a charted one within this window.
HEAD_TOLERANCE_S = 0.060


def harmonic_curve_from_feats(feats: np.ndarray) -> np.ndarray:
    """Approximate the inference-time harmonic-energy envelope from stored mels.

    ``prepare_data`` writes mel power in dB, clipped to [DB_MIN, DB_MAX] and
    scaled to [0, 1]; this inverts that, runs the same HPSS decomposition
    inference uses, and takes the harmonic part's RMS across bands.
    """
    import librosa

    mel_scaled = np.asarray(feats[:, :N_MELS], dtype=np.float32)
    mel_db = mel_scaled * (DB_MAX - DB_MIN) + DB_MIN
    mel_power = np.power(10.0, mel_db / 10.0).T  # [n_mels, T]
    harm = librosa.decompose.hpss(mel_power)[0]
    curve = np.sqrt(np.mean(harm ** 2, axis=0)).astype(np.float32)
    peak = float(curve.max())
    return curve / peak if peak > 0 else curve


def ground_truth(labels: np.ndarray, durations: np.ndarray) -> Tuple[List[dict], List[dict]]:
    """Return (onset timeline, charted holds) from one chart's frame labels.

    Holds are taken from ``durations > 0`` rather than ``labels == 3``, because
    ``prepare_data`` overwrites a hold's class with LABEL_JUMP whenever the frame
    carries two arrows — so ``labels == 3`` silently omits every jumphold while
    the duration is still recorded.
    """
    onset_frames = np.flatnonzero(labels >= 1)
    timeline = [
        {
            'frame': int(f),
            'time': float(f) / FRAMES_PER_SECOND,
            'type': 'tap',
            'confidence': 1.0,
            'num_arrows': 2 if labels[f] == 2 else 1,
        }
        for f in onset_frames
    ]
    holds = [
        {'time': float(f) / FRAMES_PER_SECOND, 'duration': float(durations[f])}
        for f in np.flatnonzero(durations > 0)
    ]
    return timeline, holds


def evaluate_chart(
    gen: MLChartGenerator,
    feats: np.ndarray,
    labels: np.ndarray,
    durations: np.ndarray,
    tempo: float,
    difficulty: str,
    style_knobs: Dict[str, float],
) -> Optional[dict]:
    """Derive holds over a chart's ground-truth timeline and score them."""
    timeline, gt_holds = ground_truth(labels, durations)
    if len(timeline) < 8:
        return None

    n_frames = int(feats.shape[0])
    curve = harmonic_curve_from_feats(feats)
    beat_times = np.arange(0.0, n_frames / FRAMES_PER_SECOND, 60.0 / max(tempo, 1e-6))

    gen._derive_holds(
        timeline, curve, tempo, beat_times, n_frames,
        get_difficulty_config(difficulty), style_knobs,
    )
    derived = [e for e in timeline if e['type'] == 'hold']

    # Greedy nearest-match within tolerance; each charted hold is claimed once.
    beat = 60.0 / max(tempo, 1e-6)
    unclaimed = sorted(gt_holds, key=lambda h: h['time'])
    claimed = [False] * len(unclaimed)
    matched, duration_errors = 0, []
    for d in sorted(derived, key=lambda e: e['time']):
        best_j, best_dt = None, HEAD_TOLERANCE_S
        for j, g in enumerate(unclaimed):
            if claimed[j]:
                continue
            dt = abs(g['time'] - d['time'])
            if dt <= best_dt:
                best_j, best_dt = j, dt
        if best_j is not None:
            claimed[best_j] = True
            matched += 1
            duration_errors.append(
                abs(d['hold_duration'] - unclaimed[best_j]['duration']) / beat
            )

    return {
        'n_notes': len(timeline),
        'n_derived': len(derived),
        'n_gt': len(gt_holds),
        'matched': matched,
        'duration_errors': duration_errors,
    }


def aggregate(rows: List[dict]) -> dict:
    n_derived = sum(r['n_derived'] for r in rows)
    n_gt = sum(r['n_gt'] for r in rows)
    matched = sum(r['matched'] for r in rows)
    n_notes = sum(r['n_notes'] for r in rows)
    errs = [e for r in rows for e in r['duration_errors']]
    return {
        'charts': len(rows),
        'notes': n_notes,
        'derived': n_derived,
        'charted': n_gt,
        'precision': matched / n_derived if n_derived else 0.0,
        'recall': matched / n_gt if n_gt else 0.0,
        'derived_rate': n_derived / n_notes if n_notes else 0.0,
        'charted_rate': n_gt / n_notes if n_notes else 0.0,
        'median_dur_err_beats': float(np.median(errs)) if errs else float('nan'),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', default='ml/data/training_data')
    ap.add_argument('--limit', type=int, default=200, help='charts to evaluate')
    ap.add_argument('--difficulty', default=None, help='only this difficulty')
    ap.add_argument('--hold-rate', type=float, default=0.04,
                    help='style-profile hold_rate knob')
    ap.add_argument('--avg-hold-beats', type=float, default=2.0,
                    help='style-profile avg_hold_beats knob')
    ap.add_argument('--decay-floor', type=float, default=None,
                    help='override MLChartGenerator.HOLD_DECAY_FLOOR')
    ap.add_argument('--tail-lead-beats', type=float, default=None,
                    help='override MLChartGenerator.HOLD_TAIL_LEAD_BEATS')
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    # _derive_holds logs one line per chart; at eval scale that buries the table.
    logging.getLogger('ml.inference').setLevel(logging.WARNING)

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    entries = manifest['entries']
    if args.difficulty:
        entries = [e for e in entries
                   if DIFFICULTY_BY_ID.get(e['difficulty_id']) == args.difficulty]

    # Deterministic spread across the corpus rather than the first N (which are
    # alphabetical by title, so a single style would dominate).
    if len(entries) > args.limit:
        idx = np.linspace(0, len(entries) - 1, args.limit).astype(int)
        entries = [entries[i] for i in idx]

    gen = MLChartGenerator.__new__(MLChartGenerator)  # post-processing only
    if args.decay_floor is not None:
        gen.HOLD_DECAY_FLOOR = args.decay_floor
    if args.tail_lead_beats is not None:
        gen.HOLD_TAIL_LEAD_BEATS = args.tail_lead_beats
    knobs = {'hold_rate': args.hold_rate, 'avg_hold_beats': args.avg_hold_beats}

    by_difficulty: Dict[str, List[dict]] = defaultdict(list)
    npz_cache: Dict[str, np.lib.npyio.NpzFile] = {}
    skipped = 0
    for entry in entries:
        difficulty = DIFFICULTY_BY_ID.get(entry['difficulty_id'])
        if difficulty is None:
            skipped += 1
            continue
        path = data_dir / entry['filename']
        if not path.exists():
            skipped += 1
            continue
        if path.name not in npz_cache:
            npz_cache.clear()  # entries are grouped per song; keep one open
            npz_cache[path.name] = np.load(path)
        archive = npz_cache[path.name]
        try:
            row = evaluate_chart(
                gen,
                archive['feats'].astype(np.float32),
                archive[entry['labels_key']],
                archive[entry['durations_key']],
                float(entry.get('tempo') or 120.0),
                difficulty,
                knobs,
            )
        except KeyError:
            skipped += 1
            continue
        if row is not None:
            by_difficulty[difficulty].append(row)

    print(f"\nHold derivation vs. charted holds "
          f"(head tolerance ±{HEAD_TOLERANCE_S * 1000:.0f}ms, "
          f"hold_rate={args.hold_rate}, avg_hold_beats={args.avg_hold_beats}, "
          f"decay_floor={gen.HOLD_DECAY_FLOOR}, "
          f"tail_lead={gen.HOLD_TAIL_LEAD_BEATS})\n")
    header = (f"{'difficulty':<11}{'charts':>7}{'derived':>9}{'charted':>9}"
              f"{'prec':>7}{'recall':>8}{'der.rate':>10}{'cht.rate':>10}"
              f"{'dur.err':>9}")
    print(header)
    print('-' * len(header))
    all_rows: List[dict] = []
    for difficulty in ('beginner', 'easy', 'medium', 'hard', 'challenge'):
        rows = by_difficulty.get(difficulty)
        if not rows:
            continue
        all_rows.extend(rows)
        a = aggregate(rows)
        print(f"{difficulty:<11}{a['charts']:>7}{a['derived']:>9}{a['charted']:>9}"
              f"{a['precision']:>7.2f}{a['recall']:>8.2f}"
              f"{a['derived_rate']:>10.3f}{a['charted_rate']:>10.3f}"
              f"{a['median_dur_err_beats']:>9.2f}")
    if all_rows:
        a = aggregate(all_rows)
        print('-' * len(header))
        print(f"{'ALL':<11}{a['charts']:>7}{a['derived']:>9}{a['charted']:>9}"
              f"{a['precision']:>7.2f}{a['recall']:>8.2f}"
              f"{a['derived_rate']:>10.3f}{a['charted_rate']:>10.3f}"
              f"{a['median_dur_err_beats']:>9.2f}")
    if skipped:
        print(f"\nskipped {skipped} entries (missing npz or unknown difficulty)")
    print("\nprec/recall are head-placement agreement with the human charter; "
          "low values are expected (hold placement is a stylistic choice, not a "
          "ground truth). Watch der.rate vs cht.rate and dur.err for tuning.")


if __name__ == '__main__':
    main()

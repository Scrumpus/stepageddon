"""Batch playability verification harness.

Generates charts for a sample of real songs (``backend/charts/*/*.ogg``) at all
five difficulties and reports how many playability violations each contains, using
``ml.playability.validate_chart``. Run it before and after the assigner rewrite to
show the before/after violation delta.

Usage (from ``backend/`` with the venv active and ML_MODEL_PATH set)::

    PYTHONPATH=. python -m scripts.validate_charts               # default sample
    PYTHONPATH=. python -m scripts.validate_charts --limit 4
    PYTHONPATH=. python -m scripts.validate_charts --songs "RED ZONE" "DEAD END"
    PYTHONPATH=. python -m scripts.validate_charts --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure `backend/` is importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.inference import MLChartGenerator  # noqa: E402
from ml.playability import report_chart, PLAYABILITY_SPEC  # noqa: E402
from src.config import settings  # noqa: E402

CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"
DIFFICULTIES = ["beginner", "easy", "medium", "hard", "challenge"]

# A varied default sample (different genres / tempos) when --songs isn't given.
DEFAULT_SAMPLE = [
    "RED ZONE",
    "Sing It Back (Boris Musical Mix)",
    "WHAT A WONDERFUL WORLD",
    "DEAD END",
]

VIOLATION_ORDER = [
    "concurrency", "fast_jack", "double_step",
    "disallowed_crossover", "unresolved_crossover", "jump_unreachable",
]


def _find_audio(song: str) -> Optional[Path]:
    d = CHARTS_DIR / song
    if not d.is_dir():
        return None
    oggs = sorted(d.glob("*.ogg"))
    return oggs[0] if oggs else None


def _discover_sample(limit: Optional[int], songs: Optional[List[str]]) -> List[Path]:
    if songs:
        picked = [(_find_audio(s), s) for s in songs]
        missing = [s for p, s in picked if p is None]
        if missing:
            print(f"WARNING: no .ogg found for: {missing}", file=sys.stderr)
        return [p for p, _ in picked if p is not None]

    paths: List[Path] = []
    for s in DEFAULT_SAMPLE:
        p = _find_audio(s)
        if p is not None:
            paths.append(p)
    # Top up from whatever else is available, deterministically by name.
    if not paths or (limit and len(paths) < limit):
        for d in sorted(CHARTS_DIR.iterdir()):
            if not d.is_dir():
                continue
            oggs = sorted(d.glob("*.ogg"))
            if oggs and oggs[0] not in paths:
                paths.append(oggs[0])
    if limit:
        paths = paths[:limit]
    return paths


def _load_generator() -> MLChartGenerator:
    model_path = settings.ML_MODEL_PATH
    profiles = None
    for cand in (
        os.path.join(os.path.dirname(model_path), "style_profiles.json"),
        os.path.join("ml", "checkpoints", "style_profiles.json"),
    ):
        if os.path.exists(cand):
            profiles = cand
            break
    print(f"Loading generator: model={model_path} profiles={profiles}")
    return MLChartGenerator(
        model_path=model_path,
        style_profiles_path=profiles,
        timing_trim=settings.GENERATION_TIMING_OFFSET_MS / 1000.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="cap number of songs")
    ap.add_argument("--songs", nargs="*", default=None, help="explicit song folder names")
    ap.add_argument("--json", type=str, default=None, help="write full results to this JSON path")
    args = ap.parse_args()

    audio_paths = _discover_sample(args.limit, args.songs)
    if not audio_paths:
        print(f"No audio found under {CHARTS_DIR}", file=sys.stderr)
        return 1

    gen = _load_generator()

    print("\nPlayability thresholds (jack_min_dt / crossovers / uncross_within / max_run):")
    for d in DIFFICULTIES:
        s = PLAYABILITY_SPEC[d]
        print(f"  {d:10s} {s.jack_min_dt*1000:4.0f}ms  "
              f"cross={str(s.crossovers_allowed):5s}  within={s.uncross_within}  run={s.max_run_length}")

    results: Dict[str, dict] = {}
    grand_total: Dict[str, int] = {}

    header = f"\n{'song':32s} {'difficulty':10s} {'steps':>6s} " + " ".join(
        f"{v[:9]:>9s}" for v in VIOLATION_ORDER) + f" {'TOTAL':>6s}"
    print(header)
    print("-" * len(header))

    for path in audio_paths:
        song = path.parent.name
        try:
            charts = gen.generate_all_difficulties(str(path), difficulties=DIFFICULTIES)
        except Exception as e:  # noqa: BLE001
            print(f"{song:32s} ERROR: {e}", file=sys.stderr)
            continue
        results[song] = {}
        for d in DIFFICULTIES:
            chart = charts.get(d)
            if chart is None:
                continue
            rep = report_chart(chart.steps, d)
            counts = rep.counts
            results[song][d] = {
                "steps": rep.step_count,
                "counts": counts,
                "total": rep.total,
                "violations": [str(v) for v in rep.violations],
            }
            for k, c in counts.items():
                grand_total[k] = grand_total.get(k, 0) + c
            row = " ".join(f"{counts.get(v, 0):>9d}" for v in VIOLATION_ORDER)
            print(f"{song[:32]:32s} {d:10s} {rep.step_count:>6d} {row} {rep.total:>6d}")

    print("-" * len(header))
    total_row = " ".join(f"{grand_total.get(v, 0):>9d}" for v in VIOLATION_ORDER)
    print(f"{'GRAND TOTAL':32s} {'':10s} {'':>6s} {total_row} "
          f"{sum(grand_total.values()):>6d}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"grand_total": grand_total, "songs": results}, f, indent=2)
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

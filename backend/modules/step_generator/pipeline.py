"""
Chart Generation Pipeline

High-level orchestration of the complete chart generation process.
Coordinates audio loading, analysis, generation, and export.

Enhanced with:
- Real pitch tracking for contour-aware arrow selection
- Phrase boundary detection for pattern switching
- Musical rest detection for breathing room
- Tempo change detection for adaptive grid
- Rhythmic simplification for difficulty scaling
- Off-grid timing support for expressive rhythms
"""

import logging
import librosa

from .schemas import Chart, StepType
from .difficulty import DIFFICULTY_PRESETS
from .audio_analysis import (
    analyze_energy,
    detect_sustained_notes,
    detect_structure,
    quantize_to_grid,
    # New analysis functions
    detect_pitch_contour,
    detect_phrase_boundaries,
    detect_musical_rests,
    quantize_to_grid_with_tolerance
)
from .madmom_analysis import (
    MadmomAnalysisCache,
    analyze_beats,
    analyze_onsets,
    detect_subdivisions,
    detect_tempo_changes,
)
from .generator import StepGenerator
from .simplification import RhythmSimplifier


logger = logging.getLogger(__name__)


class ChartGenerationPipeline:
    """Complete pipeline from audio to chart."""

    @staticmethod
    def generate_from_audio(audio_path: str, difficulty: str = 'intermediate') -> Chart:
        """
        Generate a complete chart from an audio file.

        Args:
            audio_path: Path to audio file
            difficulty: Difficulty level ('beginner', 'intermediate', 'expert')

        Returns:
            Complete Chart object

        Raises:
            ValueError: If difficulty is not recognized
        """
        logger.info(f"Loading audio from {audio_path}...")
        y, sr = librosa.load(audio_path)

        config = DIFFICULTY_PRESETS[difficulty]

        # === Initialize madmom cache (avoids re-running RNNs) ===
        madmom_cache = MadmomAnalysisCache(audio_path)

        # === Core Audio Analysis ===
        logger.info("Analyzing audio (core features)...")
        # Beat/onset detection via madmom RNNs (more accurate than librosa)
        beats, tempo = analyze_beats(audio_path, madmom_cache)
        subdivisions = detect_subdivisions(audio_path, [b.time for b in beats], madmom_cache)
        # Spectral/timbral analysis stays with librosa
        energy_sections = analyze_energy(y, sr)
        sustained_notes = detect_sustained_notes(y, sr)
        structure = detect_structure(y, sr)

        # === Enhanced Analysis (new features) ===
        logger.info("Analyzing audio (enhanced features)...")

        # Pitch tracking for contour-aware arrow selection (librosa)
        pitch_frames = []
        if config.use_contour:
            try:
                pitch_frames = detect_pitch_contour(y, sr)
                logger.info(f"Detected {len(pitch_frames)} pitch frames for contour tracking")
            except Exception as e:
                logger.warning(f"Pitch tracking failed, continuing without contour: {e}")

        # Phrase boundary detection for pattern switching (librosa, uses madmom beats)
        phrase_boundaries = detect_phrase_boundaries(y, sr, beats, tempo)
        logger.info(f"Detected {len(phrase_boundaries)} phrase boundaries")

        # Musical rest detection (librosa)
        rest_periods = detect_musical_rests(y, sr)
        logger.info(f"Detected {len(rest_periods)} musical rest periods")

        # Tempo change detection via madmom
        tempo_sections = detect_tempo_changes(audio_path, cache=madmom_cache)
        if len(tempo_sections) > 1:
            logger.info(f"Detected {len(tempo_sections)} tempo sections (variable tempo song)")
            # Use average tempo for now; full multi-tempo support is a future enhancement
            avg_tempo = sum(s.bpm for s in tempo_sections) / len(tempo_sections)
            logger.info(f"Using average tempo: {avg_tempo:.1f} BPM")
            tempo = avg_tempo
        else:
            logger.info(f"Constant tempo: {tempo:.1f} BPM")

        # === Onset Analysis with Simplification ===
        onset_times = None
        if config.use_onsets:
            raw_onsets, _ = analyze_onsets(audio_path, strength_threshold=config.onset_threshold, cache=madmom_cache)
            logger.info(f"Detected {len(raw_onsets)} raw onsets (threshold: {config.onset_threshold})")

            # Apply rhythmic simplification based on difficulty
            if config.rhythmic_simplification > 0:
                simplifier = RhythmSimplifier(tempo)
                phrase_times = [p.time for p in phrase_boundaries]
                raw_onsets = simplifier.simplify(
                    raw_onsets, beats, config.rhythmic_simplification, phrase_times
                )
                logger.info(f"Simplified to {len(raw_onsets)} onsets (level: {config.rhythmic_simplification})")

            # Quantize with tolerance (or strict grid based on config)
            grid_division = 16 if difficulty == 'expert' else 8

            if config.off_grid_tolerance_ms > 0:
                # Use tolerance-based quantization for expressive timing
                quantized_results = quantize_to_grid_with_tolerance(
                    raw_onsets, tempo, grid_division, config.off_grid_tolerance_ms
                )
                onset_times = [t for t, _ in quantized_results]
                on_grid_count = sum(1 for _, on_grid in quantized_results if on_grid)
                off_grid_count = len(quantized_results) - on_grid_count
                logger.info(f"Quantized: {on_grid_count} on-grid, {off_grid_count} off-grid "
                           f"(tolerance: {config.off_grid_tolerance_ms}ms)")
            else:
                # Strict grid quantization
                onset_times = quantize_to_grid(raw_onsets, tempo, grid_division)
                logger.info(f"Strict quantized to {len(onset_times)} grid-aligned onsets")

        logger.info(f"Detected {len(beats)} beats at {tempo:.1f} BPM")
        logger.info(f"Found {len(sustained_notes)} sustained notes for holds")

        # === Chart Generation ===
        logger.info(f"Generating {difficulty} chart...")
        generator = StepGenerator(config)

        chart = generator.generate_chart(
            beats=beats,
            subdivisions=subdivisions,
            energy_sections=energy_sections,
            sustained_notes=sustained_notes,
            structure=structure,
            tempo=tempo,
            onset_times=onset_times,
            pitch_frames=pitch_frames,
            phrase_boundaries=phrase_boundaries,
            rest_periods=rest_periods
        )

        logger.info(f"Generated {len(chart.steps)} steps")
        logger.info(f"  Taps: {len(chart.get_taps())}")
        logger.info(f"  Holds: {len(chart.get_holds())}")

        return chart


class ChartExporter:
    """Export charts to various formats."""

    @staticmethod
    def to_json(chart: Chart) -> dict:
        """
        Export chart to JSON-compatible dictionary format.

        Args:
            chart: Chart object to export

        Returns:
            Dictionary containing chart data in JSON-compatible format
        """
        steps_data = []

        for step in chart.steps:
            step_dict = {
                'time': round(step.time, 3),
                'arrows': [a.value for a in step.arrows],
                'type': step.step_type.value
            }

            if step.step_type == StepType.HOLD:
                step_dict['hold_duration'] = round(step.hold_duration, 3)

            steps_data.append(step_dict)

        chart_data = {
            'difficulty': chart.difficulty,
            'tempo': round(chart.tempo, 1),
            'duration': round(chart.duration, 2),
            'steps': steps_data,
            'stats': {
                'total_steps': len(chart.steps),
                'total_arrows': sum(len(s.arrows) for s in chart.steps),
                'tap_notes': len(chart.get_taps()),
                'hold_notes': len(chart.get_holds()),
                'singles': len([s for s in chart.steps if len(s.arrows) == 1]),
                'doubles': len([s for s in chart.steps if len(s.arrows) == 2]),
            }
        }

        return chart_data

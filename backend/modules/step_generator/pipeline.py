"""
Chart Generation Pipeline

High-level orchestration of the complete chart generation process.
Coordinates audio loading, analysis, generation, and export.
"""

import logging
import librosa

from .schemas import Chart, StepType
from .difficulty import DIFFICULTY_PRESETS
from .audio_analysis import (
    analyze_beats,
    analyze_onsets,
    detect_subdivisions,
    analyze_energy,
    detect_sustained_notes,
    detect_structure,
    quantize_to_grid,
    # New enhanced analysis functions
    separate_harmonic_percussive,
    detect_drum_events,
    analyze_weighted_onsets,
    detect_melody_notes_pyin,
    score_beat_prominence
)
from .generator import StepGenerator


logger = logging.getLogger(__name__)


class ChartGenerationPipeline:
    """Complete pipeline from audio to chart."""

    @staticmethod
    def generate_from_audio(audio_path: str, difficulty: str = 'intermediate') -> Chart:
        """
        Generate a complete chart from an audio file.

        Uses enhanced audio analysis including:
        - HPSS (Harmonic-Percussive Source Separation)
        - Frequency-band drum detection (kick/snare/hihat)
        - Weighted onset analysis with drum correlation
        - pYIN pitch tracking for accurate melody detection
        - Beat prominence scoring based on drum alignment

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

        # =========================================================================
        # Phase 1: HPSS - Separate harmonic and percussive components
        # =========================================================================
        logger.info("Performing HPSS separation...")
        y_harmonic, y_percussive = separate_harmonic_percussive(y)

        # =========================================================================
        # Phase 2: Drum detection on percussive component
        # =========================================================================
        logger.info("Detecting drums (kick/snare/hihat)...")
        drum_track = detect_drum_events(y, sr, y_percussive)

        # =========================================================================
        # Phase 3: Basic beat and structure analysis
        # =========================================================================
        logger.info("Analyzing beats and structure...")
        beats, tempo = analyze_beats(y, sr)
        subdivisions = detect_subdivisions(y, sr, [b.time for b in beats])
        energy_sections = analyze_energy(y, sr)
        structure = detect_structure(y, sr)

        # =========================================================================
        # Phase 4: Score beats by drum prominence
        # =========================================================================
        logger.info("Scoring beat prominence based on drums...")
        beats = score_beat_prominence(beats, drum_track)

        # =========================================================================
        # Phase 5: Melody detection using pYIN on harmonic component
        # =========================================================================
        logger.info("Detecting melodic notes with pYIN...")
        sustained_notes = detect_melody_notes_pyin(y, sr, y_harmonic)

        # Fall back to energy-based detection if pYIN finds nothing
        if not sustained_notes:
            logger.info("pYIN found no sustained notes, falling back to energy-based detection")
            sustained_notes = detect_sustained_notes(y, sr)

        # =========================================================================
        # Phase 6: Weighted onset analysis with drum correlation
        # =========================================================================
        weighted_onsets = None
        onset_times = None

        if config.use_onsets:
            logger.info("Analyzing weighted onsets...")
            weighted_onsets = analyze_weighted_onsets(
                y, sr,
                drum_track=drum_track,
                strength_threshold=config.onset_threshold
            )

            # Extract times for backward compatibility
            onset_times = [wo.time for wo in weighted_onsets]
            logger.info(f"Detected {len(onset_times)} weighted onsets (threshold: {config.onset_threshold})")

            # Quantize to musical grid for better flow
            grid_division = 16 if difficulty == 'expert' else 8
            onset_times = quantize_to_grid(onset_times, tempo, grid_division)
            logger.info(f"Quantized to {len(onset_times)} grid-aligned onsets ({grid_division}th notes)")

        # =========================================================================
        # Summary logging
        # =========================================================================
        logger.info(f"Detected {len(beats)} beats at {tempo:.1f} BPM")
        logger.info(f"Found {len(sustained_notes)} sustained notes for holds")
        logger.info(f"Drum events: {len(drum_track.kicks)} kicks, {len(drum_track.snares)} snares, {len(drum_track.hihats)} hi-hats")

        # =========================================================================
        # Phase 7: Generate chart with all enhanced data
        # =========================================================================
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
            drum_track=drum_track,
            weighted_onsets=weighted_onsets
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

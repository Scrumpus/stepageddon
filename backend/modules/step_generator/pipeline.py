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
    detect_subdivisions,
    analyze_energy,
    detect_sustained_notes,
    detect_structure
)
from .generator import StepGenerator


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

        logger.info("Analyzing audio...")
        beats, tempo = analyze_beats(y, sr)
        subdivisions = detect_subdivisions(y, sr, [b.time for b in beats])
        energy_sections = analyze_energy(y, sr)
        sustained_notes = detect_sustained_notes(y, sr)
        structure = detect_structure(y, sr)

        logger.info(f"Detected {len(beats)} beats at {tempo:.1f} BPM")
        logger.info(f"Found {len(sustained_notes)} sustained notes for holds")

        logger.info(f"Generating {difficulty} chart...")
        config = DIFFICULTY_PRESETS[difficulty]
        generator = StepGenerator(config)

        chart = generator.generate_chart(
            beats=beats,
            subdivisions=subdivisions,
            energy_sections=energy_sections,
            sustained_notes=sustained_notes,
            structure=structure,
            tempo=tempo
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

"""
Step Generator Module

Deterministic DDR-style chart generation from audio analysis.
This module provides a complete pipeline from audio file to playable step chart.

Main API:
    ChartGenerationPipeline: High-level pipeline for chart generation
    ChartExporter: Export charts to JSON format
    StepGenerator: Core generation algorithm

Example:
    >>> from modules.step_generator import ChartGenerationPipeline
    >>> chart = ChartGenerationPipeline.generate_from_audio('song.mp3', 'intermediate')
"""

# Core exports (most commonly used)
from .pipeline import ChartGenerationPipeline, ChartExporter
from .generator import StepGenerator
from .schemas import Chart, Step, StepType, Direction

# Advanced exports (for customization)
from .difficulty import DIFFICULTY_PRESETS, DifficultyConfig, get_difficulty_config
from .patterns import PatternTemplate

# Audio analysis functions (for advanced users)
from .audio_analysis import (
    analyze_beats,
    analyze_onsets,
    detect_subdivisions,
    analyze_energy,
    detect_energy_peaks,
    detect_sustained_notes,
    detect_structure,
    quantize_to_grid,
    # Enhanced audio analysis (HPSS, drums, melody)
    separate_harmonic_percussive,
    detect_drum_events,
    analyze_weighted_onsets,
    filter_onsets_by_strength,
    detect_melody_notes_pyin,
    score_beat_prominence,
)

# Additional data schemas
from .schemas import Beat, EnergySection, SustainedNote, SongStructure, DrumEvent, DrumTrack, WeightedOnset

__version__ = '2.0.0'

__all__ = [
    # Primary API
    "ChartGenerationPipeline",
    "ChartExporter",
    "StepGenerator",

    # Data types
    "Chart",
    "Step",
    "StepType",
    "Direction",
    "Beat",
    "EnergySection",
    "SustainedNote",
    "SongStructure",
    "DrumEvent",
    "DrumTrack",
    "WeightedOnset",

    # Configuration
    "DIFFICULTY_PRESETS",
    "DifficultyConfig",
    "get_difficulty_config",

    # Patterns
    "PatternTemplate",

    # Audio analysis (basic)
    "analyze_beats",
    "analyze_onsets",
    "detect_subdivisions",
    "analyze_energy",
    "detect_energy_peaks",
    "detect_sustained_notes",
    "detect_structure",
    "quantize_to_grid",

    # Audio analysis (enhanced - HPSS, drums, melody)
    "separate_harmonic_percussive",
    "detect_drum_events",
    "analyze_weighted_onsets",
    "filter_onsets_by_strength",
    "detect_melody_notes_pyin",
    "score_beat_prominence",
]

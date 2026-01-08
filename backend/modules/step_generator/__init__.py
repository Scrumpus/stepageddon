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
    # Advanced musical source analysis
    analyze_hpss,
    analyze_separated_sources,
    analyze_multiband_onsets,
    detect_kick_snare,
    track_melody,
    analyze_musical_sources,
    build_step_candidates,
)

# Additional data schemas
from .schemas import (
    Beat,
    EnergySection,
    SustainedNote,
    SongStructure,
    MusicalSources,
    MelodyNote,
    StepCandidate,
)

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
    "MusicalSources",
    "MelodyNote",
    "StepCandidate",

    # Configuration
    "DIFFICULTY_PRESETS",
    "DifficultyConfig",
    "get_difficulty_config",

    # Patterns
    "PatternTemplate",

    # Audio analysis
    "analyze_beats",
    "analyze_onsets",
    "detect_subdivisions",
    "analyze_energy",
    "detect_energy_peaks",
    "detect_sustained_notes",
    "detect_structure",
    "quantize_to_grid",

    # Advanced musical source analysis
    "analyze_hpss",
    "analyze_separated_sources",
    "analyze_multiband_onsets",
    "detect_kick_snare",
    "track_melody",
    "analyze_musical_sources",
    "build_step_candidates",
]

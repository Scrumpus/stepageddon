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
from .contour import ContourArrowMapper
from .simplification import RhythmSimplifier

# Audio analysis functions (for advanced users)
# Beat/onset/tempo detection via madmom RNNs
from .madmom_analysis import (
    MadmomAnalysisCache,
    analyze_beats,
    analyze_onsets,
    detect_subdivisions,
    detect_tempo_changes,
)
# Spectral/timbral analysis via librosa
from .audio_analysis import (
    analyze_energy,
    detect_energy_peaks,
    detect_sustained_notes,
    detect_structure,
    quantize_to_grid,
    detect_pitch_contour,
    classify_onsets_by_band,
    detect_phrase_boundaries,
    detect_musical_rests,
    quantize_to_grid_with_tolerance,
)

# Additional data schemas
from .schemas import (
    Beat, EnergySection, SustainedNote, SongStructure,
    # New schemas for enhanced analysis
    PitchFrame, TempoSection, PhraseBoundary, OnsetClassification, ContourDirection
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
    # New data types
    "PitchFrame",
    "TempoSection",
    "PhraseBoundary",
    "OnsetClassification",
    "ContourDirection",

    # Configuration
    "DIFFICULTY_PRESETS",
    "DifficultyConfig",
    "get_difficulty_config",

    # Patterns
    "PatternTemplate",

    # Contour
    "ContourArrowMapper",

    # Simplification
    "RhythmSimplifier",

    # Madmom analysis cache
    "MadmomAnalysisCache",

    # Audio analysis
    "analyze_beats",
    "analyze_onsets",
    "detect_subdivisions",
    "analyze_energy",
    "detect_energy_peaks",
    "detect_sustained_notes",
    "detect_structure",
    "quantize_to_grid",
    # New audio analysis functions
    "detect_pitch_contour",
    "classify_onsets_by_band",
    "detect_tempo_changes",
    "detect_phrase_boundaries",
    "detect_musical_rests",
    "quantize_to_grid_with_tolerance",
]

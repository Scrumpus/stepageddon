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

from .pipeline import ChartGenerationPipeline, ChartExporter
from .generator import StepGenerator
from .schemas import Chart, Step, StepType, Direction, DifficultyConfig, Beat, EnergySection
from .difficulty import DIFFICULTY_PRESETS, get_difficulty_config
from .patterns import PatternTemplate

__version__ = '3.0.0'

__all__ = [
    "ChartGenerationPipeline",
    "ChartExporter",
    "StepGenerator",
    "Chart",
    "Step",
    "StepType",
    "Direction",
    "DifficultyConfig",
    "Beat",
    "EnergySection",
    "DIFFICULTY_PRESETS",
    "get_difficulty_config",
    "PatternTemplate",
]

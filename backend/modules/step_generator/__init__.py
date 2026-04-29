"""
Step Generator Schemas

Shared chart schemas and difficulty presets used by the ML generator
(`ml.inference.MLChartGenerator`) and the persistence / parser layers.

Generation itself lives in `ml/`; this package now only provides the
data structures and difficulty configs.
"""

from .schemas import (
    Chart,
    Step,
    StepType,
    Direction,
    DifficultyConfig,
    Beat,
    EnergySection,
    BeatSubdivision,
)
from .difficulty import DIFFICULTY_PRESETS, get_difficulty_config

__all__ = [
    "Chart",
    "Step",
    "StepType",
    "Direction",
    "DifficultyConfig",
    "Beat",
    "EnergySection",
    "BeatSubdivision",
    "DIFFICULTY_PRESETS",
    "get_difficulty_config",
]

"""
Pydantic Schemas for Step Generation Engine

Contains all data structures used throughout the step generation pipeline.
Uses Pydantic for validation and serialization.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum


class StepType(str, Enum):
    """Types of steps in the game."""
    TAP = "tap"
    HOLD = "hold"


class Direction(str, Enum):
    """Arrow directions."""
    LEFT = "left"
    DOWN = "down"
    UP = "up"
    RIGHT = "right"


class Beat(BaseModel):
    """Represents a single beat."""
    time: float
    strength: float
    beat_type: str  # 'downbeat', 'upbeat', 'offbeat'
    measure_position: int  # 0-3 for 4/4 time
    is_strong: bool

    model_config = {"frozen": False}


class EnergySection(BaseModel):
    """Represents an energy level over a time range."""
    start_time: float
    end_time: float
    energy_level: float = Field(ge=0.0, le=1.0)  # 0.0 to 1.0
    intensity: str  # 'low', 'medium', 'high', 'climax'

    model_config = {"frozen": False}


class BeatSubdivision(str, Enum):
    """Beat subdivision for note coloring (DDR-style)."""
    QUARTER = "4th"
    EIGHTH = "8th"
    SIXTEENTH = "16th"


class Step(BaseModel):
    """A single step (tap or hold start)."""
    time: float
    arrows: List[Direction]
    step_type: StepType = StepType.TAP
    hold_duration: Optional[float] = None
    beat_subdivision: BeatSubdivision = BeatSubdivision.QUARTER

    @field_validator('hold_duration')
    @classmethod
    def validate_hold_duration(cls, v, info):
        """Validate hold duration matches step type."""
        step_type = info.data.get('step_type')
        if step_type == StepType.HOLD and v is None:
            raise ValueError("Hold steps must have hold_duration")
        if step_type == StepType.TAP and v is not None:
            raise ValueError("Tap steps cannot have hold_duration")
        return v

    model_config = {"frozen": False}


class Chart(BaseModel):
    """Complete chart with all steps."""
    steps: List[Step]
    difficulty: str
    tempo: float
    duration: float

    def get_taps(self) -> List[Step]:
        """Get only tap notes."""
        return [s for s in self.steps if s.step_type == StepType.TAP]

    def get_holds(self) -> List[Step]:
        """Get only hold notes."""
        return [s for s in self.steps if s.step_type == StepType.HOLD]

    model_config = {"frozen": False}


class DifficultyConfig(BaseModel):
    """Configuration for each difficulty level."""
    name: str
    min_density: float          # steps/second minimum
    max_density: float          # steps/second maximum
    grid_resolution: int        # 4=quarter notes, 8=eighth, 16=sixteenth
    use_onsets: bool = False    # merge onset times into candidates
    onset_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    allowed_patterns: List[str] # ['single','jump','stream','crossover','gallop','drill']
    max_stream_length: int = 0  # 0 = no streams
    max_drill_length: int = 0   # 0 = no drills
    hold_percentage: float = Field(default=0.15, ge=0.0, le=1.0)
    min_hold_duration: float = 0.5
    max_hold_duration: float = 3.0
    energy_scale_factor: float = 0.5  # how much energy affects density
    min_gap: float = 0.15      # minimum seconds between any two steps

    model_config = {"frozen": False}

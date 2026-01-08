"""
Pydantic Schemas for Step Generation Engine

Contains all data structures used throughout the step generation pipeline.
Uses Pydantic for validation and serialization.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Tuple, Optional
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


class SustainedNote(BaseModel):
    """Represents a sustained melodic note (candidate for hold)."""
    start_time: float
    end_time: float
    pitch: float  # MIDI note number or Hz frequency
    confidence: float

    model_config = {"frozen": False}


class DrumEvent(BaseModel):
    """Represents a detected drum hit."""
    time: float
    drum_type: str  # 'kick', 'snare', 'hihat'
    strength: float = Field(ge=0.0, le=1.0)

    model_config = {"frozen": False}


class DrumTrack(BaseModel):
    """Collection of drum events by type."""
    kicks: List['DrumEvent'] = []
    snares: List['DrumEvent'] = []
    hihats: List['DrumEvent'] = []

    def all_events(self) -> List['DrumEvent']:
        """Get all drum events sorted by time."""
        all_drums = self.kicks + self.snares + self.hihats
        return sorted(all_drums, key=lambda d: d.time)

    model_config = {"frozen": False}


class WeightedOnset(BaseModel):
    """An onset with its strength/prominence score."""
    time: float
    strength: float = Field(ge=0.0, le=1.0)
    is_drum: bool = False
    drum_type: Optional[str] = None  # 'kick', 'snare', 'hihat' if is_drum

    model_config = {"frozen": False}


class SongStructure(BaseModel):
    """Complete song structure."""
    intro: Tuple[float, float]
    verses: List[Tuple[float, float]]
    choruses: List[Tuple[float, float]]
    bridge: Tuple[float, float]
    outro: Tuple[float, float]
    total_duration: float

    model_config = {"frozen": False}


class Step(BaseModel):
    """A single step (tap or hold start)."""
    time: float
    arrows: List[Direction]
    step_type: StepType = StepType.TAP
    hold_duration: Optional[float] = None

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
    min_density: float
    max_density: float
    allow_singles: bool
    allow_doubles: bool
    allow_holds: bool
    hold_percentage: float = Field(ge=0.0, le=1.0)
    min_hold_duration: float
    max_hold_duration: float
    use_downbeats: bool
    use_upbeats: bool
    use_offbeats: bool
    use_8th_notes: bool
    use_16th_notes: bool
    use_onsets: bool = False  # Use onset detection for denser step placement
    onset_threshold: float = Field(default=0.3, ge=0.0, le=1.0)  # Min onset strength
    max_consecutive_jumps: int
    max_stream_length: int
    allow_crossovers: bool
    allow_brackets: bool
    energy_scale_factor: float

    model_config = {"frozen": False}

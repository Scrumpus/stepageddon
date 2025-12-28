"""
Difficulty Configuration

Defines difficulty levels and their parameters for chart generation.
Each difficulty has specific constraints on density, timing, and patterns.
"""

from .schemas import DifficultyConfig


# Difficulty Presets
DIFFICULTY_PRESETS = {
    "beginner": DifficultyConfig(
        name="beginner",
        min_density=0.6,
        max_density=1.2,
        allow_singles=True,
        allow_doubles=False,
        allow_holds=True,
        hold_percentage=0.15,
        min_hold_duration=0.8,
        max_hold_duration=2.0,
        use_downbeats=True,
        use_upbeats=False,
        use_offbeats=False,
        use_8th_notes=False,
        use_16th_notes=False,
        max_consecutive_jumps=0,
        max_stream_length=0,
        allow_crossovers=False,
        allow_brackets=False,
        energy_scale_factor=0.3
    ),

    "intermediate": DifficultyConfig(
        name="intermediate",
        min_density=1.3,
        max_density=2.3,
        allow_singles=True,
        allow_doubles=True,
        allow_holds=True,
        hold_percentage=0.20,
        min_hold_duration=0.6,
        max_hold_duration=3.0,
        use_downbeats=True,
        use_upbeats=True,
        use_offbeats=True,
        use_8th_notes=True,
        use_16th_notes=False,
        max_consecutive_jumps=2,
        max_stream_length=6,
        allow_crossovers=True,
        allow_brackets=False,
        energy_scale_factor=0.6
    ),

    "expert": DifficultyConfig(
        name="expert",
        min_density=2.2,
        max_density=4.0,
        allow_singles=True,
        allow_doubles=True,
        allow_holds=True,
        hold_percentage=0.25,
        min_hold_duration=0.5,
        max_hold_duration=4.0,
        use_downbeats=True,
        use_upbeats=True,
        use_offbeats=True,
        use_8th_notes=True,
        use_16th_notes=True,
        max_consecutive_jumps=4,
        max_stream_length=16,
        allow_crossovers=True,
        allow_brackets=True,
        energy_scale_factor=1.0
    )
}


def get_difficulty_config(difficulty: str) -> DifficultyConfig:
    """
    Get configuration for a difficulty level with validation.

    Args:
        difficulty: Difficulty level ('beginner', 'intermediate', 'expert')

    Returns:
        DifficultyConfig for the requested difficulty

    Raises:
        ValueError: If difficulty is not recognized
    """
    if difficulty not in DIFFICULTY_PRESETS:
        available = ', '.join(DIFFICULTY_PRESETS.keys())
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Available difficulties: {available}"
        )
    return DIFFICULTY_PRESETS[difficulty]

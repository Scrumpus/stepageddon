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
        min_density=0.5,
        max_density=1.0,
        grid_resolution=4,          # quarter notes only (downbeats)
        use_onsets=False,
        onset_threshold=0.35,
        allowed_patterns=["single"],
        max_stream_length=0,
        max_drill_length=0,
        hold_percentage=0.10,
        min_hold_duration=0.8,
        max_hold_duration=2.0,
        energy_scale_factor=0.25,
        min_gap=0.40,
    ),

    "easy": DifficultyConfig(
        name="easy",
        min_density=0.9,
        max_density=1.6,
        grid_resolution=4,          # mostly quarter notes
        use_onsets=False,
        onset_threshold=0.3,
        allowed_patterns=["single"],
        max_stream_length=0,
        max_drill_length=0,
        hold_percentage=0.15,
        min_hold_duration=0.7,
        max_hold_duration=2.5,
        energy_scale_factor=0.4,
        min_gap=0.30,
    ),

    "medium": DifficultyConfig(
        name="medium",
        min_density=1.3,
        max_density=2.3,
        grid_resolution=8,          # eighth notes
        use_onsets=True,
        onset_threshold=0.2,
        allowed_patterns=["single", "jump", "stream", "crossover", "gallop"],
        max_stream_length=6,
        max_drill_length=0,
        hold_percentage=0.20,
        min_hold_duration=0.6,
        max_hold_duration=3.0,
        energy_scale_factor=0.6,
        min_gap=0.15,
    ),

    "hard": DifficultyConfig(
        name="hard",
        min_density=2.2,
        max_density=4.0,
        grid_resolution=16,         # sixteenth notes
        use_onsets=True,
        onset_threshold=0.1,
        allowed_patterns=["single", "jump", "stream", "crossover", "gallop", "drill"],
        max_stream_length=16,
        max_drill_length=8,
        hold_percentage=0.25,
        min_hold_duration=0.5,
        max_hold_duration=4.0,
        energy_scale_factor=1.0,
        min_gap=0.08,
    ),

    "challenge": DifficultyConfig(
        name="challenge",
        min_density=3.5,
        max_density=6.0,
        grid_resolution=16,
        use_onsets=True,
        onset_threshold=0.05,
        allowed_patterns=["single", "jump", "stream", "crossover", "gallop", "drill"],
        max_stream_length=32,
        max_drill_length=16,
        hold_percentage=0.30,
        min_hold_duration=0.3,
        max_hold_duration=6.0,
        energy_scale_factor=1.5,
        min_gap=0.05,
    ),
}


def get_difficulty_config(difficulty: str) -> DifficultyConfig:
    """
    Get configuration for a difficulty level with validation.

    Args:
        difficulty: Difficulty level ('beginner', 'easy', 'medium', 'hard', 'challenge')

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

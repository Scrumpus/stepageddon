"""
Contour-Aware Arrow Selection

Maps melodic pitch contour to arrow directions for musically-connected step charts.
Ascending melodies move arrows "upward" visually, descending moves "downward".

Visual arrow height ordering (as seen on screen):
    ↓ (DOWN)  = 0 (lowest)
    ← (LEFT)  = 1
    → (RIGHT) = 2
    ↑ (UP)    = 3 (highest)
"""

from typing import List, Optional
from .schemas import Direction, PitchFrame, ContourDirection


class ContourArrowMapper:
    """
    Maps melodic contour to arrow directions while respecting foot alternation.

    The mapper tries to make ascending melodies feel like "climbing" and
    descending melodies feel like "falling" through the arrow positions.
    """

    # Visual height ordering (low to high on screen)
    ARROW_HEIGHT = {
        Direction.DOWN: 0,   # Lowest - appears at bottom
        Direction.LEFT: 1,   # Lower-middle
        Direction.RIGHT: 2,  # Upper-middle
        Direction.UP: 3,     # Highest - appears at top
    }

    # Reverse mapping for height to arrow
    HEIGHT_TO_ARROW = {v: k for k, v in ARROW_HEIGHT.items()}

    # Foot assignments for natural stepping
    LEFT_FOOT = [Direction.LEFT, Direction.DOWN]
    RIGHT_FOOT = [Direction.UP, Direction.RIGHT]

    # Ordered by height within each foot
    LEFT_FOOT_BY_HEIGHT = [Direction.DOWN, Direction.LEFT]   # 0, 1
    RIGHT_FOOT_BY_HEIGHT = [Direction.RIGHT, Direction.UP]   # 2, 3

    @classmethod
    def get_contour_direction(cls, pitch_frames: List[PitchFrame],
                               start_time: float, end_time: float,
                               min_change_semitones: float = 2.0) -> ContourDirection:
        """
        Analyze pitch contour over a time window.

        Args:
            pitch_frames: List of PitchFrame objects from pitch tracking
            start_time: Start of analysis window
            end_time: End of analysis window
            min_change_semitones: Minimum pitch change to be considered ascending/descending

        Returns:
            ContourDirection indicating the overall melodic movement
        """
        # Filter frames within the time window
        relevant_frames = [
            f for f in pitch_frames
            if start_time <= f.time <= end_time
        ]

        if len(relevant_frames) < 2:
            return ContourDirection.STABLE

        # Get average pitch at start and end of window
        window_duration = end_time - start_time
        mid_time = start_time + window_duration / 2

        early_frames = [f for f in relevant_frames if f.time < mid_time]
        late_frames = [f for f in relevant_frames if f.time >= mid_time]

        if not early_frames or not late_frames:
            return ContourDirection.STABLE

        # Use median for robustness against outliers
        early_pitch = sorted([f.midi_note for f in early_frames])[len(early_frames) // 2]
        late_pitch = sorted([f.midi_note for f in late_frames])[len(late_frames) // 2]

        pitch_change = late_pitch - early_pitch

        if pitch_change >= min_change_semitones:
            return ContourDirection.ASCENDING
        elif pitch_change <= -min_change_semitones:
            return ContourDirection.DESCENDING
        else:
            return ContourDirection.STABLE

    @classmethod
    def select_arrow_for_contour(cls, contour: ContourDirection,
                                  last_arrow: Optional[Direction],
                                  prefer_foot: Optional[str] = None,
                                  time_seed: int = 0) -> Direction:
        """
        Select next arrow based on melodic contour while respecting dance ergonomics.

        Args:
            contour: The melodic direction (ascending/descending/stable)
            last_arrow: The previous arrow (for foot alternation)
            prefer_foot: 'left' or 'right' to prefer a specific foot, or None for alternation
            time_seed: Seed for deterministic variation (typically int(time * 100) % 100)

        Returns:
            Direction for the next arrow
        """
        # Determine which foot to use
        if prefer_foot == 'left':
            use_left_foot = True
        elif prefer_foot == 'right':
            use_left_foot = False
        elif last_arrow is None:
            use_left_foot = True  # Start with left foot
        else:
            # Alternate feet
            use_left_foot = last_arrow not in cls.LEFT_FOOT

        # Get available arrows for this foot, ordered by height
        if use_left_foot:
            available = cls.LEFT_FOOT_BY_HEIGHT  # [DOWN, LEFT] heights 0, 1
        else:
            available = cls.RIGHT_FOOT_BY_HEIGHT  # [RIGHT, UP] heights 2, 3

        # Select based on contour
        if contour == ContourDirection.ASCENDING:
            # Prefer higher arrow within the foot's options
            return available[-1]  # Higher option
        elif contour == ContourDirection.DESCENDING:
            # Prefer lower arrow within the foot's options
            return available[0]  # Lower option
        else:
            # Stable: use time seed for variety
            idx = time_seed % len(available)
            return available[idx]

    @classmethod
    def get_contour_aware_sequence(cls, times: List[float],
                                    pitch_frames: List[PitchFrame],
                                    lookback_window: float = 0.5) -> List[Direction]:
        """
        Generate a full sequence of arrows following melodic contour.

        Args:
            times: List of step times
            pitch_frames: Pitch tracking data
            lookback_window: How far back to look for pitch context (seconds)

        Returns:
            List of Direction objects matching the input times
        """
        if not times:
            return []

        arrows = []
        last_arrow = None

        for i, time in enumerate(times):
            # Define analysis window
            start_time = max(0, time - lookback_window)
            end_time = time

            # Get contour for this moment
            contour = cls.get_contour_direction(pitch_frames, start_time, end_time)

            # Select arrow
            time_seed = int(time * 100) % 100
            arrow = cls.select_arrow_for_contour(contour, last_arrow, time_seed=time_seed)

            arrows.append(arrow)
            last_arrow = arrow

        return arrows

    @classmethod
    def get_arrow_height(cls, arrow: Direction) -> int:
        """Get the visual height of an arrow (0-3)."""
        return cls.ARROW_HEIGHT[arrow]

    @classmethod
    def get_higher_arrow(cls, arrow: Direction, same_foot: bool = False) -> Optional[Direction]:
        """
        Get the next higher arrow.

        Args:
            arrow: Current arrow
            same_foot: If True, only return arrows on the same foot

        Returns:
            Higher arrow, or None if already at highest
        """
        current_height = cls.ARROW_HEIGHT[arrow]

        if same_foot:
            # Stay on same foot
            is_left_foot = arrow in cls.LEFT_FOOT
            available = cls.LEFT_FOOT_BY_HEIGHT if is_left_foot else cls.RIGHT_FOOT_BY_HEIGHT
            current_idx = available.index(arrow)
            if current_idx < len(available) - 1:
                return available[current_idx + 1]
            return None
        else:
            # Any arrow
            if current_height < 3:
                return cls.HEIGHT_TO_ARROW[current_height + 1]
            return None

    @classmethod
    def get_lower_arrow(cls, arrow: Direction, same_foot: bool = False) -> Optional[Direction]:
        """
        Get the next lower arrow.

        Args:
            arrow: Current arrow
            same_foot: If True, only return arrows on the same foot

        Returns:
            Lower arrow, or None if already at lowest
        """
        current_height = cls.ARROW_HEIGHT[arrow]

        if same_foot:
            # Stay on same foot
            is_left_foot = arrow in cls.LEFT_FOOT
            available = cls.LEFT_FOOT_BY_HEIGHT if is_left_foot else cls.RIGHT_FOOT_BY_HEIGHT
            current_idx = available.index(arrow)
            if current_idx > 0:
                return available[current_idx - 1]
            return None
        else:
            # Any arrow
            if current_height > 0:
                return cls.HEIGHT_TO_ARROW[current_height - 1]
            return None

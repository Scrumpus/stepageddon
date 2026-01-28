"""
Rhythmic Simplification

Simplifies complex rhythmic patterns for easier difficulties while
preserving the musical feel. Expert mode gets literal rhythm reproduction,
while beginner mode gets maximum simplification.

Simplification levels:
- 0.0 (Expert): No simplification, literal rhythm
- 0.5 (Intermediate): Remove 16th notes, snap syncopations within 1/8 note
- 1.0 (Beginner): Remove 8th notes, snap to quarter notes, thin dense sections
"""

from typing import List
from .schemas import Beat


class RhythmSimplifier:
    """
    Simplifies complex rhythms for easier difficulties.

    The simplifier preserves the musical essence while making patterns
    more accessible for players at lower skill levels.
    """

    def __init__(self, tempo: float):
        """
        Initialize simplifier with tempo.

        Args:
            tempo: Tempo in BPM (needed for time calculations)
        """
        self.tempo = tempo
        self.beat_duration = 60.0 / tempo  # Quarter note duration
        self.eighth_duration = self.beat_duration / 2
        self.sixteenth_duration = self.beat_duration / 4

    def simplify(self, onset_times: List[float],
                 beats: List[Beat],
                 simplification_level: float,
                 phrase_boundaries: List[float] = None) -> List[float]:
        """
        Reduce rhythmic complexity based on simplification level.

        Simplification strategies:
        1. Remove very fast subdivisions (16ths for intermediate, 8ths for beginner)
        2. Snap syncopated notes to nearest strong beat
        3. Thin out dense clusters
        4. Always preserve downbeats and phrase boundaries

        Args:
            onset_times: Original onset times to simplify
            beats: Beat data for strength information
            simplification_level: 0.0 (literal) to 1.0 (maximum simplification)
            phrase_boundaries: Optional phrase boundary times to preserve

        Returns:
            Simplified list of onset times
        """
        if simplification_level <= 0.0:
            # No simplification - return original
            return list(onset_times)

        if not onset_times:
            return []

        times = sorted(onset_times)
        phrase_times = set(phrase_boundaries) if phrase_boundaries else set()
        downbeat_times = {b.time for b in beats if b.beat_type == 'downbeat'}
        strong_beat_times = {b.time for b in beats if b.is_strong}

        # Step 1: Remove fast subdivisions
        if simplification_level >= 0.5:
            # Remove 16th notes (keep only times that aren't 16th-note syncopations)
            times = self._remove_fast_subdivisions(times, self.eighth_duration * 0.9)

        if simplification_level >= 1.0:
            # Remove 8th notes as well
            times = self._remove_fast_subdivisions(times, self.beat_duration * 0.9)

        # Step 2: Snap syncopations to strong beats
        snap_tolerance = self.eighth_duration * simplification_level
        times = self._snap_to_strong_beats(times, beats, snap_tolerance)

        # Step 3: Thin dense clusters
        min_gap = self.eighth_duration * (1 + simplification_level)
        times = self._thin_dense_clusters(times, min_gap, strong_beat_times)

        # Step 4: Ensure we preserve important positions
        preserved = downbeat_times.union(phrase_times)
        for t in preserved:
            # Find if any onset is close to this position
            near_onsets = [o for o in onset_times if abs(o - t) < self.sixteenth_duration]
            if near_onsets and t not in times:
                # Add the preserved time
                times.append(t)

        return sorted(set(times))

    def _remove_fast_subdivisions(self, times: List[float], min_gap: float) -> List[float]:
        """
        Remove notes that create faster-than-allowed subdivisions.

        Args:
            times: List of times to filter
            min_gap: Minimum allowed gap between notes

        Returns:
            Filtered list with fast subdivisions removed
        """
        if not times:
            return []

        result = [times[0]]

        for t in times[1:]:
            # Check gap from last kept note
            if t - result[-1] >= min_gap:
                result.append(t)
            # If gap is too small, skip this note (simpler rhythm)

        return result

    def _snap_to_strong_beats(self, times: List[float],
                               beats: List[Beat],
                               tolerance: float) -> List[float]:
        """
        Move off-beat notes to nearest strong beat within tolerance.

        Args:
            times: List of times to potentially snap
            beats: Beat data with strength information
            tolerance: Maximum distance to snap

        Returns:
            List with syncopations snapped to strong beats
        """
        if not times or not beats:
            return times

        strong_beats = [b.time for b in beats if b.is_strong]
        if not strong_beats:
            return times

        result = []
        for t in times:
            # Find nearest strong beat
            nearest = min(strong_beats, key=lambda b: abs(b - t))
            distance = abs(nearest - t)

            if distance <= tolerance and distance > 0.01:
                # Snap to strong beat
                result.append(nearest)
            else:
                # Keep original
                result.append(t)

        return result

    def _thin_dense_clusters(self, times: List[float], min_gap: float,
                              preserve_times: set) -> List[float]:
        """
        Remove notes from dense clusters while preserving important positions.

        Args:
            times: List of times to thin
            min_gap: Minimum gap to enforce
            preserve_times: Times that must be kept

        Returns:
            Thinned list with minimum gaps enforced
        """
        if not times:
            return []

        times = sorted(times)
        result = []

        for t in times:
            # Always keep preserved times
            if t in preserve_times:
                result.append(t)
                continue

            # Check if this would be too close to last kept note
            if result:
                gap = t - result[-1]
                if gap < min_gap:
                    continue  # Skip this note

            result.append(t)

        return result

    def simplify_pattern_complexity(self, pattern_type: str,
                                     simplification_level: float) -> str:
        """
        Simplify pattern types for easier difficulties.

        Args:
            pattern_type: Original pattern type
            simplification_level: 0.0 to 1.0

        Returns:
            Simplified pattern type
        """
        if simplification_level <= 0.3:
            # Expert range - allow all patterns
            return pattern_type

        if simplification_level <= 0.6:
            # Intermediate range - no drills, simplify crossovers
            if pattern_type == 'drill':
                return 'stream'
            return pattern_type

        # Beginner range - mostly singles
        if pattern_type in ('drill', 'stream', 'crossover'):
            return 'single'
        if pattern_type == 'gallop':
            return 'single'

        return pattern_type

    @staticmethod
    def get_simplification_for_difficulty(difficulty: str) -> float:
        """
        Get recommended simplification level for a difficulty.

        Args:
            difficulty: Difficulty name

        Returns:
            Simplification level (0.0 to 1.0)
        """
        levels = {
            'beginner': 1.0,
            'intermediate': 0.5,
            'expert': 0.0,
            'insane': 0.0  # No simplification - literal rhythm reproduction
        }
        return levels.get(difficulty, 0.5)

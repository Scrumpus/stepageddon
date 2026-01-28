"""
Step Generator

Core chart generation algorithm. Takes analyzed audio data and generates
a playable step chart according to difficulty constraints.

Enhanced with:
- Phrase-aware pattern switching
- Contour-aware arrow selection
- Musical rest preservation
- Gallop and drill patterns
"""

import logging
from typing import List, Optional

from .schemas import (
    Beat, EnergySection, SustainedNote, SongStructure, Step, Chart,
    Direction, StepType, DifficultyConfig, PitchFrame, PhraseBoundary,
    ContourDirection
)
from .patterns import PatternTemplate
from .audio_analysis import detect_energy_peaks
from .contour import ContourArrowMapper

logger = logging.getLogger(__name__)


class StepGenerator:
    """Main step generation engine."""

    def __init__(self, config: DifficultyConfig):
        """
        Initialize generator with difficulty configuration.

        Args:
            config: DifficultyConfig object defining chart constraints
        """
        self.config = config

    def generate_chart(self,
                      beats: List[Beat],
                      subdivisions: List[float],
                      energy_sections: List[EnergySection],
                      sustained_notes: List[SustainedNote],
                      structure: SongStructure,
                      tempo: float,
                      onset_times: List[float] = None,
                      pitch_frames: List[PitchFrame] = None,
                      phrase_boundaries: List[PhraseBoundary] = None,
                      rest_periods: List[tuple] = None) -> Chart:
        """
        Main chart generation method.

        Args:
            beats: List of detected beats
            subdivisions: List of subdivision times (8th/16th notes)
            energy_sections: List of energy sections
            sustained_notes: List of sustained notes for holds
            structure: Detected song structure
            tempo: Tempo in BPM
            onset_times: Optional list of onset times (for denser charts)
            pitch_frames: Optional pitch tracking data for contour-aware selection
            phrase_boundaries: Optional phrase boundaries for pattern switching
            rest_periods: Optional list of (start, end) tuples for musical rests

        Returns:
            Complete Chart object
        """
        steps = []

        # Store optional data as instance variables for use in helper methods
        self._pitch_frames = pitch_frames or []
        self._phrase_boundaries = phrase_boundaries or []
        self._rest_periods = rest_periods or []

        duration = structure.total_duration
        target_density = (self.config.min_density + self.config.max_density) / 2

        # Use onset times if enabled, otherwise filter beats
        if self.config.use_onsets and onset_times:
            available_beats = list(onset_times)
            logger.info(f"Using {len(available_beats)} onset times as step candidates")
        else:
            available_beats = self._filter_beats(beats)
            logger.info(f"Using {len(available_beats)} filtered beats as step candidates")

        # Filter out beats that fall within rest periods
        if self._rest_periods:
            before_rest_filter = len(available_beats)
            available_beats = self._filter_rest_periods(available_beats)
            logger.info(f"Filtered rests: {before_rest_filter} -> {len(available_beats)} candidates")

        if self.config.use_8th_notes or self.config.use_16th_notes:
            before_count = len(available_beats)
            available_beats.extend(subdivisions)
            # Remove duplicates and sort
            available_beats = sorted(set(available_beats))
            # Re-filter for rests after adding subdivisions
            if self._rest_periods:
                available_beats = self._filter_rest_periods(available_beats)
            logger.info(f"Added subdivisions: {before_count} -> {len(available_beats)} candidates")

        # Phase 1: Place hold notes
        if self.config.allow_holds and sustained_notes:
            hold_steps = self._place_holds(sustained_notes, available_beats)
            steps.extend(hold_steps)

            hold_times = {s.time for s in hold_steps}
            available_beats = [b for b in available_beats if b not in hold_times]

        # Phase 2: Place accents on energy peaks
        energy_peaks = detect_energy_peaks(energy_sections)
        for peak_time in energy_peaks:
            nearest = min(available_beats, key=lambda b: abs(b - peak_time), default=None)
            if nearest and abs(nearest - peak_time) < 0.1:
                if self.config.allow_doubles:
                    step = PatternTemplate.jump_pattern(nearest, 'corners')
                    steps.append(step)
                    available_beats.remove(nearest)

        # Phase 3: Fill with patterns based on energy
        total_section_beats = 0
        total_beats_used = 0
        for section in energy_sections:
            section_beats = [b for b in available_beats
                           if section.start_time <= b <= section.end_time]
            total_section_beats += len(section_beats)

            # When using onsets, use most of them (already filtered by threshold)
            if self.config.use_onsets:
                # Use 80-100% of onsets based on energy (higher energy = more steps)
                base_usage = 0.8
                energy_bonus = section.energy_level * 0.2
                section_target = int(len(section_beats) * (base_usage + energy_bonus))
            else:
                density_multiplier = 1.0 + (section.energy_level - 0.5) * self.config.energy_scale_factor
                section_target = int(len(section_beats) * density_multiplier)
                section_target = min(section_target, len(section_beats))

            beats_to_use = self._select_beats_by_energy(section_beats, beats, section_target)
            total_beats_used += len(beats_to_use)
            section_steps = self._generate_patterns(beats_to_use, section.intensity, structure)
            steps.extend(section_steps)

        logger.info(f"Phase 3: {total_section_beats} section beats -> {total_beats_used} used -> {len(steps)} steps")

        # Sort steps before structure/validation phases
        steps.sort(key=lambda s: s.time)

        # Phase 4: Phrase-aware emphasis
        if self._phrase_boundaries:
            before_phrase = len(steps)
            steps = self._apply_phrase_emphasis(steps)
            logger.info(f"Phase 4 (phrase emphasis): {before_phrase} -> {len(steps)} steps")

        # Phase 5: Structure-aware adjustments
        before_structure = len(steps)
        steps = self._adjust_for_structure(steps, structure)
        logger.info(f"Phase 5 (structure): {before_structure} -> {len(steps)} steps")

        # Phase 6: Validate
        before_validate = len(steps)
        steps = self._validate_chart(steps, tempo)
        logger.info(f"Phase 6 (validate): {before_validate} -> {len(steps)} steps")

        return Chart(
            steps=steps,
            difficulty=self.config.name,
            tempo=tempo,
            duration=duration
        )

    def _filter_beats(self, beats: List[Beat]) -> List[float]:
        """Filter beats based on difficulty configuration."""
        filtered = []

        for beat in beats:
            if beat.beat_type == 'downbeat' and self.config.use_downbeats:
                filtered.append(beat.time)
            elif beat.beat_type == 'upbeat' and self.config.use_upbeats:
                filtered.append(beat.time)
            elif beat.beat_type == 'offbeat' and self.config.use_offbeats:
                filtered.append(beat.time)

        return filtered

    def _place_holds(self, sustained_notes: List[SustainedNote],
                     available_beats: List[float]) -> List[Step]:
        """Place hold notes on sustained melodic sections."""
        holds = []

        for note in sustained_notes:
            duration = note.end_time - note.start_time

            if not (self.config.min_hold_duration <= duration <= self.config.max_hold_duration):
                continue

            nearest_beat = min(available_beats,
                             key=lambda b: abs(b - note.start_time),
                             default=None)

            if nearest_beat and abs(nearest_beat - note.start_time) < 0.2:
                arrow = self._pitch_to_arrow(note.pitch)

                holds.append(Step(
                    time=nearest_beat,
                    arrows=[arrow],
                    step_type=StepType.HOLD,
                    hold_duration=min(duration, self.config.max_hold_duration)
                ))

        max_holds = int(len(available_beats) * self.config.hold_percentage)
        return holds[:max_holds]

    def _pitch_to_arrow(self, pitch: float) -> Direction:
        """Map pitch to arrow direction (deterministic)."""
        if pitch < 60:
            return Direction.LEFT if pitch < 55 else Direction.DOWN
        else:
            return Direction.UP if pitch < 70 else Direction.RIGHT

    def _select_beats_by_energy(self, section_beats: List[float],
                                all_beats: List[Beat],
                                target_count: int) -> List[float]:
        """Select strongest beats from a section."""
        # When using onsets, use all of them (they're already filtered by threshold)
        if self.config.use_onsets:
            return sorted(section_beats)[:target_count]

        beat_strengths = {b.time: b.strength for b in all_beats}

        sorted_beats = sorted(section_beats,
                            key=lambda t: beat_strengths.get(t, 0),
                            reverse=True)

        return sorted_beats[:target_count]

    def _generate_patterns(self, beat_times: List[float],
                          intensity: str,
                          structure: SongStructure) -> List[Step]:
        """
        Generate step patterns for given beats with musical flow.

        Enhanced with:
        - Contour-aware arrow selection
        - Gallop and drill patterns
        - Rest-aware generation
        """
        steps = []
        beat_times = sorted(beat_times)  # Ensure sorted
        i = 0

        while i < len(beat_times):
            time = beat_times[i]

            pattern_choice = self._choose_pattern(time, intensity, structure, steps)

            if pattern_choice == 'drill' and self.config.allow_drills:
                # Drill: rapid same-arrow repeats
                max_drill = min(self.config.max_drill_length, len(beat_times) - i)
                if max_drill >= 4:
                    # Check if beats are evenly spaced for a drill
                    intervals = [beat_times[i+j+1] - beat_times[i+j] for j in range(min(4, max_drill-1))]
                    if intervals and max(intervals) - min(intervals) < 0.03:  # Tighter tolerance for drills
                        drill_length = min(max_drill, 8)  # Cap drill length
                        arrow = self._choose_single_arrow_with_contour(steps, time)
                        drill_steps = PatternTemplate.drill(time, drill_length, intervals[0], arrow)
                        # Map times to actual beat times
                        for j, drill_step in enumerate(drill_steps):
                            if i + j < len(beat_times):
                                drill_step.time = beat_times[i + j]
                                steps.append(drill_step)
                        i += drill_length
                        continue

                # Fall back to single if drill not viable
                arrow = self._choose_single_arrow_with_contour(steps, time)
                steps.append(Step(time=time, arrows=[arrow], step_type=StepType.TAP))
                i += 1

            elif pattern_choice == 'gallop' and self.config.allow_gallops:
                # Gallop: quick same-foot double
                if i + 1 < len(beat_times):
                    interval = beat_times[i + 1] - time
                    # Gallops work best with short intervals (1/8 to 1/16 note)
                    if interval < 0.3:  # Roughly up to an 8th note at 100 BPM
                        arrow = self._choose_single_arrow_with_contour(steps, time)
                        gallop_steps = PatternTemplate.gallop(time, interval, arrow)
                        gallop_steps[0].time = time
                        gallop_steps[1].time = beat_times[i + 1]
                        steps.extend(gallop_steps)
                        i += 2
                        continue

                # Fall back to single if gallop not viable
                arrow = self._choose_single_arrow_with_contour(steps, time)
                steps.append(Step(time=time, arrows=[arrow], step_type=StepType.TAP))
                i += 1

            elif pattern_choice == 'stream' and i + 4 < len(beat_times):
                # Check if beats are evenly spaced (actual stream)
                intervals = [beat_times[i+j+1] - beat_times[i+j] for j in range(min(4, len(beat_times)-i-1))]
                if intervals and max(intervals) - min(intervals) < 0.05:
                    stream_length = min(self.config.max_stream_length, len(beat_times) - i)
                    interval = intervals[0]

                    # Generate stream with contour-aware foot alternation
                    for j in range(stream_length):
                        arrow = self._choose_single_arrow_with_contour(steps, beat_times[i + j])
                        steps.append(Step(time=beat_times[i + j], arrows=[arrow], step_type=StepType.TAP))
                    i += stream_length
                else:
                    # Not a real stream, just do single
                    arrow = self._choose_single_arrow_with_contour(steps, time)
                    steps.append(Step(time=time, arrows=[arrow], step_type=StepType.TAP))
                    i += 1

            elif pattern_choice == 'jump' and self.config.allow_doubles:
                # Use foot-appropriate jumps
                seed = int(time * 100) % 3
                if seed == 0:
                    arrows = [Direction.LEFT, Direction.RIGHT]  # Wide jump
                elif seed == 1:
                    arrows = [Direction.DOWN, Direction.UP]  # Vertical jump
                else:
                    arrows = [Direction.LEFT, Direction.UP]  # Diagonal
                steps.append(Step(time=time, arrows=arrows, step_type=StepType.TAP))
                i += 1

            elif pattern_choice == 'crossover' and self.config.allow_crossovers:
                if i + 3 < len(beat_times):
                    # L-R-L-R or R-L-R-L pattern
                    crossover_arrows = [Direction.LEFT, Direction.RIGHT, Direction.LEFT, Direction.RIGHT]
                    for j, arrow in enumerate(crossover_arrows):
                        if i + j < len(beat_times):
                            steps.append(Step(time=beat_times[i + j], arrows=[arrow], step_type=StepType.TAP))
                    i += 4
                else:
                    arrow = self._choose_single_arrow_with_contour(steps, time)
                    steps.append(Step(time=time, arrows=[arrow], step_type=StepType.TAP))
                    i += 1
            else:
                # Default: single with contour-aware selection
                arrow = self._choose_single_arrow_with_contour(steps, time)
                steps.append(Step(time=time, arrows=[arrow], step_type=StepType.TAP))
                i += 1

        return steps

    def _choose_pattern(self, time: float, intensity: str,
                       structure: SongStructure, existing_steps: List[Step]) -> str:
        """
        Deterministically choose pattern type.

        Enhanced with:
        - Phrase-aware pattern switching
        - Gallop and drill patterns
        """
        # First, check for phrase-based pattern selection
        previous_pattern = 'single'
        if existing_steps:
            # Infer previous pattern from step structure
            if len(existing_steps) >= 2:
                last_arrows = existing_steps[-1].arrows
                if len(last_arrows) > 1:
                    previous_pattern = 'jump'
                # Could add more sophisticated detection here

        phrase_pattern = self._select_pattern_for_phrase(time, intensity, previous_pattern)
        if phrase_pattern:
            return phrase_pattern

        # Standard intensity-based selection with new patterns
        seed = int(time * 100) % 100

        if intensity == 'climax':
            if seed < 20 and self.config.allow_drills and self.config.max_drill_length > 0:
                return 'drill'
            elif seed < 45 and self.config.max_stream_length > 0:
                return 'stream'
            elif seed < 90 and self.config.allow_doubles:
                return 'jump'

        elif intensity == 'high':
            if seed < 15 and self.config.allow_gallops:
                return 'gallop'
            elif seed < 30 and self.config.max_stream_length > 0:
                return 'stream'
            elif seed < 60 and self.config.allow_doubles:
                return 'jump'
            elif seed < 80 and self.config.allow_crossovers:
                return 'crossover'

        elif intensity == 'medium':
            if seed < 15 and self.config.allow_gallops:
                return 'gallop'
            elif seed < 35 and self.config.allow_doubles:
                return 'jump'
            elif seed < 55 and self.config.allow_crossovers:
                return 'crossover'

        elif intensity == 'low':
            # Low intensity: mostly singles, occasional gallop for variety
            if seed < 10 and self.config.allow_gallops:
                return 'gallop'

        return 'single'

    def _choose_single_arrow(self, existing_steps: List[Step], current_time: float = 0) -> Direction:
        """
        Choose next arrow for natural dance flow.

        Uses foot alternation patterns that feel natural:
        - Left foot: LEFT, DOWN
        - Right foot: UP, RIGHT

        This creates natural stepping patterns instead of random rotation.
        """
        left_foot = [Direction.LEFT, Direction.DOWN]
        right_foot = [Direction.UP, Direction.RIGHT]

        if not existing_steps:
            # Start with left foot
            return Direction.LEFT

        last_step = existing_steps[-1]
        last_arrow = last_step.arrows[-1] if last_step.arrows else Direction.LEFT

        # Determine which foot was last used
        last_was_left = last_arrow in left_foot

        # Use time to add variety within each foot
        time_seed = int(current_time * 100) % 100

        if last_was_left:
            # Switch to right foot
            return right_foot[0] if time_seed < 60 else right_foot[1]
        else:
            # Switch to left foot
            return left_foot[0] if time_seed < 60 else left_foot[1]

    def _get_onset_strength(self, time: float, onset_times: List[float], onset_env: any = None) -> float:
        """Get relative strength of an onset (0-1)."""
        if not onset_times:
            return 0.5

        # Find nearest onset
        if time in onset_times:
            idx = onset_times.index(time)
            # Earlier in list = stronger (they're sorted by detection)
            return 1.0 - (idx / len(onset_times)) * 0.5

        return 0.5

    def _adjust_for_structure(self, steps: List[Step],
                             structure: SongStructure) -> List[Step]:
        """Adjust step density based on song structure."""
        intro_start, intro_end = structure.intro
        steps = [s for s in steps if not (intro_start <= s.time <= intro_end) or
                self._keep_intro_step(s, intro_start, intro_end)]

        outro_start, outro_end = structure.outro
        steps = [s for s in steps if not (outro_start <= s.time <= outro_end) or
                self._keep_outro_step(s, outro_start, outro_end)]

        return steps

    def _keep_intro_step(self, step: Step, intro_start: float, intro_end: float) -> bool:
        """Decide if step should be kept in intro."""
        return int(step.time * 100) % 10 < 3

    def _keep_outro_step(self, step: Step, outro_start: float, outro_end: float) -> bool:
        """Decide if step should be kept in outro."""
        return int(step.time * 100) % 10 < 4

    def _validate_chart(self, steps: List[Step], tempo: float) -> List[Step]:
        """Validate and fix any problematic patterns."""
        validated = []

        for i, step in enumerate(steps):
            if i > 0 and abs(step.time - validated[-1].time) < 0.01:
                continue

            if len(step.arrows) > 4:
                step.arrows = step.arrows[:4]

            if self.config.name == 'beginner' and len(step.arrows) > 1:
                step.arrows = [step.arrows[0]]

            min_gap = self._get_min_gap()
            if i > 0:
                gap = step.time - validated[-1].time
                if gap < min_gap:
                    continue

            if step.step_type == StepType.HOLD:
                if step.hold_duration < self.config.min_hold_duration:
                    step.step_type = StepType.TAP
                    step.hold_duration = None
                elif step.hold_duration > self.config.max_hold_duration:
                    step.hold_duration = self.config.max_hold_duration

            validated.append(step)

        return validated

    def _get_min_gap(self) -> float:
        """Get minimum time gap between steps."""
        gaps = {
            'beginner': 0.35,
            'intermediate': 0.15,
            'expert': 0.08,
            'insane': 0.05  # ~50ms minimum = up to 20 steps/second possible
        }
        return gaps.get(self.config.name, 0.15)

    def _filter_rest_periods(self, times: List[float]) -> List[float]:
        """Remove times that fall within musical rest periods."""
        if not self._rest_periods:
            return times

        filtered = []
        for t in times:
            in_rest = any(start <= t <= end for start, end in self._rest_periods)
            if not in_rest:
                filtered.append(t)
        return filtered

    def _get_phrase_at_time(self, time: float) -> Optional[PhraseBoundary]:
        """Get the nearest phrase boundary before or at this time."""
        if not self._phrase_boundaries:
            return None

        # Find the most recent phrase boundary
        relevant = [p for p in self._phrase_boundaries if p.time <= time]
        if relevant:
            return max(relevant, key=lambda p: p.time)
        return None

    def _is_phrase_boundary(self, time: float, tolerance: float = 0.1) -> bool:
        """Check if this time is at or near a phrase boundary."""
        if not self._phrase_boundaries:
            return False

        for phrase in self._phrase_boundaries:
            if abs(phrase.time - time) <= tolerance:
                return True
        return False

    def _is_major_phrase_boundary(self, time: float, tolerance: float = 0.1) -> bool:
        """Check if this time is at or near a major phrase boundary."""
        if not self._phrase_boundaries:
            return False

        for phrase in self._phrase_boundaries:
            if phrase.is_major and abs(phrase.time - time) <= tolerance:
                return True
        return False

    def _apply_phrase_emphasis(self, steps: List[Step]) -> List[Step]:
        """
        Emphasize phrase boundaries in the step chart.

        At phrase boundaries:
        1. Place jump on phrase downbeat (if doubles allowed)
        2. Ensure a brief rest before major phrase starts
        3. Pattern type changes are handled in _select_pattern_for_phrase
        """
        if not self._phrase_boundaries:
            return steps

        modified_steps = []
        step_times = {s.time for s in steps}

        for step in steps:
            # Check if this is a phrase boundary
            if self._is_phrase_boundary(step.time):
                # Upgrade to jump at phrase boundaries if allowed
                if self.config.allow_doubles and len(step.arrows) == 1:
                    # Only for major boundaries or strong positions
                    if self._is_major_phrase_boundary(step.time):
                        # Create a jump
                        seed = int(step.time * 100) % 3
                        if seed == 0:
                            step.arrows = [Direction.LEFT, Direction.RIGHT]
                        elif seed == 1:
                            step.arrows = [Direction.DOWN, Direction.UP]
                        else:
                            step.arrows = [Direction.LEFT, Direction.UP]

            modified_steps.append(step)

        return modified_steps

    def _select_pattern_for_phrase(self, time: float, intensity: str,
                                    previous_pattern: str) -> str:
        """
        Select pattern type based on phrase position.

        - Phrase start: prefer new pattern type (contrast)
        - Within phrase: maintain consistency
        - High intensity phrases: more complex patterns
        """
        at_boundary = self._is_phrase_boundary(time)
        is_major = self._is_major_phrase_boundary(time)

        # At phrase boundaries, prefer to switch pattern type
        if at_boundary:
            seed = int(time * 100) % 100

            # Major boundaries get more emphasis
            if is_major:
                # Prefer jumps or new patterns at major boundaries
                if seed < 60 and self.config.allow_doubles:
                    return 'jump'
                elif seed < 80 and self.config.max_stream_length > 0:
                    return 'stream'
                else:
                    return 'single'

            # Minor boundaries: try to differ from previous pattern
            if previous_pattern == 'single':
                if seed < 40 and self.config.allow_doubles:
                    return 'jump'
                elif seed < 70 and self.config.allow_crossovers:
                    return 'crossover'
            elif previous_pattern == 'jump':
                if seed < 50 and self.config.max_stream_length > 0:
                    return 'stream'
                return 'single'
            elif previous_pattern == 'stream':
                if seed < 50 and self.config.allow_doubles:
                    return 'jump'
                return 'single'

        # Within phrase: use intensity-based selection (existing logic)
        return None  # None means use default intensity-based selection

    def _choose_single_arrow_with_contour(self, existing_steps: List[Step],
                                           current_time: float) -> Direction:
        """
        Choose arrow using melodic contour when available.

        Falls back to standard foot alternation if no pitch data.
        """
        if not self._pitch_frames or not self.config.use_contour:
            return self._choose_single_arrow(existing_steps, current_time)

        # Get contour direction for this moment
        lookback = 0.5  # Look back 500ms for context
        start_time = max(0, current_time - lookback)

        contour = ContourArrowMapper.get_contour_direction(
            self._pitch_frames, start_time, current_time
        )

        # Get last arrow for foot alternation
        last_arrow = None
        if existing_steps:
            last_arrow = existing_steps[-1].arrows[-1] if existing_steps[-1].arrows else None

        time_seed = int(current_time * 100) % 100

        return ContourArrowMapper.select_arrow_for_contour(
            contour, last_arrow, time_seed=time_seed
        )

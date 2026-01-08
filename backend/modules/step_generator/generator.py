"""
Step Generator

Core chart generation algorithm. Takes analyzed audio data and generates
a playable step chart according to difficulty constraints.

Uses musical source analysis (drums, bass, melody) to create charts that
follow the rhythm of the song more naturally.
"""

import logging
from typing import List, Optional

from .schemas import Beat, EnergySection, SustainedNote, SongStructure, Step, Chart, Direction, StepType, DifficultyConfig, StepCandidate
from .patterns import PatternTemplate
from .audio_analysis import detect_energy_peaks

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
                      step_candidates: Optional[List[StepCandidate]] = None) -> Chart:
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
            step_candidates: Optional list of StepCandidate from musical source analysis

        Returns:
            Complete Chart object
        """
        steps = []
        duration = structure.total_duration

        # Use musical source-based generation if step_candidates provided
        if step_candidates:
            logger.info(f"Using {len(step_candidates)} step candidates from musical sources")
            steps = self._generate_from_sources(
                step_candidates, energy_sections, sustained_notes, structure, tempo
            )
        else:
            # Fallback to original beat-based generation
            steps = self._generate_from_beats(
                beats, subdivisions, energy_sections, sustained_notes,
                structure, tempo, onset_times
            )

        # Sort steps before structure/validation phases
        steps.sort(key=lambda s: s.time)

        # Structure-aware adjustments
        before_structure = len(steps)
        steps = self._adjust_for_structure(steps, structure)
        logger.info(f"Structure adjustment: {before_structure} -> {len(steps)} steps")

        # Validate
        before_validate = len(steps)
        steps = self._validate_chart(steps, tempo)
        logger.info(f"Validation: {before_validate} -> {len(steps)} steps")

        return Chart(
            steps=steps,
            difficulty=self.config.name,
            tempo=tempo,
            duration=duration
        )

    def _generate_from_sources(self,
                               step_candidates: List[StepCandidate],
                               energy_sections: List[EnergySection],
                               sustained_notes: List[SustainedNote],
                               structure: SongStructure,
                               tempo: float) -> List[Step]:
        """
        Generate steps from musical source candidates.

        Uses the source-based arrow suggestions for more musical charts.
        """
        steps = []
        used_times = set()
        hold_steps = []

        # Phase 1: Place hold notes first
        if self.config.allow_holds and sustained_notes:
            candidate_times = [c.time for c in step_candidates]
            hold_steps = self._place_holds_with_sources(sustained_notes, step_candidates)
            steps.extend(hold_steps)
            used_times.update(s.time for s in hold_steps)
            logger.info(f"Phase 1 (holds): {len(hold_steps)} hold steps")

        # Phase 2: Filter candidates by energy section and density
        for section in energy_sections:
            section_candidates = [c for c in step_candidates
                                  if section.start_time <= c.time <= section.end_time
                                  and round(c.time, 3) not in used_times]

            if not section_candidates:
                continue

            # Calculate target step count based on energy and difficulty
            base_usage = 0.5 + section.energy_level * 0.4  # 50-90% based on energy
            if self.config.name == 'beginner':
                base_usage *= 0.4  # Much fewer steps for beginner
            elif self.config.name == 'expert':
                base_usage *= 1.2  # More steps for expert

            target_count = int(len(section_candidates) * base_usage)
            target_count = max(1, target_count)

            # Sort by priority and take top candidates
            sorted_candidates = sorted(section_candidates, key=lambda c: -c.priority)
            selected = sorted_candidates[:target_count]

            # Generate steps from selected candidates (avoiding held arrows)
            section_steps = self._candidates_to_steps(selected, section.intensity, hold_steps)
            steps.extend(section_steps)
            used_times.update(round(s.time, 3) for s in section_steps)

        logger.info(f"Phase 2 (sources): {len(steps)} total steps")

        # Phase 3: Add jumps on energy peaks if doubles allowed
        if self.config.allow_doubles:
            energy_peaks = detect_energy_peaks(energy_sections)
            for peak_time in energy_peaks:
                # Find if there's already a step near this peak
                existing = [s for s in steps if abs(s.time - peak_time) < 0.1]
                if existing and len(existing[0].arrows) == 1:
                    # Convert to jump (but not on held arrows)
                    existing_arrow = existing[0].arrows[0]
                    held = self._get_held_arrows_at_time(hold_steps, peak_time)
                    second_arrow = self._get_jump_partner(existing_arrow)
                    if second_arrow not in held:
                        existing[0].arrows.append(second_arrow)

        return steps

    def _place_holds_with_sources(self,
                                  sustained_notes: List[SustainedNote],
                                  step_candidates: List[StepCandidate]) -> List[Step]:
        """Place hold notes using source-aware arrow selection."""
        holds = []
        candidate_map = {round(c.time, 3): c for c in step_candidates}

        for note in sustained_notes:
            duration = note.end_time - note.start_time

            if not (self.config.min_hold_duration <= duration <= self.config.max_hold_duration):
                continue

            # Find nearest candidate
            nearest_time = min(candidate_map.keys(),
                               key=lambda t: abs(t - note.start_time),
                               default=None)

            if nearest_time and abs(nearest_time - note.start_time) < 0.2:
                candidate = candidate_map[nearest_time]
                # Use source-suggested arrow, or fallback to pitch-based
                if candidate.suggested_arrows:
                    arrow = candidate.suggested_arrows[0]
                else:
                    arrow = self._pitch_to_arrow(note.pitch)

                holds.append(Step(
                    time=nearest_time,
                    arrows=[arrow],
                    step_type=StepType.HOLD,
                    hold_duration=min(duration, self.config.max_hold_duration)
                ))

        max_holds = int(len(step_candidates) * self.config.hold_percentage)
        return holds[:max_holds]

    def _candidates_to_steps(self,
                            candidates: List[StepCandidate],
                            intensity: str,
                            hold_steps: List[Step] = None) -> List[Step]:
        """Convert step candidates to actual steps with spread-out arrow selection."""
        steps = []
        prev_arrow = None
        arrow_counts = {d: 0 for d in Direction}
        hold_steps = hold_steps or []

        for candidate in sorted(candidates, key=lambda c: c.time):
            # Get arrows that are currently being held at this time
            held_arrows = self._get_held_arrows_at_time(hold_steps, candidate.time)

            # Choose arrow with good spread and flow, avoiding held arrows
            arrow = self._choose_spread_arrow(prev_arrow, arrow_counts, candidate.time, held_arrows)

            steps.append(Step(
                time=candidate.time,
                arrows=[arrow],
                step_type=StepType.TAP
            ))

            prev_arrow = arrow
            arrow_counts[arrow] += 1

        return steps

    def _get_held_arrows_at_time(self, hold_steps: List[Step], time: float) -> set:
        """Get set of arrows that are being held at a given time."""
        held = set()
        for step in hold_steps:
            if step.step_type == StepType.HOLD and step.hold_duration:
                hold_end = step.time + step.hold_duration
                # Check if time falls within the hold (with small buffer)
                if step.time <= time <= hold_end + 0.05:
                    held.update(step.arrows)
        return held

    def _choose_spread_arrow(self,
                             prev: Direction,
                             counts: dict,
                             current_time: float,
                             held_arrows: set = None) -> Direction:
        """
        Choose arrow that spreads usage across all directions.

        Uses foot alternation (left foot: LEFT/DOWN, right foot: UP/RIGHT)
        while balancing arrow counts for variety.
        Avoids arrows that are currently being held.
        """
        left_foot = [Direction.LEFT, Direction.DOWN]
        right_foot = [Direction.UP, Direction.RIGHT]
        all_arrows = [Direction.LEFT, Direction.DOWN, Direction.UP, Direction.RIGHT]
        held_arrows = held_arrows or set()

        # Filter out held arrows
        available = [d for d in all_arrows if d not in held_arrows]
        if not available:
            # All arrows held (shouldn't happen), fall back to any
            available = all_arrows

        if prev is None:
            # Start with least used available arrow
            return min(available, key=lambda d: counts[d])

        # Determine which foot to use (alternate)
        last_was_left = prev in left_foot
        foot_options = right_foot if last_was_left else left_foot

        # Filter foot options to exclude held arrows
        foot_options = [d for d in foot_options if d not in held_arrows]

        # If no options on preferred foot, use other foot
        if not foot_options:
            foot_options = [d for d in (left_foot if last_was_left else right_foot) if d not in held_arrows]

        # If still no options, use any available
        if not foot_options:
            foot_options = available

        if len(foot_options) == 1:
            return foot_options[0]

        # Pick the less-used arrow from this foot's options
        # Add some time-based variation
        time_seed = int(current_time * 100) % 100

        if counts[foot_options[0]] < counts[foot_options[1]]:
            # First option is less used
            return foot_options[0] if time_seed < 70 else foot_options[1]
        elif counts[foot_options[1]] < counts[foot_options[0]]:
            # Second option is less used
            return foot_options[1] if time_seed < 70 else foot_options[0]
        else:
            # Equal counts - use time for variety
            return foot_options[0] if time_seed < 50 else foot_options[1]

    def _get_jump_partner(self, arrow: Direction) -> Direction:
        """Get a good partner arrow for a jump."""
        # Prefer comfortable jumps
        partners = {
            Direction.LEFT: Direction.RIGHT,  # Wide jump
            Direction.RIGHT: Direction.LEFT,
            Direction.DOWN: Direction.UP,     # Vertical jump
            Direction.UP: Direction.DOWN,
        }
        return partners.get(arrow, Direction.DOWN)

    def _generate_from_beats(self,
                            beats: List[Beat],
                            subdivisions: List[float],
                            energy_sections: List[EnergySection],
                            sustained_notes: List[SustainedNote],
                            structure: SongStructure,
                            tempo: float,
                            onset_times: List[float] = None) -> List[Step]:
        """
        Original beat-based generation (fallback when no step_candidates).
        """
        steps = []

        # Use onset times if enabled, otherwise filter beats
        if self.config.use_onsets and onset_times:
            available_beats = list(onset_times)
            logger.info(f"Using {len(available_beats)} onset times as step candidates")
        else:
            available_beats = self._filter_beats(beats)
            logger.info(f"Using {len(available_beats)} filtered beats as step candidates")

        if self.config.use_8th_notes or self.config.use_16th_notes:
            before_count = len(available_beats)
            available_beats.extend(subdivisions)
            available_beats = sorted(set(available_beats))
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

            if self.config.use_onsets:
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

        logger.info(f"Beat-based generation: {total_section_beats} beats -> {total_beats_used} used -> {len(steps)} steps")

        return steps

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
        """Generate step patterns for given beats with musical flow."""
        steps = []
        beat_times = sorted(beat_times)  # Ensure sorted
        i = 0

        while i < len(beat_times):
            time = beat_times[i]

            pattern_choice = self._choose_pattern(time, intensity, structure, steps)

            if pattern_choice == 'stream' and i + 4 < len(beat_times):
                # Check if beats are evenly spaced (actual stream)
                intervals = [beat_times[i+j+1] - beat_times[i+j] for j in range(min(4, len(beat_times)-i-1))]
                if intervals and max(intervals) - min(intervals) < 0.05:
                    stream_length = min(self.config.max_stream_length, len(beat_times) - i)
                    interval = intervals[0]

                    # Generate stream with proper foot alternation
                    for j in range(stream_length):
                        arrow = self._choose_single_arrow(steps, time + j * interval)
                        steps.append(Step(time=beat_times[i + j], arrows=[arrow], step_type=StepType.TAP))
                    i += stream_length
                else:
                    # Not a real stream, just do single
                    arrow = self._choose_single_arrow(steps, time)
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
                    arrow = self._choose_single_arrow(steps, time)
                    steps.append(Step(time=time, arrows=[arrow], step_type=StepType.TAP))
                    i += 1
            else:
                arrow = self._choose_single_arrow(steps, time)
                steps.append(Step(time=time, arrows=[arrow], step_type=StepType.TAP))
                i += 1

        return steps

    def _choose_pattern(self, time: float, intensity: str,
                       structure: SongStructure, existing_steps: List[Step]) -> str:
        """Deterministically choose pattern type."""
        seed = int(time * 100) % 100

        if intensity == 'climax':
            if seed < 40 and self.config.max_stream_length > 0:
                return 'stream'
            elif seed < 90 and self.config.allow_doubles:
                return 'jump'

        elif intensity == 'high':
            if seed < 25 and self.config.max_stream_length > 0:
                return 'stream'
            elif seed < 60 and self.config.allow_doubles:
                return 'jump'
            elif seed < 80 and self.config.allow_crossovers:
                return 'crossover'

        elif intensity == 'medium':
            if seed < 30 and self.config.allow_doubles:
                return 'jump'
            elif seed < 50 and self.config.allow_crossovers:
                return 'crossover'

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
            'expert': 0.08
        }
        return gaps.get(self.config.name, 0.15)

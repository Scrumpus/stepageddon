"""
Step Generator

Core chart generation algorithm. Takes analyzed audio data and generates
a playable step chart according to difficulty constraints.
"""

import logging
from typing import List

from .schemas import Beat, EnergySection, SustainedNote, SongStructure, Step, Chart, Direction, StepType, DifficultyConfig
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
                      onset_times: List[float] = None) -> Chart:
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

        Returns:
            Complete Chart object
        """
        steps = []

        duration = structure.total_duration
        target_density = (self.config.min_density + self.config.max_density) / 2

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
            # Remove duplicates and sort
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

        # Phase 4: Structure-aware adjustments
        before_structure = len(steps)
        steps = self._adjust_for_structure(steps, structure)
        logger.info(f"Phase 4 (structure): {before_structure} -> {len(steps)} steps")

        # Phase 5: Validate
        before_validate = len(steps)
        steps = self._validate_chart(steps, tempo)
        logger.info(f"Phase 5 (validate): {before_validate} -> {len(steps)} steps")

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
        """Generate step patterns for given beats."""
        steps = []
        i = 0

        while i < len(beat_times):
            time = beat_times[i]

            pattern_choice = self._choose_pattern(time, intensity, structure, steps)

            if pattern_choice == 'stream' and i + 4 < len(beat_times):
                stream_length = min(self.config.max_stream_length, len(beat_times) - i)
                interval = beat_times[i + 1] - beat_times[i] if i + 1 < len(beat_times) else 0.25

                stream_steps = PatternTemplate.single_stream(
                    time, stream_length, interval, Direction.LEFT
                )
                steps.extend(stream_steps)
                i += stream_length

            elif pattern_choice == 'jump' and self.config.allow_doubles:
                jump = PatternTemplate.jump_pattern(time, 'corners')
                steps.append(jump)
                i += 1

            elif pattern_choice == 'crossover' and self.config.allow_crossovers:
                if i + 3 < len(beat_times):
                    interval = beat_times[i + 1] - beat_times[i]
                    crossover = PatternTemplate.crossover_pattern(time, interval)
                    steps.extend(crossover)
                    i += 4
                else:
                    steps.append(Step(time=time, arrows=[Direction.LEFT], step_type=StepType.TAP))
                    i += 1
            else:
                arrow = self._choose_single_arrow(steps)
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

    def _choose_single_arrow(self, existing_steps: List[Step]) -> Direction:
        """Choose next arrow for natural alternation."""
        if not existing_steps:
            return Direction.LEFT

        last_step = existing_steps[-1]
        last_arrow = last_step.arrows[-1] if last_step.arrows else Direction.LEFT

        rotation = {
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.UP: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }

        return rotation[last_arrow]

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

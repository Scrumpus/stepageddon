"""
Step Generator

Core chart generation algorithm. Takes analyzed audio data and generates
a playable step chart according to difficulty constraints.
"""

import logging
from typing import List

from .schemas import Beat, EnergySection, SustainedNote, SongStructure, Step, Chart, Direction, StepType, DifficultyConfig, DrumTrack, WeightedOnset
from .patterns import PatternTemplate
from .audio_analysis import detect_energy_peaks

from typing import Optional

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
                      drum_track: Optional[DrumTrack] = None,
                      weighted_onsets: Optional[List[WeightedOnset]] = None) -> Chart:
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
            drum_track: Optional DrumTrack with kick/snare/hihat events
            weighted_onsets: Optional list of WeightedOnset for strength-based selection

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

        # Phase 2: Place accents on energy peaks AND snare-driven jumps
        energy_peaks = detect_energy_peaks(energy_sections)
        used_times = set()

        # 2a: Energy peak jumps
        for peak_time in energy_peaks:
            nearest = min(available_beats, key=lambda b: abs(b - peak_time), default=None)
            if nearest and abs(nearest - peak_time) < 0.1:
                if self.config.allow_doubles:
                    step = PatternTemplate.jump_pattern(nearest, 'corners')
                    steps.append(step)
                    used_times.add(nearest)
                    available_beats.remove(nearest)

        # 2b: Snare-driven jump placement
        # Snares typically mark emphasized backbeats - perfect for jumps
        snare_jumps_placed = 0
        if drum_track and self.config.allow_doubles:
            for snare in drum_track.snares:
                # Only place jumps on strong snare hits
                if snare.strength < 0.5:
                    continue

                # Find nearest available beat to snare hit
                nearest = min(available_beats, key=lambda b: abs(b - snare.time), default=None)
                if nearest and abs(nearest - snare.time) < 0.05:  # Tighter tolerance for snares
                    # Don't double up on existing jumps
                    if nearest not in used_times:
                        # Alternate jump types for variety
                        seed = int(snare.time * 100) % 3
                        if seed == 0:
                            arrows = [Direction.LEFT, Direction.RIGHT]
                        elif seed == 1:
                            arrows = [Direction.DOWN, Direction.UP]
                        else:
                            arrows = [Direction.LEFT, Direction.UP]

                        steps.append(Step(time=nearest, arrows=arrows, step_type=StepType.TAP))
                        used_times.add(nearest)
                        available_beats.remove(nearest)
                        snare_jumps_placed += 1

                        # Limit snare jumps to avoid overwhelming the chart
                        if snare_jumps_placed >= 10:
                            break

        logger.info(f"Phase 2: {len(energy_peaks)} energy peaks, {snare_jumps_placed} snare jumps placed")

        # Phase 3: Fill with patterns based on energy
        # Use weighted onsets if available for better strength-based selection
        total_section_beats = 0
        total_beats_used = 0

        # Build weighted onset lookup for strength-based selection
        onset_strengths = {}
        if weighted_onsets:
            for wo in weighted_onsets:
                onset_strengths[round(wo.time, 3)] = wo.strength
                # Boost drum-correlated onsets
                if wo.is_drum:
                    onset_strengths[round(wo.time, 3)] += 0.2

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

            # Use weighted onset selection if available, otherwise fall back to beat energy
            if weighted_onsets and onset_strengths:
                beats_to_use = self._select_beats_by_onset_strength(section_beats, onset_strengths, section_target)
            else:
                beats_to_use = self._select_beats_by_energy(section_beats, beats, section_target)

            total_beats_used += len(beats_to_use)
            section_steps = self._generate_patterns(beats_to_use, section.intensity, structure, drum_track)
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

    def _select_beats_by_onset_strength(self, section_beats: List[float],
                                        onset_strengths: dict,
                                        target_count: int) -> List[float]:
        """
        Select beats by onset strength (weighted onset selection).

        Prioritizes beats that align with strong onsets, especially
        those correlated with drum hits.

        Args:
            section_beats: Candidate beat times in this section
            onset_strengths: Dict mapping time -> strength (with drum boost)
            target_count: Number of beats to select

        Returns:
            List of selected beat times, sorted by time
        """
        def get_strength(beat_time: float) -> float:
            # Check exact match first
            rounded = round(beat_time, 3)
            if rounded in onset_strengths:
                return onset_strengths[rounded]

            # Check nearby onsets (within 30ms)
            for t, s in onset_strengths.items():
                if abs(t - beat_time) < 0.03:
                    return s

            return 0.0  # No matching onset

        # Sort by strength (descending), then by time for stability
        sorted_beats = sorted(
            section_beats,
            key=lambda t: (get_strength(t), -t),
            reverse=True
        )

        # Take top N by strength, then sort by time for playback order
        selected = sorted_beats[:target_count]
        return sorted(selected)

    def _generate_patterns(self, beat_times: List[float],
                          intensity: str,
                          structure: SongStructure,
                          drum_track: Optional[DrumTrack] = None) -> List[Step]:
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

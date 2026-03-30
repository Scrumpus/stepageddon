"""
Step Generator

Core chart generation algorithm. Takes analyzed audio data and generates
a playable step chart according to difficulty constraints.

3-phase pipeline:
  Phase 1: Build candidate times from beats + onsets
  Phase 2: Place steps in a single chronological pass
  Phase 3: Validate and fix the chart
"""

import logging
from typing import List, Optional

from .schemas import (
    Beat, EnergySection, Step, Chart,
    Direction, StepType, DifficultyConfig, BeatSubdivision
)

logger = logging.getLogger(__name__)


class StepGenerator:
    """Main step generation engine."""

    def __init__(self, config: DifficultyConfig):
        self.config = config

    def generate_chart(self,
                       beats: List[Beat],
                       onsets: List[float],
                       energy_sections: List[EnergySection],
                       tempo: float,
                       duration: float) -> Chart:
        """
        Generate a complete step chart.

        Args:
            beats: Detected beats with classification
            onsets: Quantized onset times (empty if use_onsets is False)
            energy_sections: Energy profile sections
            tempo: Tempo in BPM
            duration: Song duration in seconds

        Returns:
            Complete Chart object
        """
        # Phase 1: Build candidate times
        candidates = self._build_candidates(beats, onsets)
        logger.info(f"Phase 1: {len(candidates)} candidates")

        # Phase 2: Place steps
        steps = self._place_steps(candidates, energy_sections, beats, duration)
        logger.info(f"Phase 2: {len(steps)} steps placed")

        # Phase 3: Validate
        steps = self._validate_chart(steps, duration)
        logger.info(f"Phase 3: {len(steps)} steps after validation")

        # Classify beat subdivisions
        for step in steps:
            step.beat_subdivision = self._classify_subdivision(step.time, tempo)

        return Chart(
            steps=steps,
            difficulty=self.config.name,
            tempo=tempo,
            duration=duration
        )

    # ── Phase 1: Build Candidates ──────────────────────────────────────

    def _build_candidates(self, beats: List[Beat], onsets: List[float]) -> List[float]:
        """Build a single sorted list of candidate step times."""
        candidates = set()

        # Filter beats by grid resolution
        for beat in beats:
            if self.config.grid_resolution == 4:
                # Quarter notes: downbeats only
                if beat.measure_position == 0:
                    candidates.add(beat.time)
            else:
                # 8th/16th: all beats from beat_track
                candidates.add(beat.time)

        # Merge onsets if enabled
        if self.config.use_onsets and onsets:
            for t in onsets:
                candidates.add(round(t, 4))

        return sorted(candidates)

    # ── Phase 2: Place Steps ───────────────────────────────────────────

    def _place_steps(self, candidates: List[float],
                     energy_sections: List[EnergySection],
                     beats: List[Beat],
                     duration: float) -> List[Step]:
        """Place steps in a single chronological pass over candidates."""
        steps: List[Step] = []
        downbeat_times = [b.time for b in beats if b.beat_type == 'downbeat']
        downbeat_set = set(downbeat_times)

        i = 0
        while i < len(candidates):
            time = candidates[i]

            # Gap check
            if steps and time - steps[-1].time < self.config.min_gap:
                i += 1
                continue

            # Energy lookup
            section = self._find_energy_section(time, energy_sections)
            energy_level = section.energy_level if section else 0.5
            intensity = section.intensity if section else 'medium'

            # Rest check: skip very low energy regions
            if energy_level < 0.1:
                i += 1
                continue

            # Density check
            recent = sum(1 for s in steps if s.time > time - 2.0)
            local_density = recent / 2.0
            allowed_density = self.config.min_density + (
                (self.config.max_density - self.config.min_density)
                * energy_level * self.config.energy_scale_factor
            )
            allowed_density = min(allowed_density, self.config.max_density)

            if local_density >= allowed_density:
                i += 1
                continue

            # Phrase boundary detection
            is_phrase = False
            is_major_phrase = False
            if time in downbeat_set:
                db_idx = downbeat_times.index(time)
                if db_idx > 0:
                    is_phrase = (db_idx % 4 == 0)
                    is_major_phrase = (db_idx % 8 == 0)

            # Choose pattern
            pattern = self._choose_pattern(time, intensity, is_phrase, is_major_phrase)

            # ── Multi-candidate patterns ──

            if pattern == 'stream' and i + 4 < len(candidates):
                consumed = self._try_stream(candidates, i, steps)
                if consumed > 1:
                    i += consumed
                    continue

            if pattern == 'crossover' and i + 3 < len(candidates):
                consumed = self._try_crossover(candidates, i, steps)
                if consumed > 1:
                    i += consumed
                    continue

            if pattern == 'gallop' and i + 1 < len(candidates):
                consumed = self._try_gallop(candidates, i, steps)
                if consumed > 1:
                    i += consumed
                    continue

            if pattern == 'drill' and i + 3 < len(candidates):
                consumed = self._try_drill(candidates, i, steps)
                if consumed > 1:
                    i += consumed
                    continue

            # ── Single step (default + jump + hold) ──

            if pattern == 'jump' and 'jump' in self.config.allowed_patterns:
                arrows = self._choose_jump_arrows(time)
                steps.append(Step(time=time, arrows=arrows, step_type=StepType.TAP))
            else:
                # Single tap or hold
                arrow = self._choose_single_arrow(steps, time)
                if self._should_hold(candidates, i, energy_sections, steps):
                    hold_dur = candidates[i + 1] - time
                    hold_dur = min(hold_dur, self.config.max_hold_duration)
                    steps.append(Step(
                        time=time, arrows=[arrow],
                        step_type=StepType.HOLD, hold_duration=hold_dur
                    ))
                else:
                    steps.append(Step(time=time, arrows=[arrow], step_type=StepType.TAP))

            i += 1

        return steps

    # ── Multi-candidate pattern helpers ────────────────────────────────

    def _try_stream(self, candidates: List[float], start: int,
                    steps: List[Step]) -> int:
        """Try to place a stream. Returns number of candidates consumed (0 if failed)."""
        max_len = min(self.config.max_stream_length, len(candidates) - start)
        if max_len < 4:
            return 0

        # Check even spacing
        base_interval = candidates[start + 1] - candidates[start]
        actual_len = 1
        for j in range(1, max_len):
            interval = candidates[start + j] - candidates[start + j - 1]
            if abs(interval - base_interval) > 0.05:
                break
            actual_len += 1

        if actual_len < 4:
            return 0

        for j in range(actual_len):
            arrow = self._choose_single_arrow(steps, candidates[start + j])
            steps.append(Step(time=candidates[start + j], arrows=[arrow], step_type=StepType.TAP))

        return actual_len

    def _try_crossover(self, candidates: List[float], start: int,
                       steps: List[Step]) -> int:
        """Try to place a crossover (L-R-L-R). Returns candidates consumed."""
        crossover_arrows = [Direction.LEFT, Direction.RIGHT, Direction.LEFT, Direction.RIGHT]
        count = min(4, len(candidates) - start)
        for j in range(count):
            steps.append(Step(
                time=candidates[start + j],
                arrows=[crossover_arrows[j]],
                step_type=StepType.TAP
            ))
        return count

    def _try_gallop(self, candidates: List[float], start: int,
                    steps: List[Step]) -> int:
        """Try to place a gallop (quick double). Returns candidates consumed."""
        interval = candidates[start + 1] - candidates[start]
        if interval >= 0.3:
            return 0

        arrow = self._choose_single_arrow(steps, candidates[start])
        steps.append(Step(time=candidates[start], arrows=[arrow], step_type=StepType.TAP))
        steps.append(Step(time=candidates[start + 1], arrows=[arrow], step_type=StepType.TAP))
        return 2

    def _try_drill(self, candidates: List[float], start: int,
                   steps: List[Step]) -> int:
        """Try to place a drill (rapid same-arrow repeats). Returns candidates consumed."""
        max_len = min(self.config.max_drill_length, len(candidates) - start)
        if max_len < 4:
            return 0

        base_interval = candidates[start + 1] - candidates[start]
        actual_len = 1
        for j in range(1, max_len):
            interval = candidates[start + j] - candidates[start + j - 1]
            if abs(interval - base_interval) > 0.03:
                break
            actual_len += 1

        if actual_len < 4:
            return 0

        arrow = self._choose_single_arrow(steps, candidates[start])
        for j in range(actual_len):
            steps.append(Step(time=candidates[start + j], arrows=[arrow], step_type=StepType.TAP))

        return actual_len

    # ── Hold detection ─────────────────────────────────────────────────

    def _should_hold(self, candidates: List[float], idx: int,
                     energy_sections: List[EnergySection],
                     steps: List[Step]) -> bool:
        """Decide if this candidate should be a hold note."""
        if self.config.hold_percentage <= 0:
            return False
        if idx + 1 >= len(candidates):
            return False

        gap = candidates[idx + 1] - candidates[idx]
        if not (self.config.min_hold_duration <= gap <= self.config.max_hold_duration):
            return False

        # Check energy stays up during the hold
        mid_time = candidates[idx] + gap / 2
        mid_section = self._find_energy_section(mid_time, energy_sections)
        if mid_section and mid_section.energy_level < 0.3:
            return False

        # Check current hold ratio
        current_holds = sum(1 for s in steps if s.step_type == StepType.HOLD)
        current_ratio = current_holds / max(len(steps), 1)
        if current_ratio >= self.config.hold_percentage:
            return False

        # Deterministic selection via time seed
        seed = int(candidates[idx] * 1000) % 100
        return seed < self.config.hold_percentage * 100

    # ── Pattern choice ─────────────────────────────────────────────────

    def _choose_pattern(self, time: float, intensity: str,
                        is_phrase: bool, is_major_phrase: bool) -> str:
        """Deterministically choose pattern type."""
        allowed = self.config.allowed_patterns
        seed = int(time * 100) % 100

        # Major phrase boundaries: prefer jumps
        if is_major_phrase:
            if seed < 60 and 'jump' in allowed:
                return 'jump'
            if seed < 80 and 'stream' in allowed:
                return 'stream'
            return 'single'

        # Minor phrase boundaries: switch things up
        if is_phrase:
            if seed < 40 and 'jump' in allowed:
                return 'jump'
            if seed < 60 and 'crossover' in allowed:
                return 'crossover'
            if seed < 80 and 'stream' in allowed:
                return 'stream'
            return 'single'

        # Intensity-based selection
        if intensity == 'climax':
            if seed < 20 and 'drill' in allowed:
                return 'drill'
            if seed < 45 and 'stream' in allowed:
                return 'stream'
            if seed < 90 and 'jump' in allowed:
                return 'jump'

        elif intensity == 'high':
            if seed < 15 and 'gallop' in allowed:
                return 'gallop'
            if seed < 30 and 'stream' in allowed:
                return 'stream'
            if seed < 60 and 'jump' in allowed:
                return 'jump'
            if seed < 80 and 'crossover' in allowed:
                return 'crossover'

        elif intensity == 'medium':
            if seed < 15 and 'gallop' in allowed:
                return 'gallop'
            if seed < 35 and 'jump' in allowed:
                return 'jump'
            if seed < 55 and 'crossover' in allowed:
                return 'crossover'

        elif intensity == 'low':
            if seed < 10 and 'gallop' in allowed:
                return 'gallop'

        return 'single'

    # ── Arrow selection ────────────────────────────────────────────────

    def _choose_single_arrow(self, existing_steps: List[Step],
                             current_time: float) -> Direction:
        """
        Choose next arrow using foot alternation.

        Left foot: LEFT, DOWN
        Right foot: UP, RIGHT
        """
        left_foot = [Direction.LEFT, Direction.DOWN]
        right_foot = [Direction.UP, Direction.RIGHT]

        if not existing_steps:
            return Direction.LEFT

        last_arrow = existing_steps[-1].arrows[-1] if existing_steps[-1].arrows else Direction.LEFT
        last_was_left = last_arrow in left_foot

        time_seed = int(current_time * 100) % 100

        if last_was_left:
            return right_foot[0] if time_seed < 60 else right_foot[1]
        else:
            return left_foot[0] if time_seed < 60 else left_foot[1]

    def _choose_jump_arrows(self, time: float) -> List[Direction]:
        """Choose two arrows for a jump."""
        seed = int(time * 100) % 3
        if seed == 0:
            return [Direction.LEFT, Direction.RIGHT]
        elif seed == 1:
            return [Direction.DOWN, Direction.UP]
        else:
            return [Direction.LEFT, Direction.UP]

    # ── Helpers ────────────────────────────────────────────────────────

    def _find_energy_section(self, time: float,
                             sections: List[EnergySection]) -> Optional[EnergySection]:
        """Find the energy section containing a given time."""
        for s in sections:
            if s.start_time <= time <= s.end_time:
                return s
        return None

    # ── Beat subdivision classification ─────────────────────────────────

    @staticmethod
    def _classify_subdivision(time: float, tempo: float) -> BeatSubdivision:
        """Classify a step time as quarter, eighth, or sixteenth note."""
        beat_duration = 60.0 / tempo
        # Position within the beat (0.0 to 1.0)
        beat_position = (time % beat_duration) / beat_duration
        # Quarter note: lands on the beat (position ~0.0)
        if beat_position < 0.05 or beat_position > 0.95:
            return BeatSubdivision.QUARTER
        # Eighth note: lands on the half-beat (position ~0.5)
        if abs(beat_position - 0.5) < 0.05:
            return BeatSubdivision.EIGHTH
        # Everything else is a sixteenth
        return BeatSubdivision.SIXTEENTH

    # ── Phase 3: Validate ──────────────────────────────────────────────

    def _validate_chart(self, steps: List[Step], duration: float) -> List[Step]:
        """Validate and fix the step chart."""
        steps.sort(key=lambda s: s.time)
        validated = []

        intro_end = duration * 0.10
        outro_start = duration * 0.90

        for step in steps:
            # Deduplicate
            if validated and abs(step.time - validated[-1].time) < 0.01:
                continue

            # Enforce min gap
            if validated and step.time - validated[-1].time < self.config.min_gap:
                continue

            # Cap arrows if jumps not allowed
            if 'jump' not in self.config.allowed_patterns and len(step.arrows) > 1:
                step.arrows = [step.arrows[0]]

            # Fix hold durations
            if step.step_type == StepType.HOLD:
                if step.hold_duration < self.config.min_hold_duration:
                    step.step_type = StepType.TAP
                    step.hold_duration = None
                elif step.hold_duration > self.config.max_hold_duration:
                    step.hold_duration = self.config.max_hold_duration

            # Thin intro (~30% kept)
            if step.time <= intro_end:
                if int(step.time * 100) % 10 >= 3:
                    continue

            # Thin outro (~40% kept)
            if step.time >= outro_start:
                if int(step.time * 100) % 10 >= 4:
                    continue

            validated.append(step)

        return validated

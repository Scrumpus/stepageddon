"""
Pattern Templates

Reusable step pattern generators for creating common chart patterns.
All methods are static and deterministic.
"""

from typing import List
from .schemas import Step, Direction, StepType


class PatternTemplate:
    """Defines reusable step patterns."""

    @staticmethod
    def single_stream(start_time: float, count: int, interval: float,
                     start_arrow: Direction) -> List[Step]:
        """
        Generate a stream of single arrows.

        Args:
            start_time: Starting time in seconds
            count: Number of arrows in the stream
            interval: Time between arrows
            start_arrow: Starting arrow direction

        Returns:
            List of Step objects forming a stream pattern
        """
        arrows_cycle = [Direction.LEFT, Direction.DOWN, Direction.UP, Direction.RIGHT]
        start_idx = arrows_cycle.index(start_arrow)

        steps = []
        for i in range(count):
            arrow = arrows_cycle[(start_idx + i) % 4]
            steps.append(Step(
                time=start_time + i * interval,
                arrows=[arrow],
                step_type=StepType.TAP
            ))
        return steps

    @staticmethod
    def crossover_pattern(start_time: float, interval: float) -> List[Step]:
        """
        Generate a crossover pattern (left-right alternation).

        Args:
            start_time: Starting time in seconds
            interval: Time between steps

        Returns:
            List of 4 Step objects forming a crossover
        """
        return [
            Step(time=start_time, arrows=[Direction.LEFT], step_type=StepType.TAP),
            Step(time=start_time + interval, arrows=[Direction.RIGHT], step_type=StepType.TAP),
            Step(time=start_time + interval * 2, arrows=[Direction.LEFT], step_type=StepType.TAP),
            Step(time=start_time + interval * 3, arrows=[Direction.RIGHT], step_type=StepType.TAP)
        ]

    @staticmethod
    def jump_pattern(start_time: float, jump_type: str) -> Step:
        """
        Generate a jump (2 arrows simultaneously).

        Args:
            start_time: Time for the jump
            jump_type: Type of jump ('corners', 'brackets', 'middle', 'diagonal1', 'diagonal2')

        Returns:
            Step object with 2 arrows
        """
        jumps = {
            'corners': [Direction.LEFT, Direction.DOWN],
            'brackets': [Direction.LEFT, Direction.RIGHT],
            'middle': [Direction.DOWN, Direction.UP],
            'diagonal1': [Direction.LEFT, Direction.UP],
            'diagonal2': [Direction.DOWN, Direction.RIGHT]
        }
        return Step(time=start_time, arrows=jumps.get(jump_type, jumps['corners']), step_type=StepType.TAP)

    @staticmethod
    def hold_note(start_time: float, arrow: Direction, duration: float) -> Step:
        """
        Generate a hold note.

        Args:
            start_time: Time when hold starts
            arrow: Arrow direction for the hold
            duration: Duration of the hold in seconds

        Returns:
            Step object with hold type
        """
        return Step(
            time=start_time,
            arrows=[arrow],
            step_type=StepType.HOLD,
            hold_duration=duration
        )

    @staticmethod
    def gallop(start_time: float, interval: float, arrow: Direction) -> List[Step]:
        """
        Generate a gallop pattern - quick same-foot double tap.

        Gallops create a distinctive "da-DUM" rhythm commonly used in DDR
        for accenting syncopated beats. The second note typically lands
        on the beat while the first is a quick pickup.

        Args:
            start_time: Time for the first (pickup) note
            interval: Time between the two notes (typically 1/8 or 1/16 note)
            arrow: Arrow direction (usually same arrow or same-foot pair)

        Returns:
            List of 2 Step objects forming the gallop
        """
        return [
            Step(time=start_time, arrows=[arrow], step_type=StepType.TAP),
            Step(time=start_time + interval, arrows=[arrow], step_type=StepType.TAP),
        ]

    @staticmethod
    def gallop_alternating(start_time: float, interval: float,
                           arrow1: Direction, arrow2: Direction) -> List[Step]:
        """
        Generate an alternating gallop - quick same-foot pair.

        Uses two different arrows on the same foot for more variety.
        Example: LEFT-DOWN or UP-RIGHT

        Args:
            start_time: Time for the first note
            interval: Time between notes
            arrow1: First arrow direction
            arrow2: Second arrow direction

        Returns:
            List of 2 Step objects
        """
        return [
            Step(time=start_time, arrows=[arrow1], step_type=StepType.TAP),
            Step(time=start_time + interval, arrows=[arrow2], step_type=StepType.TAP),
        ]

    @staticmethod
    def drill(start_time: float, count: int, interval: float, arrow: Direction) -> List[Step]:
        """
        Generate a drill pattern - rapid repeated taps on the same arrow.

        Drills test stamina and speed, commonly used during intense
        percussive sections like rapid hi-hats or drum rolls.

        Args:
            start_time: Time for the first note
            count: Number of taps in the drill (typically 4-8)
            interval: Time between taps (typically 1/16 note)
            arrow: Arrow direction for all taps

        Returns:
            List of Step objects forming the drill
        """
        return [
            Step(time=start_time + i * interval, arrows=[arrow], step_type=StepType.TAP)
            for i in range(count)
        ]

    @staticmethod
    def alternating_drill(start_time: float, count: int, interval: float,
                          arrow1: Direction, arrow2: Direction) -> List[Step]:
        """
        Generate an alternating drill - rapid alternating between two arrows.

        More comfortable than single-arrow drills at high speed because
        it allows foot alternation.

        Args:
            start_time: Time for the first note
            count: Total number of taps
            interval: Time between taps
            arrow1: First arrow (odd positions)
            arrow2: Second arrow (even positions)

        Returns:
            List of Step objects alternating between arrows
        """
        steps = []
        for i in range(count):
            arrow = arrow1 if i % 2 == 0 else arrow2
            steps.append(Step(
                time=start_time + i * interval,
                arrows=[arrow],
                step_type=StepType.TAP
            ))
        return steps

    @staticmethod
    def rest_aware_stream(beat_times: List[float],
                          rest_periods: List[tuple],
                          arrows: List[Direction]) -> List[Step]:
        """
        Generate a stream that respects musical rest periods.

        Creates steps for the given beat times but skips any that fall
        within rest periods, preserving musical breathing room.

        Args:
            beat_times: List of potential step times
            rest_periods: List of (start_time, end_time) tuples for rests
            arrows: List of arrow directions (cycled if shorter than beat_times)

        Returns:
            List of Step objects, excluding steps during rests
        """
        steps = []

        for i, time in enumerate(beat_times):
            # Check if this time falls within a rest period
            in_rest = any(start <= time <= end for start, end in rest_periods)

            if not in_rest:
                arrow = arrows[i % len(arrows)] if arrows else Direction.LEFT
                steps.append(Step(
                    time=time,
                    arrows=[arrow],
                    step_type=StepType.TAP
                ))

        return steps

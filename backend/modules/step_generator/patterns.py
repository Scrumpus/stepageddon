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

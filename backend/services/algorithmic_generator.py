"""
Enhanced Algorithmic Step Generation
Pure librosa-based step generation without AI dependency
"""

import logging
import random
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class EnhancedAlgorithmicGenerator:
    """
    Advanced algorithmic step generation using librosa analysis
    Creates musically intelligent patterns without AI
    """
    
    def __init__(self):
        self.directions = ["left", "down", "up", "right"]
        # Opposite directions for doubles
        self.opposites = {
            "left": "right",
            "right": "left",
            "up": "down",
            "down": "up"
        }
        # Adjacent directions for patterns
        self.adjacent = {
            "left": ["down", "up"],
            "down": ["left", "right"],
            "up": ["left", "right"],
            "right": ["down", "up"]
        }
    
    def generate_steps(
        self,
        audio_analysis: Dict,
        difficulty: str
    ) -> List[Dict]:
        """
        Generate step chart using advanced algorithmic methods
        
        Args:
            audio_analysis: Complete audio analysis from librosa
            difficulty: 'beginner', 'intermediate', or 'expert'
            
        Returns:
            List of step objects
        """
        logger.info(f"Generating algorithmic steps for {difficulty} difficulty")
        
        params = self._get_difficulty_params(difficulty)
        
        # Get timing candidates from analysis
        timing_points = self._get_timing_points(audio_analysis, params)
        
        # Analyze energy and structure for intelligent placement
        energy_profile = audio_analysis["energy_profile"]
        structure = audio_analysis.get("structure", {})
        
        # Generate steps based on timing and energy
        steps = self._create_pattern_steps(
            timing_points,
            energy_profile,
            structure,
            params,
            audio_analysis
        )
        
        logger.info(f"Generated {len(steps)} algorithmic steps")
        return steps
    
    def _get_difficulty_params(self, difficulty: str) -> Dict:
        """Get parameters for difficulty level"""
        params = {
            "beginner": {
                "notes_per_second": (0.5, 1.5),
                "allow_doubles": False,
                "allow_jumps": False,
                "min_gap": 0.5,
                "double_probability": 0.0,
                "pattern_complexity": 1,
                "use_onsets": False,
                "energy_sensitivity": 0.3,
            },
            "intermediate": {
                "notes_per_second": (1.5, 2.5),
                "allow_doubles": True,
                "allow_jumps": False,
                "min_gap": 0.3,
                "double_probability": 0.15,
                "pattern_complexity": 2,
                "use_onsets": True,
                "energy_sensitivity": 0.5,
            },
            "expert": {
                "notes_per_second": (2.5, 4.0),
                "allow_doubles": True,
                "allow_jumps": True,
                "min_gap": 0.15,
                "double_probability": 0.25,
                "pattern_complexity": 3,
                "use_onsets": True,
                "energy_sensitivity": 0.7,
            }
        }
        return params.get(difficulty, params["intermediate"])
    
    def _get_timing_points(
        self,
        audio_analysis: Dict,
        params: Dict
    ) -> List[float]:
        """
        Get candidate timing points from beat and onset detection
        """
        beat_times = audio_analysis.get("beat_times", [])
        onset_times = audio_analysis.get("onset_times", [])
        
        # Start with beats
        timing_points = list(beat_times)
        
        # Add onsets for higher difficulties
        if params["use_onsets"]:
            # Add onsets that don't overlap with beats
            for onset in onset_times:
                if not any(abs(onset - beat) < 0.1 for beat in beat_times):
                    timing_points.append(onset)
        
        # Sort and deduplicate
        timing_points = sorted(set(timing_points))
        
        return timing_points
    
    def _create_pattern_steps(
        self,
        timing_points: List[float],
        energy_profile: Dict,
        structure: Dict,
        params: Dict,
        audio_analysis: Dict
    ) -> List[Dict]:
        """
        Create steps with intelligent patterns based on music
        """
        steps = []
        last_direction = None
        last_time = -params["min_gap"]
        pattern_state = {"streak": 0, "last_pattern": None}
        
        # Get energy values for intelligent placement
        energy_values = energy_profile.get("profile", [])
        energy_timestamps = energy_profile.get("timestamps", [])
        
        for time in timing_points:
            # Enforce minimum gap
            if time - last_time < params["min_gap"]:
                continue
            
            # Get energy at this time
            energy = self._get_energy_at_time(time, energy_timestamps, energy_values)
            
            # Decide if we should place a note here based on energy
            if not self._should_place_note(energy, params, time, audio_analysis):
                continue
            
            # Decide between single, double, or jump
            step_type = self._decide_step_type(
                energy,
                params,
                pattern_state,
                last_direction
            )
            
            if step_type == "single":
                direction = self._choose_direction(
                    last_direction,
                    params["pattern_complexity"],
                    pattern_state,
                    energy
                )
                steps.append({
                    "time": float(time),
                    "direction": direction,
                    "type": "single"
                })
                last_direction = direction
                
            elif step_type == "double":
                # Add two arrows at same time (opposite directions)
                dir1, dir2 = self._choose_double_directions(last_direction)
                steps.append({
                    "time": float(time),
                    "direction": dir1,
                    "type": "double"
                })
                steps.append({
                    "time": float(time),
                    "direction": dir2,
                    "type": "double"
                })
                last_direction = dir1
                
            elif step_type == "jump":
                # Jump: two simultaneous arrows
                dir1, dir2 = self._choose_jump_directions()
                steps.append({
                    "time": float(time),
                    "direction": dir1,
                    "type": "double"
                })
                steps.append({
                    "time": float(time),
                    "direction": dir2,
                    "type": "double"
                })
                last_direction = dir1
            
            last_time = time
        
        return steps
    
    def _get_energy_at_time(
        self,
        time: float,
        timestamps: List[float],
        energy_values: List[float]
    ) -> float:
        """Get energy value at specific time"""
        if not timestamps or not energy_values:
            return 0.5
        
        # Find closest timestamp
        idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - time))
        return energy_values[idx]
    
    def _should_place_note(
        self,
        energy: float,
        params: Dict,
        time: float,
        audio_analysis: Dict
    ) -> bool:
        """
        Decide if note should be placed based on energy and context
        """
        # Always place at very high energy
        if energy > 0.8:
            return True
        
        # For beginner, be more conservative
        if params["pattern_complexity"] == 1:
            return energy > 0.4
        
        # For intermediate/expert, use energy sensitivity
        threshold = 0.5 - (params["energy_sensitivity"] * 0.2)
        
        # Add some randomness for variety
        return energy > threshold or random.random() < 0.1
    
    def _decide_step_type(
        self,
        energy: float,
        params: Dict,
        pattern_state: Dict,
        last_direction: str
    ) -> str:
        """
        Decide between single, double, or jump based on energy and difficulty
        """
        # High energy = higher chance of doubles/jumps
        if energy > 0.85 and params["allow_jumps"]:
            if random.random() < 0.3:
                return "jump"
        
        if params["allow_doubles"]:
            # More doubles at high energy
            double_chance = params["double_probability"]
            if energy > 0.7:
                double_chance *= 1.5
            
            if random.random() < double_chance:
                return "double"
        
        return "single"
    
    def _choose_direction(
        self,
        last_direction: str,
        complexity: int,
        pattern_state: Dict,
        energy: float
    ) -> str:
        """
        Choose direction based on patterns and difficulty
        """
        if last_direction is None:
            return random.choice(self.directions)
        
        # Complexity 1 (Beginner): Avoid repeats, simple alternation
        if complexity == 1:
            available = [d for d in self.directions if d != last_direction]
            return random.choice(available)
        
        # Complexity 2 (Intermediate): Create simple patterns
        elif complexity == 2:
            # Occasionally continue patterns
            if pattern_state["streak"] < 3 and random.random() < 0.3:
                pattern_state["streak"] += 1
                # Continue in adjacent direction
                return random.choice(self.adjacent[last_direction] + [last_direction])
            else:
                pattern_state["streak"] = 0
                available = [d for d in self.directions if d != last_direction]
                return random.choice(available)
        
        # Complexity 3 (Expert): Complex patterns, all directions
        else:
            # High energy = more variety
            if energy > 0.7:
                return random.choice(self.directions)
            
            # Create runs and patterns
            if random.random() < 0.4:
                # Adjacent move
                return random.choice(self.adjacent[last_direction])
            else:
                # Any direction
                return random.choice(self.directions)
    
    def _choose_double_directions(self, last_direction: str) -> Tuple[str, str]:
        """Choose two directions for a double arrow"""
        # Prefer opposites (left+right or up+down)
        if last_direction in self.opposites:
            opposite = self.opposites[last_direction]
            if random.random() < 0.7:
                return (last_direction, opposite)
        
        # Otherwise pick two different directions
        dir1 = random.choice(self.directions)
        available = [d for d in self.directions if d != dir1]
        dir2 = random.choice(available)
        
        return (dir1, dir2)
    
    def _choose_jump_directions(self) -> Tuple[str, str]:
        """Choose two directions for a jump (expert mode)"""
        # Jumps are typically opposites
        if random.random() < 0.8:
            # Left+Right or Up+Down
            if random.random() < 0.5:
                return ("left", "right")
            else:
                return ("up", "down")
        else:
            # Diagonal-ish jumps
            return random.choice([
                ("left", "up"),
                ("left", "down"),
                ("right", "up"),
                ("right", "down")
            ])

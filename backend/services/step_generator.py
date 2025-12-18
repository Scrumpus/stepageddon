import json
import logging
from typing import Dict, List

from core.config import settings
from services.algorithmic_generator import EnhancedAlgorithmicGenerator

logger = logging.getLogger(__name__)

class StepGenerator:
    def __init__(self, use_ai: bool = False):
        """
        Initialize step generator
        
        Args:
            use_ai: Will implement later
        """
        self.use_ai = use_ai
        self.client = None
        
        # Always initialize algorithmic generator
        self.algorithmic_generator = EnhancedAlgorithmicGenerator()
    
    async def generate_steps(
        self,
        audio_analysis: Dict,
        difficulty: str,
        song_info: Dict = None
    ) -> List[Dict]:
        """
        Generate step chart
        
        Args:
            audio_analysis: Audio analysis results
            difficulty: 'beginner', 'intermediate', or 'expert'
            song_info: Optional song metadata
            
        Returns:
            List of step objects with time and direction
        """
        try:
            logger.info(f"Generating {difficulty} steps...")
            
            # Use AI if enabled and available
            if self.use_ai and self.client:
                try:
                    steps = await self._generate_ai_steps(
                        audio_analysis, difficulty, song_info
                    )
                    logger.info(f"✓ AI generated {len(steps)} steps")
                    return steps
                except Exception as e:
                    logger.warning(f"AI generation failed: {e}, falling back to algorithmic")
            
            # Use enhanced algorithmic generation
            steps = self.algorithmic_generator.generate_steps(audio_analysis, difficulty)
            logger.info(f"✓ Algorithmically generated {len(steps)} steps")
            return steps
            
        except Exception as e:
            logger.error(f"Step generation failed: {e}", exc_info=True)
            raise
    
    def _get_difficulty_params(self, difficulty: str) -> Dict:
        """Get parameters for difficulty level"""
        params = {
            "beginner": {
                "notes_per_second": (0.5, 1.5),
                "allow_doubles": False,
                "allow_jumps": False,
                "min_gap": 0.5,
                "description": "Beginner - slow, single arrows only"
            },
            "intermediate": {
                "notes_per_second": (1.5, 2.5),
                "allow_doubles": True,
                "allow_jumps": False,
                "min_gap": 0.3,
                "description": "Intermediate - moderate speed, some doubles"
            },
            "expert": {
                "notes_per_second": (2.5, 4.0),
                "allow_doubles": True,
                "allow_jumps": True,
                "min_gap": 0.15,
                "description": "Expert - fast, complex patterns"
            }
        }
        return params.get(difficulty, params["intermediate"])
    
    def _validate_and_refine_steps(
        self,
        steps: List[Dict],
        duration: float,
        params: Dict
    ) -> List[Dict]:
        """Validate and refine step chart"""
        # Sort by time
        steps.sort(key=lambda x: x["time"])
        
        # Remove duplicates
        seen_times = set()
        unique_steps = []
        for step in steps:
            time_key = round(step["time"], 2)
            if time_key not in seen_times:
                seen_times.add(time_key)
                unique_steps.append(step)
        steps = unique_steps
        
        # Remove out-of-bounds steps
        steps = [s for s in steps if 0 <= s["time"] <= duration]
        
        # Ensure minimum gap
        min_gap = params["min_gap"]
        filtered_steps = []
        last_time = -min_gap
        
        for step in steps:
            if step["time"] - last_time >= min_gap:
                filtered_steps.append(step)
                last_time = step["time"]
        
        # Validate directions
        valid_directions = ["left", "down", "up", "right"]
        filtered_steps = [
            s for s in filtered_steps 
            if s["direction"] in valid_directions
        ]
        
        return filtered_steps
    
    def _generate_algorithmic_steps(
        self,
        audio_analysis: Dict,
        difficulty: str
    ) -> List[Dict]:
        """Fallback algorithmic step generation"""
        logger.info("Using algorithmic step generation")
        
        params = self._get_difficulty_params(difficulty)
        steps = []
        
        beat_times = audio_analysis["beat_times"]
        onset_times = audio_analysis["onset_times"]
        
        # Combine beats and onsets
        all_times = sorted(set(beat_times + onset_times))
        
        # Filter based on difficulty
        min_gap = params["min_gap"]
        filtered_times = []
        last_time = -min_gap
        
        for t in all_times:
            if t - last_time >= min_gap:
                filtered_times.append(t)
                last_time = t
        
        # Generate steps
        directions = ["left", "down", "up", "right"]
        last_direction = None
        
        for i, time in enumerate(filtered_times):
            # Choose direction (avoid repeating)
            available = [d for d in directions if d != last_direction]
            direction = available[i % len(available)]
            
            steps.append({
                "time": float(time),
                "direction": direction,
                "type": "single"
            })
            
            last_direction = direction
            
            # Add doubles for expert mode occasionally
            if params["allow_doubles"] and i % 8 == 7:
                opposite = {
                    "left": "right",
                    "right": "left",
                    "up": "down",
                    "down": "up"
                }
                steps.append({
                    "time": float(time),
                    "direction": opposite[direction],
                    "type": "double"
                })
        
        return steps

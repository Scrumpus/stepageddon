"""
Step Generation Service
AI-powered and algorithmic step chart generation
"""

import json
import logging
from typing import Dict, List
from anthropic import Anthropic

from core.config import settings
from services.algorithmic_generator import EnhancedAlgorithmicGenerator

logger = logging.getLogger(__name__)


class StepGenerator:
    """Generate step charts using AI and algorithmic methods"""
    
    def __init__(self, use_ai: bool = False):
        """
        Initialize step generator
        
        Args:
            use_ai: If True, use AI (requires API key). If False, use pure algorithmic.
        """
        self.use_ai = use_ai and bool(settings.ANTHROPIC_API_KEY)
        self.client = None
        
        if self.use_ai:
            self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info("AI generation enabled")
        else:
            logger.info("Using pure algorithmic generation (no AI required)")
        
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
    
    async def _generate_ai_steps(
        self,
        audio_analysis: Dict,
        difficulty: str,
        song_info: Dict = None
    ) -> List[Dict]:
        """Generate steps using Claude AI"""
        
        # Prepare analysis summary for AI
        analysis_summary = self._prepare_analysis_summary(audio_analysis, song_info)
        
        # Difficulty parameters
        difficulty_params = self._get_difficulty_params(difficulty)
        
        # Enhanced prompt
        prompt = self._create_ai_prompt(analysis_summary, difficulty_params)
        
        # Call Claude API
        message = self.client.messages.create(
            model=settings.AI_MODEL,
            max_tokens=4096,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Parse response
        response_text = message.content[0].text
        steps = self._parse_ai_response(response_text)
        
        # Validate and refine
        steps = self._validate_and_refine_steps(
            steps,
            audio_analysis["duration"],
            difficulty_params
        )
        
        return steps
    
    def _prepare_analysis_summary(self, analysis: Dict, song_info: Dict = None) -> str:
        """Prepare concise analysis summary for AI"""
        summary_parts = [
            f"Duration: {analysis['duration']:.1f}s",
            f"Tempo: {analysis['tempo']:.1f} BPM",
            f"Beats detected: {len(analysis['beat_times'])}",
            f"Genre/Style: {analysis['spectral_features']['genre_hint']}",
            f"Energy level: {'High' if analysis['energy_profile']['mean'] > 0.05 else 'Medium' if analysis['energy_profile']['mean'] > 0.03 else 'Low'}",
        ]
        
        if song_info:
            if song_info.get("title"):
                summary_parts.insert(0, f"Song: {song_info['title']}")
            if song_info.get("artist"):
                summary_parts.insert(1, f"Artist: {song_info['artist']}")
        
        # Add beat times (first 20 for reference)
        beat_times_str = ", ".join([f"{t:.2f}" for t in analysis['beat_times'][:20]])
        if len(analysis['beat_times']) > 20:
            beat_times_str += "..."
        summary_parts.append(f"Beat times (s): {beat_times_str}")
        
        return "\n".join(summary_parts)
    
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
    
    def _create_ai_prompt(self, analysis_summary: str, difficulty_params: Dict) -> str:
        """Create enhanced prompt for Claude"""
        
        return f"""You are an expert DDR (Dance Dance Revolution) step chart creator. Generate a step chart for this song.

SONG ANALYSIS:
{analysis_summary}

DIFFICULTY: {difficulty_params['description']}
- Target density: {difficulty_params['notes_per_second'][0]}-{difficulty_params['notes_per_second'][1]} notes/second
- Double arrows: {'Allowed' if difficulty_params['allow_doubles'] else 'Not allowed'}
- Jump arrows (simultaneous): {'Allowed' if difficulty_params['allow_jumps'] else 'Not allowed'}
- Minimum gap between notes: {difficulty_params['min_gap']}s

ARROW DIRECTIONS:
- left (←)
- down (↓)
- up (↑)
- right (→)

GUIDELINES:
1. Place arrows on or near beat times for musical accuracy
2. Create patterns that flow naturally (avoid awkward foot movements)
3. Use directional variety (don't repeat same arrow too much)
4. Match energy: more arrows in high-energy sections, fewer in calm parts
5. For doubles, use opposite directions (left+right or up+down)
6. Build intensity gradually (easier at start, harder towards chorus)
7. Leave gaps for breathing/recovery in long songs

OUTPUT FORMAT (JSON):
Return ONLY a JSON array of step objects. Each object must have:
- "time": float (timestamp in seconds)
- "direction": string (one of: "left", "down", "up", "right")
- "type": string ("single" or "double")

Example:
[
  {{"time": 1.23, "direction": "left", "type": "single"}},
  {{"time": 1.75, "direction": "down", "type": "single"}},
  {{"time": 2.15, "direction": "left", "type": "double"}},
  {{"time": 2.15, "direction": "right", "type": "double"}},
  {{"time": 2.89, "direction": "up", "type": "single"}}
]

Generate the complete step chart now:"""
    
    def _parse_ai_response(self, response: str) -> List[Dict]:
        """Parse AI response into step objects"""
        try:
            # Try to extract JSON from response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            # Parse JSON
            steps = json.loads(response)
            
            # Validate structure
            if not isinstance(steps, list):
                raise ValueError("Response is not a list")
            
            for step in steps:
                if not all(k in step for k in ["time", "direction", "type"]):
                    raise ValueError(f"Invalid step structure: {step}")
            
            return steps
            
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.debug(f"Response was: {response[:500]}")
            raise ValueError("Could not parse AI response into valid steps")
    
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

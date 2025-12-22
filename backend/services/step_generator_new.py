"""
Deterministic Step Generation Engine
Main module for generating DDR-style charts with hold notes.
"""

import librosa
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
import json


# ============================================================================
# Data Classes
# ============================================================================

class StepType(Enum):
    """Types of steps in the game."""
    TAP = "tap"
    HOLD = "hold"


class Direction(Enum):
    """Arrow directions."""
    LEFT = "left"
    DOWN = "down"
    UP = "up"
    RIGHT = "right"


@dataclass
class Beat:
    """Represents a single beat."""
    time: float
    strength: float
    beat_type: str  # 'downbeat', 'upbeat', 'offbeat'
    measure_position: int  # 0-3 for 4/4 time
    is_strong: bool


@dataclass
class EnergySection:
    """Represents an energy level over a time range."""
    start_time: float
    end_time: float
    energy_level: float  # 0.0 to 1.0
    intensity: str  # 'low', 'medium', 'high', 'climax'


@dataclass
class SustainedNote:
    """Represents a sustained melodic note (candidate for hold)."""
    start_time: float
    end_time: float
    pitch: float  # MIDI note number
    confidence: float


@dataclass
class SongStructure:
    """Complete song structure."""
    intro: Tuple[float, float]
    verses: List[Tuple[float, float]]
    choruses: List[Tuple[float, float]]
    bridge: Tuple[float, float]
    outro: Tuple[float, float]
    total_duration: float


@dataclass
class Step:
    """A single step (tap or hold start)."""
    time: float
    arrows: List[Direction]
    step_type: StepType = StepType.TAP
    hold_duration: Optional[float] = None
    
    def __post_init__(self):
        """Validate step."""
        if self.step_type == StepType.HOLD and self.hold_duration is None:
            raise ValueError("Hold steps must have hold_duration")
        if self.step_type == StepType.TAP and self.hold_duration is not None:
            raise ValueError("Tap steps cannot have hold_duration")


@dataclass
class Chart:
    """Complete chart with all steps."""
    steps: List[Step]
    difficulty: str
    tempo: float
    duration: float
    
    def get_taps(self) -> List[Step]:
        """Get only tap notes."""
        return [s for s in self.steps if s.step_type == StepType.TAP]
    
    def get_holds(self) -> List[Step]:
        """Get only hold notes."""
        return [s for s in self.steps if s.step_type == StepType.HOLD]


@dataclass
class DifficultyConfig:
    """Configuration for each difficulty level."""
    name: str
    min_density: float
    max_density: float
    allow_singles: bool
    allow_doubles: bool
    allow_holds: bool
    hold_percentage: float
    min_hold_duration: float
    max_hold_duration: float
    use_downbeats: bool
    use_upbeats: bool
    use_offbeats: bool
    use_8th_notes: bool
    use_16th_notes: bool
    max_consecutive_jumps: int
    max_stream_length: int
    allow_crossovers: bool
    allow_brackets: bool
    energy_scale_factor: float


# ============================================================================
# Difficulty Presets
# ============================================================================

DIFFICULTY_PRESETS = {
    "beginner": DifficultyConfig(
        name="beginner",
        min_density=0.6,
        max_density=1.2,
        allow_singles=True,
        allow_doubles=False,
        allow_holds=True,
        hold_percentage=0.15,
        min_hold_duration=0.8,
        max_hold_duration=2.0,
        use_downbeats=True,
        use_upbeats=False,
        use_offbeats=False,
        use_8th_notes=False,
        use_16th_notes=False,
        max_consecutive_jumps=0,
        max_stream_length=0,
        allow_crossovers=False,
        allow_brackets=False,
        energy_scale_factor=0.3
    ),
    
    "intermediate": DifficultyConfig(
        name="intermediate",
        min_density=1.3,
        max_density=2.3,
        allow_singles=True,
        allow_doubles=True,
        allow_holds=True,
        hold_percentage=0.20,
        min_hold_duration=0.6,
        max_hold_duration=3.0,
        use_downbeats=True,
        use_upbeats=True,
        use_offbeats=True,
        use_8th_notes=True,
        use_16th_notes=False,
        max_consecutive_jumps=2,
        max_stream_length=6,
        allow_crossovers=True,
        allow_brackets=False,
        energy_scale_factor=0.6
    ),
    
    "expert": DifficultyConfig(
        name="expert",
        min_density=2.2,
        max_density=4.0,
        allow_singles=True,
        allow_doubles=True,
        allow_holds=True,
        hold_percentage=0.25,
        min_hold_duration=0.5,
        max_hold_duration=4.0,
        use_downbeats=True,
        use_upbeats=True,
        use_offbeats=True,
        use_8th_notes=True,
        use_16th_notes=True,
        max_consecutive_jumps=4,
        max_stream_length=16,
        allow_crossovers=True,
        allow_brackets=True,
        energy_scale_factor=1.0
    )
}


# ============================================================================
# Audio Analysis Functions
# ============================================================================

def analyze_beats(y: np.ndarray, sr: int) -> Tuple[List[Beat], float]:
    """
    Detect and classify all beats in the audio.
    """
    # Get tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Calculate onset strength for each beat
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    beats = []
    for i, (frame, time) in enumerate(zip(beat_frames, beat_times)):
        # Get strength at this beat
        strength = onset_env[frame] if frame < len(onset_env) else 0
        
        # Classify beat position in measure (assume 4/4 time)
        measure_position = i % 4
        
        # Determine beat type
        if measure_position == 0:
            beat_type = 'downbeat'
            is_strong = True
        elif measure_position == 2:
            beat_type = 'upbeat'
            is_strong = True
        else:
            beat_type = 'offbeat'
            is_strong = False
        
        beats.append(Beat(
            time=float(time),
            strength=float(strength),
            beat_type=beat_type,
            measure_position=measure_position,
            is_strong=is_strong
        ))
    
    return beats, float(tempo)


def detect_subdivisions(y: np.ndarray, sr: int, beat_times: List[float]) -> List[float]:
    """
    Detect 8th and 16th note subdivisions between beats.
    """
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    subdivisions = []
    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]
        
        between = [float(t) for t in onset_times if beat_start < t < beat_end]
        subdivisions.extend(between)
    
    return subdivisions


def analyze_energy(y: np.ndarray, sr: int, window_size: float = 2.0) -> List[EnergySection]:
    """
    Analyze energy levels throughout the song.
    """
    hop_length = int(window_size * sr / 2)
    
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    
    smoothed = gaussian_filter1d(rms, sigma=2)
    energy_normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)
    
    sections = []
    for i in range(len(times) - 1):
        energy = float(energy_normalized[i])
        
        if energy < 0.3:
            intensity = 'low'
        elif energy < 0.6:
            intensity = 'medium'
        elif energy < 0.85:
            intensity = 'high'
        else:
            intensity = 'climax'
        
        sections.append(EnergySection(
            start_time=float(times[i]),
            end_time=float(times[i + 1]),
            energy_level=energy,
            intensity=intensity
        ))
    
    return sections


def detect_energy_peaks(sections: List[EnergySection]) -> List[float]:
    """Find peak energy moments."""
    energies = [s.energy_level for s in sections]
    times = [s.start_time for s in sections]
    
    peaks, _ = find_peaks(energies, height=0.7, distance=4)
    peak_times = [times[p] for p in peaks]
    
    return peak_times


def detect_sustained_notes(y: np.ndarray, sr: int, min_duration: float = 0.5) -> List[SustainedNote]:
    """
    Detect sustained notes/vocals for hold note placement.
    """
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    times = librosa.times_like(pitches, sr=sr)
    
    pitch_sequence = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        confidence = magnitudes[index, t]
        pitch_sequence.append((float(times[t]), float(pitch), float(confidence)))
    
    sustained = []
    current_note = None
    
    for time, pitch, confidence in pitch_sequence:
        if confidence < 0.1 or pitch < 50:
            if current_note:
                if current_note['end'] - current_note['start'] >= min_duration:
                    sustained.append(SustainedNote(
                        start_time=current_note['start'],
                        end_time=current_note['end'],
                        pitch=current_note['pitch'],
                        confidence=current_note['confidence']
                    ))
                current_note = None
            continue
        
        if current_note is None:
            current_note = {
                'start': time,
                'end': time,
                'pitch': pitch,
                'confidence': confidence
            }
        else:
            if abs(pitch - current_note['pitch']) < 2:
                current_note['end'] = time
            else:
                if current_note['end'] - current_note['start'] >= min_duration:
                    sustained.append(SustainedNote(
                        start_time=current_note['start'],
                        end_time=current_note['end'],
                        pitch=current_note['pitch'],
                        confidence=current_note['confidence']
                    ))
                current_note = {
                    'start': time,
                    'end': time,
                    'pitch': pitch,
                    'confidence': confidence
                }
    
    return sustained


def detect_structure(y: np.ndarray, sr: int) -> SongStructure:
    """
    Detect song structure using self-similarity analysis.
    """
    duration = librosa.get_duration(y=y, sr=sr)
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    R = librosa.segment.recurrence_matrix(chroma, mode='affinity', metric='cosine')
    boundaries_frames = librosa.segment.agglomerative(chroma, k=8)
    boundaries = librosa.frames_to_time(boundaries_frames, sr=sr)
    
    segments_energy = []
    for i in range(len(boundaries) - 1):
        start_frame = librosa.time_to_frames(boundaries[i], sr=sr)
        end_frame = librosa.time_to_frames(boundaries[i + 1], sr=sr)
        segment_audio = y[start_frame:end_frame]
        energy = np.sqrt(np.mean(segment_audio ** 2))
        segments_energy.append(float(energy))
    
    max_energy = max(segments_energy) if segments_energy else 1.0
    segments_energy = [e / max_energy for e in segments_energy]
    
    intro_end = min(float(boundaries[1]), duration * 0.15)
    intro = (0.0, intro_end)
    
    outro_start = max(float(boundaries[-2]), duration * 0.85)
    outro = (outro_start, duration)
    
    high_energy_threshold = 0.7
    high_energy_indices = [i for i, e in enumerate(segments_energy) if e > high_energy_threshold]
    
    choruses = []
    verses = []
    
    for i in range(1, len(boundaries) - 2):
        segment_start = float(boundaries[i])
        segment_end = float(boundaries[i + 1])
        
        if segment_start < intro_end or segment_start > outro_start:
            continue
        
        if i in high_energy_indices:
            choruses.append((segment_start, segment_end))
        else:
            verses.append((segment_start, segment_end))
    
    bridge_candidates = [(float(boundaries[i]), float(boundaries[i + 1])) 
                        for i in range(len(boundaries) - 1)
                        if 0.6 * duration < boundaries[i] < 0.8 * duration]
    bridge = bridge_candidates[0] if bridge_candidates else (0.7 * duration, 0.75 * duration)
    
    return SongStructure(
        intro=intro,
        verses=verses,
        choruses=choruses,
        bridge=bridge,
        outro=outro,
        total_duration=duration
    )


# ============================================================================
# Pattern Templates
# ============================================================================

class PatternTemplate:
    """Defines reusable step patterns."""
    
    @staticmethod
    def single_stream(start_time: float, count: int, interval: float, 
                     start_arrow: Direction) -> List[Step]:
        """Generate a stream of single arrows."""
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
        """Generate a crossover pattern."""
        return [
            Step(start_time, [Direction.LEFT], StepType.TAP),
            Step(start_time + interval, [Direction.RIGHT], StepType.TAP),
            Step(start_time + interval * 2, [Direction.LEFT], StepType.TAP),
            Step(start_time + interval * 3, [Direction.RIGHT], StepType.TAP)
        ]
    
    @staticmethod
    def jump_pattern(start_time: float, jump_type: str) -> Step:
        """Generate a jump (2 arrows)."""
        jumps = {
            'corners': [Direction.LEFT, Direction.DOWN],
            'brackets': [Direction.LEFT, Direction.RIGHT],
            'middle': [Direction.DOWN, Direction.UP],
            'diagonal1': [Direction.LEFT, Direction.UP],
            'diagonal2': [Direction.DOWN, Direction.RIGHT]
        }
        return Step(start_time, jumps.get(jump_type, jumps['corners']), StepType.TAP)
    
    @staticmethod
    def hold_note(start_time: float, arrow: Direction, duration: float) -> Step:
        """Generate a hold note."""
        return Step(
            time=start_time,
            arrows=[arrow],
            step_type=StepType.HOLD,
            hold_duration=duration
        )


# ============================================================================
# Step Generator
# ============================================================================

class StepGenerator:
    """Main step generation engine."""
    
    def __init__(self, config: DifficultyConfig):
        self.config = config
    
    def generate_chart(self, 
                      beats: List[Beat],
                      subdivisions: List[float],
                      energy_sections: List[EnergySection],
                      sustained_notes: List[SustainedNote],
                      structure: SongStructure,
                      tempo: float) -> Chart:
        """Main chart generation method."""
        steps = []
        
        duration = structure.total_duration
        target_density = (self.config.min_density + self.config.max_density) / 2
        
        available_beats = self._filter_beats(beats)
        
        if self.config.use_8th_notes or self.config.use_16th_notes:
            available_beats.extend(subdivisions)
            available_beats.sort()
        
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
        for section in energy_sections:
            section_beats = [b for b in available_beats 
                           if section.start_time <= b <= section.end_time]
            
            density_multiplier = 1.0 + (section.energy_level - 0.5) * self.config.energy_scale_factor
            section_target = int(len(section_beats) * density_multiplier)
            section_target = min(section_target, len(section_beats))
            
            beats_to_use = self._select_beats_by_energy(section_beats, beats, section_target)
            section_steps = self._generate_patterns(beats_to_use, section.intensity, structure)
            steps.extend(section_steps)
        
        # Phase 4: Structure-aware adjustments
        steps = self._adjust_for_structure(steps, structure)
        
        # Phase 5: Validate
        steps = self._validate_chart(steps, tempo)
        
        steps.sort(key=lambda s: s.time)
        
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
                    steps.append(Step(time, [Direction.LEFT], StepType.TAP))
                    i += 1
            else:
                arrow = self._choose_single_arrow(steps)
                steps.append(Step(time, [arrow], StepType.TAP))
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


# ============================================================================
# Chart Generation Pipeline
# ============================================================================

class ChartGenerationPipeline:
    """Complete pipeline from audio to chart."""
    
    @staticmethod
    def generate_from_audio(audio_path: str, difficulty: str = 'intermediate') -> Chart:
        """
        Generate a complete chart from an audio file.
        """
        print(f"Loading audio from {audio_path}...")
        y, sr = librosa.load(audio_path)
        
        print("Analyzing audio...")
        beats, tempo = analyze_beats(y, sr)
        subdivisions = detect_subdivisions(y, sr, [b.time for b in beats])
        energy_sections = analyze_energy(y, sr)
        sustained_notes = detect_sustained_notes(y, sr)
        structure = detect_structure(y, sr)
        
        print(f"Detected {len(beats)} beats at {tempo:.1f} BPM")
        print(f"Found {len(sustained_notes)} sustained notes for holds")
        
        print(f"Generating {difficulty} chart...")
        config = DIFFICULTY_PRESETS[difficulty]
        generator = StepGenerator(config)
        
        chart = generator.generate_chart(
            beats=beats,
            subdivisions=subdivisions,
            energy_sections=energy_sections,
            sustained_notes=sustained_notes,
            structure=structure,
            tempo=tempo
        )
        
        print(f"Generated {len(chart.steps)} steps")
        print(f"  Taps: {len(chart.get_taps())}")
        print(f"  Holds: {len(chart.get_holds())}")
        
        return chart


# ============================================================================
# Chart Exporter
# ============================================================================

class ChartExporter:
    """Export charts to various formats."""
    
    @staticmethod
    def to_json(chart: Chart) -> str:
        """Export chart to JSON format."""
        steps_data = []
        
        for step in chart.steps:
            step_dict = {
                'time': round(step.time, 3),
                'arrows': [a.value for a in step.arrows],
                'type': step.step_type.value
            }
            
            if step.step_type == StepType.HOLD:
                step_dict['hold_duration'] = round(step.hold_duration, 3)
            
            steps_data.append(step_dict)
        
        chart_data = {
            'difficulty': chart.difficulty,
            'tempo': round(chart.tempo, 1),
            'duration': round(chart.duration, 2),
            'steps': steps_data,
            'stats': {
                'total_steps': len(chart.steps),
                'total_arrows': sum(len(s.arrows) for s in chart.steps),
                'tap_notes': len(chart.get_taps()),
                'hold_notes': len(chart.get_holds()),
                'singles': len([s for s in chart.steps if len(s.arrows) == 1]),
                'doubles': len([s for s in chart.steps if len(s.arrows) == 2]),
            }
        }
        
        return json.dumps(chart_data, indent=2)

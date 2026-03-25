"""
Audio Analysis Functions

Pure functions for analyzing audio using librosa.
All functions are deterministic - same input produces same output.
"""

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple

from .schemas import Beat, EnergySection


def analyze_onsets(y: np.ndarray, sr: int, strength_threshold: float = 0.3) -> Tuple[List[float], np.ndarray]:
    """
    Detect all onsets (note attacks) in the audio.

    Args:
        y: Audio time series
        sr: Sample rate
        strength_threshold: Minimum onset strength (0.0-1.0) to include

    Returns:
        Tuple of (list of onset times in seconds, onset strength envelope)
    """
    # Get onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Normalize envelope to 0-1
    onset_env_norm = onset_env / (onset_env.max() + 1e-8)

    # Detect onset frames with backtracking for precision
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, units='frames', backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Filter by strength threshold
    filtered_times = []
    for frame, time in zip(onset_frames, onset_times):
        if frame < len(onset_env_norm) and onset_env_norm[frame] >= strength_threshold:
            filtered_times.append(float(time))

    return filtered_times, onset_env


def analyze_beats(y: np.ndarray, sr: int) -> Tuple[List[Beat], float]:
    """
    Detect and classify all beats in the audio.

    Args:
        y: Audio time series
        sr: Sample rate

    Returns:
        Tuple of (list of Beat objects, tempo in BPM)
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


def analyze_energy(y: np.ndarray, sr: int, window_size: float = 2.0) -> List[EnergySection]:
    """
    Analyze energy levels throughout the song.

    Args:
        y: Audio time series
        sr: Sample rate
        window_size: Window size in seconds for energy analysis

    Returns:
        List of EnergySection objects representing energy over time
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


def quantize_to_grid(times: List[float], tempo: float, grid_division: int = 8) -> List[float]:
    """
    Quantize times to musical grid based on tempo.

    Args:
        times: List of times in seconds
        tempo: Tempo in BPM
        grid_division: Grid resolution (4=quarter, 8=eighth, 16=sixteenth)

    Returns:
        List of quantized times snapped to nearest grid position
    """
    if not times or tempo <= 0:
        return times

    # Calculate grid spacing
    beat_duration = 60.0 / tempo  # Duration of one beat
    grid_spacing = beat_duration / (grid_division / 4)  # Grid unit duration

    quantized = []
    for t in times:
        # Find nearest grid position
        grid_position = round(t / grid_spacing)
        quantized_time = grid_position * grid_spacing
        quantized.append(quantized_time)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for t in quantized:
        t_rounded = round(t, 4)  # Avoid floating point issues
        if t_rounded not in seen:
            seen.add(t_rounded)
            unique.append(t_rounded)

    return sorted(unique)

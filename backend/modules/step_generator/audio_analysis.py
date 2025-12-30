"""
Audio Analysis Functions

Pure functions for analyzing audio using librosa.
All functions are deterministic - same input produces same output.
"""

import librosa
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple

from .schemas import Beat, EnergySection, SustainedNote, SongStructure


def analyze_onsets(y: np.ndarray, sr: int, strength_threshold: float = 0.3) -> Tuple[List[float], np.ndarray]:
    """
    Detect all onsets (note attacks) in the audio.

    This captures every distinct sound event, not just the rhythmic pulse.
    Use this for denser step charts that follow the actual notes being played.

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


def detect_subdivisions(y: np.ndarray, sr: int, beat_times: List[float]) -> List[float]:
    """
    Detect 8th and 16th note subdivisions between beats.

    Args:
        y: Audio time series
        sr: Sample rate
        beat_times: List of beat times in seconds

    Returns:
        List of subdivision times (onsets between beats)
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


def detect_energy_peaks(sections: List[EnergySection]) -> List[float]:
    """
    Find peak energy moments.

    Args:
        sections: List of EnergySection objects

    Returns:
        List of times (in seconds) where energy peaks occur
    """
    energies = [s.energy_level for s in sections]
    times = [s.start_time for s in sections]

    peaks, _ = find_peaks(energies, height=0.7, distance=4)
    peak_times = [times[p] for p in peaks]

    return peak_times


def detect_sustained_notes(y: np.ndarray, sr: int, min_duration: float = 0.5) -> List[SustainedNote]:
    """
    Detect sustained notes/vocals for hold note placement.

    Args:
        y: Audio time series
        sr: Sample rate
        min_duration: Minimum duration for a note to be considered sustained

    Returns:
        List of SustainedNote objects representing melodic holds
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

    Args:
        y: Audio time series
        sr: Sample rate

    Returns:
        SongStructure object with detected sections
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

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

from .schemas import (
    Beat, EnergySection, SustainedNote, SongStructure,
    PitchFrame, TempoSection, PhraseBoundary, OnsetClassification
)


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
    Detect sustained notes using energy analysis (more reliable than pitch tracking).

    Looks for periods of sustained high energy between onsets, which indicate
    held notes, sustained synths, or vocal holds.

    Args:
        y: Audio time series
        sr: Sample rate
        min_duration: Minimum duration for a note to be considered sustained

    Returns:
        List of SustainedNote objects representing melodic holds
    """
    # Get RMS energy with small hop for precision
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    # Smooth the energy curve
    rms_smooth = gaussian_filter1d(rms, sigma=3)

    # Normalize
    rms_norm = rms_smooth / (rms_smooth.max() + 1e-8)

    # Get onsets to find gaps
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units='frames')

    sustained = []

    # Look for sustained energy between onsets
    for i in range(len(onset_frames) - 1):
        start_frame = onset_frames[i]
        end_frame = onset_frames[i + 1]

        # Check duration
        start_time = float(times[start_frame]) if start_frame < len(times) else 0
        end_time = float(times[min(end_frame, len(times) - 1)])
        duration = end_time - start_time

        if duration < min_duration:
            continue

        # Check if energy stays high throughout
        segment_energy = rms_norm[start_frame:end_frame]
        if len(segment_energy) == 0:
            continue

        avg_energy = float(np.mean(segment_energy))
        min_energy = float(np.min(segment_energy))

        # Sustained note: high average energy that doesn't drop much
        if avg_energy > 0.3 and min_energy > avg_energy * 0.5:
            # Use spectral centroid as proxy for "pitch" (higher = brighter)
            segment_audio = y[start_frame * hop_length:end_frame * hop_length]
            if len(segment_audio) > 0:
                centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)
                pitch_proxy = float(np.mean(centroid)) / 100  # Scale to reasonable range

                sustained.append(SustainedNote(
                    start_time=start_time,
                    end_time=end_time,
                    pitch=pitch_proxy,
                    confidence=avg_energy
                ))

    return sustained


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


def detect_pitch_contour(y: np.ndarray, sr: int, confidence_threshold: float = 0.8) -> List[PitchFrame]:
    """
    Track melodic pitch over time using probabilistic YIN (pYIN).

    This provides real pitch tracking for:
    - Contour-aware arrow selection (ascending melody = arrows move up)
    - Accurate hold note detection on sustained melodic content

    Args:
        y: Audio time series
        sr: Sample rate
        confidence_threshold: Minimum confidence to include a pitch frame (0.0-1.0)

    Returns:
        List of PitchFrame objects with time, frequency, MIDI note, and confidence
    """
    # Use pYIN for probabilistic pitch tracking
    # fmin/fmax cover typical vocal and instrument ranges
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz
        fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
        sr=sr
    )

    # Get time stamps for each frame
    times = librosa.times_like(f0, sr=sr)

    pitch_frames = []
    for i, (time, freq, voiced, prob) in enumerate(zip(times, f0, voiced_flag, voiced_probs)):
        # Skip unvoiced or low-confidence frames
        if not voiced or np.isnan(freq) or prob < confidence_threshold:
            continue

        # Convert frequency to MIDI note number
        midi_note = librosa.hz_to_midi(freq)

        pitch_frames.append(PitchFrame(
            time=float(time),
            frequency=float(freq),
            midi_note=float(midi_note),
            confidence=float(prob)
        ))

    return pitch_frames


def classify_onsets_by_band(y: np.ndarray, sr: int) -> dict:
    """
    Classify onsets into frequency bands for drum/melodic separation.

    This allows prioritizing different sound sources:
    - Kick drums (low frequency) for strong downbeats
    - Snare (mid frequency) for backbeats
    - Hi-hats (high frequency) for subdivisions
    - Melodic content (harmonic) for pitch-following

    Args:
        y: Audio time series
        sr: Sample rate

    Returns:
        Dictionary with keys 'kick', 'snare', 'hihat', 'melodic' containing
        lists of OnsetClassification objects
    """
    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Define frequency bands for percussion classification
    # Kick: < 200 Hz
    # Snare: 200-2000 Hz
    # Hi-hat: > 6000 Hz

    # Create bandpass filtered versions of percussive signal
    # Low-pass for kick
    y_low = librosa.effects.preemphasis(y_percussive, coef=-0.97)  # Boost low frequencies
    stft_low = librosa.stft(y_percussive)
    freqs = librosa.fft_frequencies(sr=sr)

    # Create frequency masks
    kick_mask = freqs < 200
    snare_mask = (freqs >= 200) & (freqs < 2000)
    hihat_mask = freqs >= 6000

    # Apply masks and reconstruct
    stft_kick = stft_low.copy()
    stft_kick[~kick_mask, :] = 0
    y_kick = librosa.istft(stft_kick)

    stft_snare = stft_low.copy()
    stft_snare[~snare_mask, :] = 0
    y_snare = librosa.istft(stft_snare)

    stft_hihat = stft_low.copy()
    stft_hihat[~hihat_mask, :] = 0
    y_hihat = librosa.istft(stft_hihat)

    # Detect onsets in each band
    def get_onsets_with_strength(audio, sr):
        if len(audio) == 0:
            return []
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, units='frames', backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        results = []
        for frame, time in zip(onset_frames, onset_times):
            strength = onset_env[frame] / (onset_env.max() + 1e-8) if frame < len(onset_env) else 0.5
            results.append((float(time), float(strength)))
        return results

    kick_onsets = get_onsets_with_strength(y_kick, sr)
    snare_onsets = get_onsets_with_strength(y_snare, sr)
    hihat_onsets = get_onsets_with_strength(y_hihat, sr)
    melodic_onsets = get_onsets_with_strength(y_harmonic, sr)

    return {
        'kick': [OnsetClassification(time=t, band='kick', strength=s) for t, s in kick_onsets],
        'snare': [OnsetClassification(time=t, band='snare', strength=s) for t, s in snare_onsets],
        'hihat': [OnsetClassification(time=t, band='hihat', strength=s) for t, s in hihat_onsets],
        'melodic': [OnsetClassification(time=t, band='melodic', strength=s) for t, s in melodic_onsets]
    }


def detect_tempo_changes(y: np.ndarray, sr: int, min_section_duration: float = 4.0) -> List[TempoSection]:
    """
    Detect tempo variations throughout the song.

    This handles songs with tempo changes, accelerandos, and ritardandos
    by building an adaptive beat grid per section.

    Args:
        y: Audio time series
        sr: Sample rate
        min_section_duration: Minimum duration for a tempo section in seconds

    Returns:
        List of TempoSection objects with start/end times and BPM
    """
    duration = librosa.get_duration(y=y, sr=sr)

    # Get per-frame tempo estimates using tempogram
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

    # Get dominant tempo at each frame
    tempo_frames = np.argmax(tempogram, axis=0)
    # Convert tempogram bin to BPM (default tempogram has 384 bins from 0-384 BPM)
    bpm_per_frame = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
    tempo_per_frame = bpm_per_frame[tempo_frames]

    # Get frame times
    hop_length = 512
    frame_times = librosa.frames_to_time(np.arange(len(tempo_per_frame)), sr=sr, hop_length=hop_length)

    # Smooth tempo curve to reduce noise
    tempo_smooth = gaussian_filter1d(tempo_per_frame, sigma=10)

    # Find significant tempo changes (> 5% shift)
    sections = []
    section_start = 0.0
    section_tempo = float(tempo_smooth[0])

    for i, (time, tempo) in enumerate(zip(frame_times, tempo_smooth)):
        tempo = float(tempo)

        # Check for significant tempo change
        tempo_change = abs(tempo - section_tempo) / (section_tempo + 1e-8)
        section_duration = time - section_start

        if tempo_change > 0.05 and section_duration >= min_section_duration:
            # End current section and start new one
            sections.append(TempoSection(
                start_time=section_start,
                end_time=time,
                bpm=section_tempo
            ))
            section_start = time
            section_tempo = tempo

    # Add final section
    if section_start < duration:
        sections.append(TempoSection(
            start_time=section_start,
            end_time=duration,
            bpm=section_tempo
        ))

    # If no significant changes detected, return single section with global tempo
    if len(sections) == 0:
        global_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        sections.append(TempoSection(
            start_time=0.0,
            end_time=duration,
            bpm=float(global_tempo)
        ))

    return sections


def detect_phrase_boundaries(y: np.ndarray, sr: int, beats: List[Beat], tempo: float) -> List[PhraseBoundary]:
    """
    Detect musical phrase boundaries (typically 4, 8, or 16 bars).

    Phrases are detected through:
    1. Regular bar counting (every 4th and 8th bar)
    2. Novelty curve analysis (spectral changes)
    3. Energy drops indicating breath/pause points

    Args:
        y: Audio time series
        sr: Sample rate
        beats: List of Beat objects from analyze_beats()
        tempo: Tempo in BPM

    Returns:
        List of PhraseBoundary objects
    """
    duration = librosa.get_duration(y=y, sr=sr)

    # Calculate bar duration (4 beats per bar in 4/4 time)
    beat_duration = 60.0 / tempo
    bar_duration = beat_duration * 4

    # Compute novelty curve using spectral flux
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Smooth and find local maxima
    onset_smooth = gaussian_filter1d(onset_env, sigma=5)
    novelty = np.diff(onset_smooth, prepend=onset_smooth[0])
    novelty = np.maximum(novelty, 0)  # Only positive changes (onsets)

    # Normalize novelty
    novelty_norm = novelty / (novelty.max() + 1e-8)
    novelty_times = librosa.frames_to_time(np.arange(len(novelty_norm)), sr=sr)

    # Also compute energy for detecting drops
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    rms_norm = rms / (rms.max() + 1e-8)

    phrases = []
    bar_number = 0

    # Walk through song by bars
    current_time = 0.0
    while current_time < duration:
        bar_number += 1

        # Check if this is a phrase boundary (every 4 or 8 bars)
        is_4bar = bar_number % 4 == 0
        is_8bar = bar_number % 8 == 0

        if is_4bar or is_8bar:
            # Get novelty score near this point
            time_idx = np.argmin(np.abs(novelty_times - current_time))
            window = slice(max(0, time_idx - 5), min(len(novelty_norm), time_idx + 5))
            local_novelty = float(np.max(novelty_norm[window]))

            # Check for energy drop (indicates natural phrase break)
            rms_idx = np.argmin(np.abs(rms_times - current_time))
            rms_window = slice(max(0, rms_idx - 3), min(len(rms_norm), rms_idx + 3))
            local_energy = float(np.min(rms_norm[rms_window]))

            # Major boundary if: 8-bar mark OR high novelty OR low energy
            is_major = is_8bar or local_novelty > 0.6 or local_energy < 0.3

            phrase_length = 8 if is_8bar else 4

            phrases.append(PhraseBoundary(
                time=current_time,
                bar_number=bar_number,
                phrase_length=phrase_length,
                is_major=is_major,
                novelty_score=local_novelty
            ))

        current_time += bar_duration

    return phrases


def detect_musical_rests(y: np.ndarray, sr: int, min_duration: float = 0.3,
                          energy_threshold: float = 0.1) -> List[tuple]:
    """
    Detect intentional musical silence/rests.

    Rests are important for chart feel - they create breathing room
    and should not have steps placed during them.

    Args:
        y: Audio time series
        sr: Sample rate
        min_duration: Minimum duration for a rest in seconds
        energy_threshold: RMS energy threshold below which is considered silence

    Returns:
        List of (start_time, end_time) tuples representing rest periods
    """
    # Compute RMS energy with small hop for precision
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    # Normalize RMS
    rms_norm = rms / (rms.max() + 1e-8)

    # Find contiguous regions below threshold
    rests = []
    in_rest = False
    rest_start = 0.0

    for time, energy in zip(times, rms_norm):
        if energy < energy_threshold:
            if not in_rest:
                # Start of a potential rest
                in_rest = True
                rest_start = float(time)
        else:
            if in_rest:
                # End of rest
                rest_end = float(time)
                rest_duration = rest_end - rest_start

                if rest_duration >= min_duration:
                    rests.append((rest_start, rest_end))

                in_rest = False

    # Handle rest that extends to end of audio
    if in_rest:
        rest_end = float(times[-1])
        rest_duration = rest_end - rest_start
        if rest_duration >= min_duration:
            rests.append((rest_start, rest_end))

    return rests


def quantize_to_grid_with_tolerance(
    times: List[float],
    tempo: float,
    grid_division: int = 8,
    tolerance_ms: float = 0.0
) -> List[tuple]:
    """
    Quantize times to musical grid with optional off-grid tolerance.

    Unlike strict quantization, this allows notes within tolerance to stay
    on-grid while preserving expressive timing for notes outside tolerance.

    Args:
        times: List of times in seconds
        tempo: Tempo in BPM
        grid_division: Grid resolution (4=quarter, 8=eighth, 16=sixteenth)
        tolerance_ms: Tolerance in milliseconds. 0 = strict quantization.
                      Notes within tolerance snap to grid, others keep original time.

    Returns:
        List of (time, is_on_grid) tuples
    """
    if not times or tempo <= 0:
        return [(t, True) for t in times]

    tolerance_sec = tolerance_ms / 1000.0

    # Calculate grid spacing
    beat_duration = 60.0 / tempo
    grid_spacing = beat_duration / (grid_division / 4)

    results = []
    seen_times = set()

    for t in times:
        # Find nearest grid position
        grid_position = round(t / grid_spacing)
        snap_position = grid_position * grid_spacing
        distance = abs(t - snap_position)

        if tolerance_ms == 0.0 or distance <= tolerance_sec:
            # Snap to grid
            quantized_time = snap_position
            is_on_grid = True
        else:
            # Keep original off-grid time
            quantized_time = t
            is_on_grid = False

        # Avoid duplicates
        time_rounded = round(quantized_time, 4)
        if time_rounded not in seen_times:
            seen_times.add(time_rounded)
            results.append((quantized_time, is_on_grid))

    return sorted(results, key=lambda x: x[0])

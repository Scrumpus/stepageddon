"""
Audio Analysis Functions

Pure functions for analyzing audio using librosa.
All functions are deterministic - same input produces same output.

Enhanced with:
- HPSS (Harmonic-Percussive Source Separation) for better drum/melody isolation
- Frequency-band drum detection (kick vs snare vs hihat)
- pYIN pitch tracking for accurate melody detection
- Onset strength weighting for prioritizing strong transients
"""

import librosa
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional
import logging

from .schemas import Beat, EnergySection, SustainedNote, SongStructure, DrumEvent, DrumTrack, WeightedOnset

logger = logging.getLogger(__name__)


# =============================================================================
# HPSS (Harmonic-Percussive Source Separation)
# =============================================================================

def separate_harmonic_percussive(y: np.ndarray, margin: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate audio into harmonic and percussive components using HPSS.

    This is the foundation for better drum/melody detection:
    - Percussive component: drums, transients, attacks
    - Harmonic component: melody, sustained notes, chords

    Args:
        y: Audio time series
        margin: Separation margin (higher = cleaner separation, default 3.0)

    Returns:
        Tuple of (harmonic_audio, percussive_audio)
    """
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=margin)
    logger.debug(f"HPSS separation complete: harmonic RMS={np.sqrt(np.mean(y_harmonic**2)):.4f}, percussive RMS={np.sqrt(np.mean(y_percussive**2)):.4f}")
    return y_harmonic, y_percussive


# =============================================================================
# Frequency-Band Drum Detection
# =============================================================================

def _bandpass_filter(y: np.ndarray, sr: int, low_freq: float, high_freq: float) -> np.ndarray:
    """Apply bandpass filter to isolate frequency range."""
    nyquist = sr / 2
    low = max(low_freq / nyquist, 0.001)
    high = min(high_freq / nyquist, 0.999)

    if low >= high:
        return y

    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, y)


def detect_drum_events(y: np.ndarray, sr: int, y_percussive: Optional[np.ndarray] = None) -> DrumTrack:
    """
    Detect drum events by frequency band analysis.

    Separates kicks, snares, and hi-hats by their characteristic frequency ranges:
    - Kicks: 20-200 Hz (low thump)
    - Snares: 200-2000 Hz (mid crack/snap)
    - Hi-hats: 5000-15000 Hz (high sizzle)

    Args:
        y: Original audio time series
        sr: Sample rate
        y_percussive: Pre-separated percussive component (optional, will compute if not provided)

    Returns:
        DrumTrack with kicks, snares, and hihats lists
    """
    # Use percussive component for cleaner drum detection
    if y_percussive is None:
        _, y_percussive = separate_harmonic_percussive(y)

    kicks = []
    snares = []
    hihats = []

    # Kick detection: 20-200 Hz
    y_kick = _bandpass_filter(y_percussive, sr, 20, 200)
    kick_env = librosa.onset.onset_strength(y=y_kick, sr=sr)
    kick_env_norm = kick_env / (kick_env.max() + 1e-8)
    kick_frames = librosa.onset.onset_detect(
        y=y_kick, sr=sr, units='frames', backtrack=True,
        pre_max=3, post_max=3
    )
    kick_times = librosa.frames_to_time(kick_frames, sr=sr)

    for frame, time in zip(kick_frames, kick_times):
        strength = kick_env_norm[frame] if frame < len(kick_env_norm) else 0.5
        if strength > 0.3:  # Filter weak kicks
            kicks.append(DrumEvent(time=float(time), drum_type='kick', strength=float(strength)))

    # Snare detection: 200-2000 Hz
    y_snare = _bandpass_filter(y_percussive, sr, 200, 2000)
    snare_env = librosa.onset.onset_strength(y=y_snare, sr=sr)
    snare_env_norm = snare_env / (snare_env.max() + 1e-8)
    snare_frames = librosa.onset.onset_detect(
        y=y_snare, sr=sr, units='frames', backtrack=True,
        pre_max=3, post_max=3
    )
    snare_times = librosa.frames_to_time(snare_frames, sr=sr)

    for frame, time in zip(snare_frames, snare_times):
        strength = snare_env_norm[frame] if frame < len(snare_env_norm) else 0.5
        if strength > 0.25:  # Filter weak snares
            snares.append(DrumEvent(time=float(time), drum_type='snare', strength=float(strength)))

    # Hi-hat detection: 5000-15000 Hz
    y_hihat = _bandpass_filter(y_percussive, sr, 5000, min(15000, sr/2 - 100))
    hihat_env = librosa.onset.onset_strength(y=y_hihat, sr=sr)
    hihat_env_norm = hihat_env / (hihat_env.max() + 1e-8)
    hihat_frames = librosa.onset.onset_detect(
        y=y_hihat, sr=sr, units='frames', backtrack=True
    )
    hihat_times = librosa.frames_to_time(hihat_frames, sr=sr)

    for frame, time in zip(hihat_frames, hihat_times):
        strength = hihat_env_norm[frame] if frame < len(hihat_env_norm) else 0.5
        if strength > 0.2:  # Hi-hats are often subtle
            hihats.append(DrumEvent(time=float(time), drum_type='hihat', strength=float(strength)))

    logger.info(f"Detected drums: {len(kicks)} kicks, {len(snares)} snares, {len(hihats)} hi-hats")
    return DrumTrack(kicks=kicks, snares=snares, hihats=hihats)


# =============================================================================
# Onset Strength Weighting
# =============================================================================

def analyze_weighted_onsets(
    y: np.ndarray,
    sr: int,
    drum_track: Optional[DrumTrack] = None,
    strength_threshold: float = 0.3
) -> List[WeightedOnset]:
    """
    Analyze onsets with strength weighting, prioritizing strong transients.

    Uses onset envelope magnitude to rank onsets. Optionally correlates
    with drum events to flag drum-related onsets.

    Args:
        y: Audio time series
        sr: Sample rate
        drum_track: Optional DrumTrack to correlate with drum events
        strength_threshold: Minimum strength to include (0.0-1.0)

    Returns:
        List of WeightedOnset objects sorted by time
    """
    # Get onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_env_norm = onset_env / (onset_env.max() + 1e-8)

    # Detect onset frames
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, units='frames', backtrack=True,
        pre_max=3, post_max=3
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Build drum time lookup for fast correlation
    drum_times = {}
    if drum_track:
        for kick in drum_track.kicks:
            drum_times[round(kick.time, 3)] = 'kick'
        for snare in drum_track.snares:
            drum_times[round(snare.time, 3)] = 'snare'
        for hihat in drum_track.hihats:
            drum_times[round(hihat.time, 3)] = 'hihat'

    weighted_onsets = []
    for frame, time in zip(onset_frames, onset_times):
        strength = onset_env_norm[frame] if frame < len(onset_env_norm) else 0.5

        if strength < strength_threshold:
            continue

        # Check if this onset correlates with a drum event (within 30ms)
        is_drum = False
        drum_type = None
        time_rounded = round(time, 3)

        if drum_track:
            for dt, dtype in drum_times.items():
                if abs(time_rounded - dt) < 0.03:
                    is_drum = True
                    drum_type = dtype
                    break

        weighted_onsets.append(WeightedOnset(
            time=float(time),
            strength=float(strength),
            is_drum=is_drum,
            drum_type=drum_type
        ))

    # Sort by time
    weighted_onsets.sort(key=lambda o: o.time)

    logger.info(f"Analyzed {len(weighted_onsets)} weighted onsets (threshold: {strength_threshold})")
    drum_count = sum(1 for o in weighted_onsets if o.is_drum)
    logger.info(f"  {drum_count} correlated with drum events")

    return weighted_onsets


def filter_onsets_by_strength(
    weighted_onsets: List[WeightedOnset],
    percentile: float = 50.0
) -> List[WeightedOnset]:
    """
    Filter to keep only onsets above a strength percentile.

    Args:
        weighted_onsets: List of WeightedOnset objects
        percentile: Keep onsets above this percentile (0-100)

    Returns:
        Filtered list of WeightedOnset objects
    """
    if not weighted_onsets:
        return []

    strengths = [o.strength for o in weighted_onsets]
    threshold = np.percentile(strengths, percentile)

    filtered = [o for o in weighted_onsets if o.strength >= threshold]
    logger.debug(f"Filtered onsets: {len(weighted_onsets)} -> {len(filtered)} (>{percentile}th percentile)")

    return filtered


# =============================================================================
# pYIN Pitch Tracking for Melody Detection
# =============================================================================

def detect_melody_notes_pyin(
    y: np.ndarray,
    sr: int,
    y_harmonic: Optional[np.ndarray] = None,
    min_duration: float = 0.4
) -> List[SustainedNote]:
    """
    Detect sustained melodic notes using pYIN pitch tracking.

    pYIN is more accurate than spectral centroid for actual pitch detection.
    Uses the harmonic component for cleaner melody isolation.

    Args:
        y: Original audio time series
        sr: Sample rate
        y_harmonic: Pre-separated harmonic component (optional, will compute if not provided)
        min_duration: Minimum duration for a note to be considered sustained

    Returns:
        List of SustainedNote objects representing melodic holds
    """
    # Use harmonic component for cleaner pitch tracking
    if y_harmonic is None:
        y_harmonic, _ = separate_harmonic_percussive(y)

    # pYIN pitch tracking
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_harmonic,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz
        fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
        sr=sr,
        fill_na=0.0
    )

    times = librosa.times_like(f0, sr=sr)

    sustained = []
    current_note_start = None
    current_note_freqs = []

    for i, (freq, voiced, prob, time) in enumerate(zip(f0, voiced_flag, voiced_probs, times)):
        if voiced and freq > 0 and prob > 0.5:
            # We're in a voiced region
            if current_note_start is None:
                current_note_start = time
                current_note_freqs = [freq]
            else:
                current_note_freqs.append(freq)
        else:
            # End of voiced region
            if current_note_start is not None:
                duration = time - current_note_start
                if duration >= min_duration and current_note_freqs:
                    avg_freq = float(np.median(current_note_freqs))
                    confidence = float(np.mean([voiced_probs[j] for j in range(
                        max(0, i - len(current_note_freqs)), i
                    ) if j < len(voiced_probs)]))

                    sustained.append(SustainedNote(
                        start_time=float(current_note_start),
                        end_time=float(time),
                        pitch=avg_freq,  # Hz frequency
                        confidence=confidence
                    ))

                current_note_start = None
                current_note_freqs = []

    # Handle final note if song ends on a sustained note
    if current_note_start is not None and current_note_freqs:
        duration = times[-1] - current_note_start
        if duration >= min_duration:
            avg_freq = float(np.median(current_note_freqs))
            sustained.append(SustainedNote(
                start_time=float(current_note_start),
                end_time=float(times[-1]),
                pitch=avg_freq,
                confidence=0.7
            ))

    logger.info(f"pYIN detected {len(sustained)} sustained melodic notes")
    return sustained


# =============================================================================
# Beat Prominence Scoring (Drum-Aware)
# =============================================================================

def score_beat_prominence(
    beats: List[Beat],
    drum_track: DrumTrack,
    tolerance: float = 0.05
) -> List[Beat]:
    """
    Score beats by drum prominence - beats aligned with drums get higher strength.

    This makes step placement favor beats that land on actual drum hits:
    - Kick hits add significant weight (drives the pulse)
    - Snare hits add moderate weight (backbeat emphasis)
    - Hi-hat hits add slight weight (subdivision feel)

    Args:
        beats: List of Beat objects
        drum_track: DrumTrack with detected drum events
        tolerance: Time tolerance for drum alignment (seconds)

    Returns:
        List of Beat objects with updated strength values
    """
    kick_times = {k.time for k in drum_track.kicks}
    snare_times = {s.time for s in drum_track.snares}
    hihat_times = {h.time for h in drum_track.hihats}

    scored_beats = []
    for beat in beats:
        # Start with original strength
        new_strength = beat.strength

        # Check for drum alignment
        kick_hit = any(abs(beat.time - kt) < tolerance for kt in kick_times)
        snare_hit = any(abs(beat.time - st) < tolerance for st in snare_times)
        hihat_hit = any(abs(beat.time - ht) < tolerance for ht in hihat_times)

        # Add drum bonuses
        if kick_hit:
            new_strength += 0.5  # Kick = strong downbeat feel
        if snare_hit:
            new_strength += 0.3  # Snare = backbeat emphasis
        if hihat_hit:
            new_strength += 0.1  # Hi-hat = subdivision

        # Cap at 1.0 for normalization, but store raw for ranking
        scored_beats.append(Beat(
            time=beat.time,
            strength=float(min(new_strength, 2.0)),  # Allow up to 2.0 for ranking
            beat_type=beat.beat_type,
            measure_position=beat.measure_position,
            is_strong=beat.is_strong or kick_hit or snare_hit
        ))

    # Log drum alignment stats
    if beats:
        original_strength = beats[0].strength
        drum_aligned = sum(1 for b in scored_beats if b.strength > original_strength)
    else:
        drum_aligned = 0
    logger.info(f"Beat prominence scoring: {drum_aligned}/{len(beats)} beats aligned with drums")

    return scored_beats


# =============================================================================
# Grid Snapping (TODO - Commented out per request)
# =============================================================================

# TODO: Implement tempo-aware grid snapping for onset filtering
# This would snap onsets to the nearest musical grid position and filter
# onsets that are too far off-grid.
#
# def snap_to_grid(onset_time: float, tempo: float, subdivision: int = 4) -> Optional[float]:
#     """
#     Snap onset to nearest grid position, or return None if too far off-grid.
#
#     Args:
#         onset_time: Time of onset in seconds
#         tempo: Tempo in BPM
#         subdivision: Grid subdivision (4=quarter, 8=eighth, 16=sixteenth)
#
#     Returns:
#         Snapped time if within tolerance, None if too far off-grid
#     """
#     beat_duration = 60.0 / tempo
#     grid_interval = beat_duration / subdivision
#
#     nearest_grid = round(onset_time / grid_interval) * grid_interval
#
#     # 30ms tolerance for grid alignment
#     if abs(onset_time - nearest_grid) < 0.03:
#         return nearest_grid
#     return None  # Off-grid, skip
#
#
# def filter_onsets_to_grid(
#     onset_times: List[float],
#     tempo: float,
#     subdivision: int = 8
# ) -> List[float]:
#     """
#     Filter onsets to only include those that align with the musical grid.
#
#     Args:
#         onset_times: List of onset times in seconds
#         tempo: Tempo in BPM
#         subdivision: Grid subdivision (4, 8, or 16)
#
#     Returns:
#         List of grid-aligned onset times
#     """
#     grid_aligned = []
#     for t in onset_times:
#         snapped = snap_to_grid(t, tempo, subdivision)
#         if snapped is not None:
#             grid_aligned.append(snapped)
#
#     # Remove duplicates
#     return sorted(set(grid_aligned))


# =============================================================================
# Original Functions (Updated to work with HPSS)
# =============================================================================

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

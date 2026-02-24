"""
Madmom-based Audio Analysis Functions

Uses madmom's trained RNN models for beat/onset/tempo detection,
which provides more accurate rhythm analysis than librosa's heuristic-based approach.

Librosa is still used for spectral/timbral analysis (energy, pitch, structure).
"""

import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional

import madmom

from .schemas import Beat, TempoSection

logger = logging.getLogger(__name__)


class MadmomAnalysisCache:
    """
    Lazy-loads and caches madmom RNN activations per audio file.

    Each RNN processor is expensive to run, so we cache activations
    to avoid running them multiple times when multiple analysis
    functions need the same underlying data.
    """

    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self._onset_activations: Optional[np.ndarray] = None
        self._beat_activations: Optional[np.ndarray] = None
        self._downbeat_activations: Optional[np.ndarray] = None

    @property
    def onset_activations(self) -> np.ndarray:
        if self._onset_activations is None:
            logger.info("Running RNNOnsetProcessor...")
            proc = madmom.features.onsets.RNNOnsetProcessor()
            self._onset_activations = proc(self.audio_path)
            logger.info(f"Onset activations shape: {self._onset_activations.shape}")
        return self._onset_activations

    @property
    def beat_activations(self) -> np.ndarray:
        if self._beat_activations is None:
            logger.info("Running RNNBeatProcessor...")
            proc = madmom.features.beats.RNNBeatProcessor()
            self._beat_activations = proc(self.audio_path)
            logger.info(f"Beat activations shape: {self._beat_activations.shape}")
        return self._beat_activations

    @property
    def downbeat_activations(self) -> np.ndarray:
        if self._downbeat_activations is None:
            logger.info("Running RNNDownBeatProcessor...")
            proc = madmom.features.downbeats.RNNDownBeatProcessor()
            self._downbeat_activations = proc(self.audio_path)
            logger.info(f"Downbeat activations shape: {self._downbeat_activations.shape}")
        return self._downbeat_activations


def analyze_beats(audio_path: str, cache: MadmomAnalysisCache) -> Tuple[List[Beat], float]:
    """
    Detect and classify all beats using madmom's RNN downbeat tracker.

    Key improvement over librosa: real downbeat detection instead of
    assuming beat 0 is always a downbeat (i % 4 heuristic).

    Args:
        audio_path: Path to audio file
        cache: MadmomAnalysisCache instance for this audio file

    Returns:
        Tuple of (list of Beat objects, tempo in BPM)
    """
    # Use DBNDownBeatTrackingProcessor for joint beat + downbeat tracking
    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
        beats_per_bar=[4],
        fps=100
    )
    downbeat_results = proc(cache.downbeat_activations)

    # downbeat_results is Nx2 array: [time, beat_position]
    # beat_position: 1 = downbeat, 2-4 = other beats in bar

    # Get onset activations for beat strength
    onset_act = cache.onset_activations
    fps = 100  # madmom default fps

    beats = []
    beat_times = []

    for time, beat_pos in downbeat_results:
        time = float(time)
        beat_pos = int(beat_pos)
        beat_times.append(time)

        # Get onset strength at this time
        frame_idx = min(int(time * fps), len(onset_act) - 1)
        strength = float(onset_act[frame_idx]) if frame_idx >= 0 else 0.0

        # Map madmom beat position (1-indexed) to our schema (0-indexed)
        measure_position = beat_pos - 1  # 0=downbeat, 1-3=other beats

        # Classify beat type
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
            time=time,
            strength=strength,
            beat_type=beat_type,
            measure_position=measure_position,
            is_strong=is_strong
        ))

    # Calculate tempo from median inter-beat interval
    if len(beat_times) >= 2:
        ibis = np.diff(beat_times)
        median_ibi = float(np.median(ibis))
        tempo = 60.0 / median_ibi if median_ibi > 0 else 120.0
    else:
        tempo = 120.0
        logger.warning("Too few beats detected, defaulting to 120 BPM")

    logger.info(f"Madmom detected {len(beats)} beats at {tempo:.1f} BPM")
    return beats, tempo


def analyze_onsets(
    audio_path: str,
    strength_threshold: float = 0.3,
    cache: Optional[MadmomAnalysisCache] = None
) -> Tuple[List[float], np.ndarray]:
    """
    Detect all onsets using madmom's RNN onset detector.

    Args:
        audio_path: Path to audio file
        strength_threshold: Minimum onset strength (0.0-1.0) to include
        cache: Optional MadmomAnalysisCache instance

    Returns:
        Tuple of (list of onset times in seconds, onset strength envelope)
    """
    if cache is not None:
        onset_act = cache.onset_activations
    else:
        proc = madmom.features.onsets.RNNOnsetProcessor()
        onset_act = proc(audio_path)

    # Peak-pick the onset activation function
    picker = madmom.features.onsets.OnsetPeakPickingProcessor(
        threshold=strength_threshold,
        fps=100
    )
    onset_times = picker(onset_act)

    # Convert to list of floats
    onset_times_list = [float(t) for t in onset_times]

    logger.info(f"Madmom detected {len(onset_times_list)} onsets (threshold: {strength_threshold})")
    return onset_times_list, onset_act


def detect_subdivisions(
    audio_path: str,
    beat_times: List[float],
    cache: Optional[MadmomAnalysisCache] = None
) -> List[float]:
    """
    Detect subdivisions (8th/16th notes) between beats using madmom onsets.

    Reuses onset detection results to find onsets that fall between beats.

    Args:
        audio_path: Path to audio file
        beat_times: List of beat times in seconds
        cache: Optional MadmomAnalysisCache instance

    Returns:
        List of subdivision times (onsets between beats)
    """
    # Get all onsets with a low threshold to capture subdivisions
    onset_times, _ = analyze_onsets(audio_path, strength_threshold=0.2, cache=cache)

    subdivisions = []
    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]

        between = [t for t in onset_times if beat_start < t < beat_end]
        subdivisions.extend(between)

    return subdivisions


def detect_tempo_changes(
    audio_path: str,
    min_section_duration: float = 4.0,
    cache: Optional[MadmomAnalysisCache] = None
) -> List[TempoSection]:
    """
    Detect tempo variations throughout the song using madmom beat tracking.

    Uses windowed inter-beat-interval analysis with Gaussian smoothing
    to find sections with different tempos.

    Args:
        audio_path: Path to audio file
        min_section_duration: Minimum duration for a tempo section in seconds
        cache: Optional MadmomAnalysisCache instance

    Returns:
        List of TempoSection objects with start/end times and BPM
    """
    if cache is not None:
        beat_act = cache.beat_activations
    else:
        proc = madmom.features.beats.RNNBeatProcessor()
        beat_act = proc(audio_path)

    # Track beats
    beat_proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    beat_times = beat_proc(beat_act)
    beat_times = [float(t) for t in beat_times]

    if len(beat_times) < 4:
        logger.warning("Too few beats for tempo change detection")
        return [TempoSection(start_time=0.0, end_time=beat_times[-1] if beat_times else 60.0, bpm=120.0)]

    # Calculate inter-beat intervals
    ibis = np.diff(beat_times)

    # Windowed local tempo (window of 8 beats)
    window_size = min(8, len(ibis))
    local_tempos = []
    local_times = []

    for i in range(len(ibis) - window_size + 1):
        window_ibi = ibis[i:i + window_size]
        median_ibi = float(np.median(window_ibi))
        local_bpm = 60.0 / median_ibi if median_ibi > 0 else 120.0
        local_tempos.append(local_bpm)
        local_times.append(beat_times[i + window_size // 2])

    if not local_tempos:
        median_ibi = float(np.median(ibis))
        global_tempo = 60.0 / median_ibi if median_ibi > 0 else 120.0
        return [TempoSection(start_time=0.0, end_time=beat_times[-1], bpm=global_tempo)]

    # Smooth local tempo curve
    local_tempos_arr = np.array(local_tempos)
    tempo_smooth = gaussian_filter1d(local_tempos_arr, sigma=3)

    # Segment into sections based on tempo changes (> 5% shift)
    sections = []
    section_start = 0.0
    section_tempo = float(tempo_smooth[0])

    for time, tempo in zip(local_times, tempo_smooth):
        tempo = float(tempo)
        tempo_change = abs(tempo - section_tempo) / (section_tempo + 1e-8)
        section_duration = time - section_start

        if tempo_change > 0.05 and section_duration >= min_section_duration:
            sections.append(TempoSection(
                start_time=section_start,
                end_time=time,
                bpm=section_tempo
            ))
            section_start = time
            section_tempo = tempo

    # Add final section
    end_time = beat_times[-1] if beat_times else 60.0
    if section_start < end_time:
        sections.append(TempoSection(
            start_time=section_start,
            end_time=end_time,
            bpm=section_tempo
        ))

    if len(sections) == 0:
        median_ibi = float(np.median(ibis))
        global_tempo = 60.0 / median_ibi if median_ibi > 0 else 120.0
        sections.append(TempoSection(start_time=0.0, end_time=end_time, bpm=global_tempo))

    return sections

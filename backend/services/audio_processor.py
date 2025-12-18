"""
Audio Processing Service
Handles audio analysis, beat detection, and feature extraction using librosa
"""

import librosa
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio analysis and processing using librosa"""
    
    def __init__(self):
        self.sample_rate = 22050
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Comprehensive audio analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Analyzing audio file: {audio_path}")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = librosa.get_duration(y=y, sr=sr)
            
            logger.info(f"Duration: {duration:.2f}s, Sample rate: {sr}")
            
            # Beat and tempo analysis
            tempo, beats = self._detect_beats(y, sr)
            beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
            
            # Onset detection (for more precise timing)
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, units='frames', backtrack=True
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
            
            # Energy analysis
            energy_profile = self._analyze_energy(y, sr)
            
            # Spectral features (for genre/mood detection)
            spectral_features = self._extract_spectral_features(y, sr)
            
            # Structure detection
            structure = self._detect_structure(y, sr)
            
            analysis = {
                "duration": float(duration),
                "tempo": float(tempo),
                "beat_times": beat_times,
                "onset_times": onset_times,
                "energy_profile": energy_profile,
                "spectral_features": spectral_features,
                "structure": structure,
                "sample_rate": sr
            }
            
            logger.info(f"Tempo: {tempo}")
            logger.info(f"Analysis complete: Tempo={tempo:.1f} BPM, {len(beat_times)} beats")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}", exc_info=True)
            raise
    
    def _detect_beats(self, y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        """Detect beats and estimate tempo"""
        try:
            # Dynamic tempo estimation
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            [tempo], beats = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                units='frames'
            )
            
            # If tempo seems wrong, try alternative method
            if tempo < 60 or tempo > 200:
                dtempo = librosa.beat.tempo(
                    onset_envelope=onset_env,
                    sr=sr,
                    aggregate=None
                )
                tempo = np.median(dtempo)
            
            return tempo, beats
            
        except Exception as e:
            logger.warning(f"Beat detection failed: {e}")
            # Fallback: assume 120 BPM
            return 120.0, np.array([])
    
    def _analyze_energy(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze energy levels over time"""
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Split into sections for energy profile
        hop_length = 512
        frame_times = librosa.frames_to_time(
            np.arange(len(rms)), 
            sr=sr, 
            hop_length=hop_length
        )
        
        # Normalize
        rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
        
        return {
            "mean": float(np.mean(rms)),
            "max": float(np.max(rms)),
            "std": float(np.std(rms)),
            "profile": rms_normalized.tolist()[::10],  # Downsample for efficiency
            "timestamps": frame_times.tolist()[::10]
        }
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract spectral features for genre/mood detection"""
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral rolloff (frequency content)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Zero crossing rate (percussiveness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # MFCC (timbre)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            "brightness": float(np.mean(spectral_centroid)),
            "brightness_std": float(np.std(spectral_centroid)),
            "rolloff": float(np.mean(spectral_rolloff)),
            "zcr": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
            "genre_hint": self._infer_genre(
                np.mean(spectral_centroid),
                np.mean(zcr),
                np.mean(mfcc[0])
            )
        }
    
    def _infer_genre(self, brightness: float, zcr: float, mfcc0: float) -> str:
        """Simple genre inference based on features"""
        # Very basic heuristic - can be improved
        if brightness > 2000 and zcr > 0.1:
            return "electronic/rock"
        elif brightness < 1500:
            return "chill/ambient"
        elif zcr > 0.12:
            return "energetic/upbeat"
        else:
            return "general"
    
    def _detect_structure(self, y: np.ndarray, sr: int) -> Dict:
        """Detect song structure (intro, verse, chorus, etc.)"""
        try:
            # Compute chroma features for structure
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Self-similarity matrix
            rec = librosa.segment.recurrence_matrix(
                chroma,
                mode='affinity',
                metric='cosine'
            )
            
            # Detect boundaries
            boundaries = librosa.segment.agglomerative(
                chroma,
                k=8  # Number of segments
            )
            
            boundary_times = librosa.frames_to_time(boundaries, sr=sr)
            
            return {
                "boundaries": boundary_times.tolist(),
                "num_sections": len(boundary_times) - 1
            }
            
        except Exception as e:
            logger.warning(f"Structure detection failed: {e}")
            return {"boundaries": [], "num_sections": 0}
    
    def get_duration(self, audio_path: str) -> float:
        """Get audio duration without full analysis"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=1)
            duration = librosa.get_duration(path=audio_path)
            return float(duration)
        except Exception as e:
            logger.error(f"Error getting duration: {e}")
            return 0.0

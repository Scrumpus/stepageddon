/**
 * Menu feature types
 */

import { DifficultyLevel, SongInfo } from '@/types/common.types';
import { Step } from '@/features/game/types/step.types';

/**
 * Audio source type
 */
export type AudioSource = 'file' | 'audius' | 'jamendo';

/**
 * Difficulty information for display
 */
export interface DifficultyInfo {
  name: string;
  color: string;
}

/**
 * Difficulty display information
 */
export const DIFFICULTY_INFO: Record<DifficultyLevel, DifficultyInfo> = {
  beginner: {
    name: 'Beginner',
    color: 'text-green-400'
  },
  easy: {
    name: 'Easy',
    color: 'text-lime-400'
  },
  medium: {
    name: 'Medium',
    color: 'text-yellow-400'
  },
  hard: {
    name: 'Hard',
    color: 'text-orange-400'
  },
  challenge: {
    name: 'Challenge',
    color: 'text-red-500'
  }
} as const;

/**
 * Per-difficulty chart payload returned by the generation endpoints.
 */
export interface ChartPayload {
  steps: Step[];
  difficulty: string;
  tempo: number;
  duration: number;
  stats: {
    total_steps: number;
    total_arrows: number;
    tap_notes: number;
    hold_notes: number;
    singles: number;
    doubles: number;
  };
}

/**
 * API response for step generation. Charts are emitted for every difficulty
 * in a single pass; the user picks which to play after generation.
 */
export interface StepGenerationResponse {
  song_id: string;
  song_info: SongInfo;
  charts: Partial<Record<DifficultyLevel, ChartPayload>>;
  audio_url: string;
}

/**
 * Loading state
 */
export interface LoadingState {
  isLoading: boolean;
  message: string;
  progress: number;
}

/**
 * One search result from /api/search-songs (Audius + Jamendo, merged).
 */
export interface SongSearchResult {
  id: string;
  source: 'audius' | 'jamendo';
  title: string;
  artist: string;
  album: string | null;
  duration_seconds: number | null;
  thumbnail: string;
  url: string;
}

export interface SongSearchResponse {
  results: SongSearchResult[];
}

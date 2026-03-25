/**
 * Menu feature types
 */

import { DifficultyLevel, SongInfo } from '@/types/common.types';
import { Step } from '@/features/game/types/step.types';

/**
 * Audio source type
 */
export type AudioSource = 'file' | 'youtube' | 'spotify';

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
  intermediate: {
    name: 'Intermediate',
    color: 'text-yellow-400'
  },
  expert: {
    name: 'Expert',
    color: 'text-red-400'
  },
  insane: {
    name: 'Insane',
    color: 'text-red-800'
  }
} as const;

/**
 * New step generator output format
 */
export interface NewStepsResponse {
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
 * API response for step generation
 */
export interface StepGenerationResponse {
  song_info: SongInfo;
  steps: any[];                      // Legacy format (deprecated)
  new_steps?: NewStepsResponse;      // New generator output
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

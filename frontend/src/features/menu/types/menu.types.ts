/**
 * Menu feature types
 */

import { DifficultyLevel, SongInfo } from '@/types/common.types';
import { Step } from '@/features/game/types/step.types';

/**
 * Upload method selection
 */
export type UploadMethod = 'file' | 'url';

/**
 * Difficulty information for display
 */
export interface DifficultyInfo {
  name: string;
  description: string;
  color: string;
}

/**
 * Difficulty display information
 */
export const DIFFICULTY_INFO: Record<DifficultyLevel, DifficultyInfo> = {
  beginner: {
    name: 'Beginner',
    description: 'Slow pace, single arrows only',
    color: 'text-green-400'
  },
  intermediate: {
    name: 'Intermediate',
    description: 'Moderate speed, some doubles',
    color: 'text-yellow-400'
  },
  expert: {
    name: 'Expert',
    description: 'Fast, complex patterns',
    color: 'text-red-400'
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

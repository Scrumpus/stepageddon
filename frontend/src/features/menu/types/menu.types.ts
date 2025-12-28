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
 * API response for step generation
 */
export interface StepGenerationResponse {
  song_info: SongInfo;
  steps: Step[];
  new_steps?: any;      // New generator output (optional)
  new_steps_json?: any; // New generator JSON output (optional)
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

/**
 * API functions for step generation
 */

import { api, getAudioUrl } from '@/lib/axios';
import { DifficultyLevel } from '@/types/common.types';
import { StepGenerationResponse } from '../types/menu.types';

/**
 * Generate steps from uploaded audio file
 */
export async function generateStepsFromFile(
  file: File,
  difficulty: DifficultyLevel
): Promise<StepGenerationResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('difficulty', difficulty);

  const response = await api.post<StepGenerationResponse>(
    '/api/generate-steps',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
}

/**
 * Generate steps from URL (YouTube or Spotify)
 */
export async function generateStepsFromUrl(
  url: string,
  difficulty: DifficultyLevel
): Promise<StepGenerationResponse> {
  const response = await api.post<StepGenerationResponse>(
    '/api/generate-steps-url',
    {
      url,
      difficulty,
    }
  );

  return response.data;
}

export { getAudioUrl };

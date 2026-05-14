/**
 * API functions for step generation
 */

import { api, getAudioUrl } from '@/lib/axios';
import { StepGenerationResponse } from '../types/menu.types';

/**
 * Generate charts for every difficulty from an uploaded audio file.
 */
export async function generateStepsFromFile(
  file: File
): Promise<StepGenerationResponse> {
  const formData = new FormData();
  formData.append('file', file);

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
 * Generate charts for every difficulty from a URL (YouTube or Spotify).
 */
export async function generateStepsFromUrl(
  url: string
): Promise<StepGenerationResponse> {
  const response = await api.post<StepGenerationResponse>(
    '/api/generate-steps-url',
    { url }
  );

  return response.data;
}

export { getAudioUrl };

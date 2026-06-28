/**
 * API functions for step generation
 */

import { api, getAudioUrl } from '@/lib/axios';
import { StepGenerationResponse } from '../types/menu.types';

/**
 * Generate charts for every difficulty from an uploaded audio file.
 *
 * @param file - Audio file to generate charts from
 * @param style - Style profile name (e.g. 'Stream-Heavy', 'Jump-Heavy') or 'auto'
 */
export async function generateStepsFromFile(
  file: File,
  style: string = 'auto',
): Promise<StepGenerationResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('style', style);

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
 * Generate charts for every difficulty from a URL (Audius or Jamendo).
 *
 * @param url - Audius or Jamendo track URL
 * @param style - Style profile name (e.g. 'Stream-Heavy') or 'auto'
 */
export async function generateStepsFromUrl(
  url: string,
  style: string = 'auto',
): Promise<StepGenerationResponse> {
  const response = await api.post<StepGenerationResponse>(
    '/api/generate-steps-url',
    { url, style }
  );

  return response.data;
}

export { getAudioUrl };

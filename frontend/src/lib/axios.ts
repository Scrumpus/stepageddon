/**
 * Configured axios instance for API calls
 */

import axios from 'axios';

/**
 * API base URL from environment or default
 */
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Pre-configured axios instance
 */
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Helper to construct full audio URL
 * @param audioPath - Relative audio path from API
 * @returns Full audio URL
 */
export const getAudioUrl = (audioPath: string): string => {
  return `${API_BASE_URL}${audioPath}`;
};

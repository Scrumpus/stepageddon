/**
 * API configuration
 */

/**
 * API base URL from environment or default
 */
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * API endpoints
 */
export const API_ENDPOINTS = {
  GENERATE_STEPS_FILE: '/api/generate-steps',
  GENERATE_STEPS_URL: '/api/generate-steps-url',
} as const;

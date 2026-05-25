/**
 * API endpoint catalogue. The base URL lives in @/lib/axios so there's a
 * single source of truth — import API_BASE_URL from there if you need it.
 */

export { API_BASE_URL } from '@/lib/axios';

export const API_ENDPOINTS = {
  GENERATE_STEPS_FILE: '/api/generate-steps',
  GENERATE_STEPS_URL: '/api/generate-steps-url',
} as const;

/**
 * API-related TypeScript types
 */

/**
 * Base API response structure
 */
export interface ApiResponse<T = unknown> {
  data: T;
  message?: string;
  error?: string;
}

/**
 * API error response
 */
export interface ApiError {
  detail: string;
  code?: string;
  field?: string;
}

/**
 * HTTP methods
 */
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

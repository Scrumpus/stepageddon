/**
 * Results feature types
 */

import { HitAccuracy } from '@/features/game/types/game.types';

/**
 * Grade levels
 */
export type Grade = 'S' | 'A' | 'B' | 'C' | 'D' | 'F';

/**
 * Final game results
 */
export interface GameResults {
  score: number;
  maxCombo: number;
  hitAccuracy: HitAccuracy;
  accuracy: number;
  totalNotes: number;
  grade?: Grade;
}

/**
 * Grade color mapping for display
 */
export const GRADE_COLORS: Record<Grade, string> = {
  'S': 'text-yellow-400',
  'A': 'text-green-400',
  'B': 'text-blue-400',
  'C': 'text-purple-400',
  'D': 'text-orange-400',
  'F': 'text-red-400'
} as const;

/**
 * Grade thresholds (accuracy percentage)
 */
export const GRADE_THRESHOLDS = {
  S: 95,
  A: 90,
  B: 80,
  C: 70,
  D: 60,
  F: 0
} as const;

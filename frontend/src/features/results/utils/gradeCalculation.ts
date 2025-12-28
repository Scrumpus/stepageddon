/**
 * Grade calculation utilities for results
 */

import { Grade, GRADE_THRESHOLDS } from '../types';

/**
 * Calculate letter grade from accuracy percentage
 * @param accuracy - Accuracy percentage (0-100)
 * @returns Letter grade (S, A, B, C, D, or F)
 */
export const calculateGrade = (accuracy: number): Grade => {
  if (accuracy >= GRADE_THRESHOLDS.S) return 'S';
  if (accuracy >= GRADE_THRESHOLDS.A) return 'A';
  if (accuracy >= GRADE_THRESHOLDS.B) return 'B';
  if (accuracy >= GRADE_THRESHOLDS.C) return 'C';
  if (accuracy >= GRADE_THRESHOLDS.D) return 'D';
  return 'F';
};

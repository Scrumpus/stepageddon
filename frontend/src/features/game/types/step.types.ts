/**
 * Step/Note types for the game
 */

import { Direction } from '@/types/common.types';

/**
 * Step type from backend
 */
export type StepType = 'tap' | 'hold';

/**
 * Single step/note in the chart (new format from backend)
 * A step can have multiple arrows (for jumps)
 */
export interface Step {
  time: number;              // Time in seconds when note should be hit
  arrows: Direction[];       // Arrow directions (can be multiple for jumps)
  type: StepType;            // 'tap' or 'hold'
  hold_duration?: number;    // Duration for hold notes
}

/**
 * Active arrow with rendering information
 * Flattened from Step - one ActiveArrow per direction
 */
export interface ActiveArrow {
  time: number;              // Time in seconds when note should be hit
  direction: Direction;      // Single arrow direction
  type: StepType;            // 'tap' or 'hold'
  hold_duration?: number;    // Duration for hold notes
  stepIndex: number;         // Index of parent step
  arrowIndex: number;        // Index within step's arrows array
  y: number;                 // Y position for rendering
  timeUntilHit: number;      // Time until hit zone in seconds
}

/**
 * Active hold being tracked during gameplay
 */
export interface ActiveHold {
  arrowKey: string;          // Unique key: stepIndex-arrowIndex
  direction: Direction;      // Direction being held
  startTime: number;         // When hold started (audio time)
  endTime: number;           // When hold should end (audio time)
  lastTickTime: number;      // Last time we awarded hold points
  totalDuration: number;     // Original hold duration
  holdProgress: number;      // Progress 0-1 for animation
}

/**
 * Step/Note types for the game
 */

import { Direction } from '@/types/common.types';

/**
 * Single step/note in the chart
 */
export interface Step {
  time: number;           // Time in seconds when note should be hit
  direction: Direction;   // Arrow direction
  hit?: boolean;          // Whether note has been hit
  index?: number;         // Original index in steps array
}

/**
 * Active arrow with rendering information
 */
export interface ActiveArrow extends Step {
  y: number;              // Y position for rendering
  timeUntilHit: number;   // Time until hit zone in seconds
}

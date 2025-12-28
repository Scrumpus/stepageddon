/**
 * Game-specific types and constants
 */

import { Direction } from '@/types/common.types';

/**
 * Judgment levels for hit accuracy
 */
export enum Judgment {
  PERFECT = 'PERFECT',
  GOOD = 'GOOD',
  OK = 'OK',
  MISS = 'MISS'
}

/**
 * Hit accuracy tracking
 */
export interface HitAccuracy {
  perfect: number;
  good: number;
  ok: number;
  miss: number;
}

/**
 * Judgment display state
 */
export interface JudgmentDisplay {
  judgment: Judgment;
  points: number;
}

/**
 * Timing windows in milliseconds
 */
export const TIMING = {
  PERFECT: 50,   // ±50ms
  GOOD: 100,     // ±100ms
  OK: 150,       // ±150ms
  MISS: 200      // Beyond ±150ms
} as const;

/**
 * Base points per judgment
 */
export const POINTS = {
  PERFECT: 100,
  GOOD: 50,
  OK: 25,
  MISS: 0
} as const;

/**
 * Combo multipliers at specific thresholds
 */
export const COMBO_MULTIPLIER = {
  10: 1.1,
  25: 1.25,
  50: 1.5,
  100: 2.0
} as const;

/**
 * Visual/rendering configuration
 */
export const VISUAL_CONFIG = {
  ARROW_SPEED: 400,        // pixels per second
  HIT_ZONE_Y: 600,         // Y position of hit zone
  ARROW_SIZE: 80,          // Arrow size in pixels
  VISIBLE_WINDOW: 2        // Seconds of lookahead for visible arrows
} as const;

/**
 * Key mapping for arrow controls
 */
export const KEY_MAP: Record<string, Direction> = {
  'ArrowLeft': Direction.LEFT,
  'ArrowDown': Direction.DOWN,
  'ArrowUp': Direction.UP,
  'ArrowRight': Direction.RIGHT
};

/**
 * Arrow directions array for iteration
 */
export const DIRECTIONS: Direction[] = [
  Direction.LEFT,
  Direction.DOWN,
  Direction.UP,
  Direction.RIGHT
];

/**
 * Game play state (local to game screen)
 */
export interface GamePlayState {
  currentTime: number;
  score: number;
  combo: number;
  maxCombo: number;
  hitAccuracy: HitAccuracy;
  activeKeys: Record<Direction, boolean>;
}

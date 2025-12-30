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
 * Hold note scoring configuration
 */
export const HOLD_SCORING = {
  TICK_INTERVAL: 0.1,        // Award points every 100ms while holding
  POINTS_PER_TICK: 10,       // Base points per tick
  COMPLETION_BONUS: 50,      // Bonus for completing the hold
  EARLY_RELEASE_PENALTY: 0,  // No penalty, just stop scoring
} as const;

/**
 * Visual/rendering configuration
 */
export const VISUAL_CONFIG = {
  HIT_ZONE_Y: 80,          // Y position of hit zone (near top)
  ARROW_SIZE: 72,          // Arrow size in pixels
  VISIBLE_WINDOW: 2,       // Seconds of lookahead for visible arrows
  SPAWN_Y: 700,            // Y position where arrows spawn (bottom)
  // Speed is calculated based on tempo: faster tempo = faster arrows
  BASE_SPEED: 300,         // Base pixels per second at 100 BPM
  TEMPO_SPEED_FACTOR: 2.5, // Speed multiplier per BPM
} as const;

/**
 * Calculate arrow speed based on song tempo
 */
export function getArrowSpeed(tempo: number): number {
  // Scale speed with tempo: 120 BPM = ~400px/s, 180 BPM = ~550px/s
  return VISUAL_CONFIG.BASE_SPEED + (tempo - 100) * VISUAL_CONFIG.TEMPO_SPEED_FACTOR;
}

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

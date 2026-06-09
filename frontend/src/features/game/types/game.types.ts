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
  ARROW_SIZE: 90,          // Arrow size in pixels
  VISIBLE_WINDOW: 5,
  SPAWN_Y: 700,            // Y position where arrows spawn (bottom)
} as const;

export const GENERATION_TIMING_OFFSET_S =
  (typeof __GENERATION_TIMING_OFFSET_MS__ !== 'undefined'
    ? __GENERATION_TIMING_OFFSET_MS__
    : -30) / 1000;

const SPEED_MOD_BANDS: ReadonlyArray<readonly [minBpm: number, mod: number]> = [
  [410, 1.0],
  [293, 1.5],
  [240, 2.0],
  [186, 2.5],
  [157, 3.0],
  [136, 3.5],
  [120, 4.0],
  [108, 4.5],
  [98, 5.0],
  [89, 5.5],
  [82, 6.0],
  [76, 6.5],
  [71, 7.0],
  [65, 7.5],
] as const;

const LOWEST_BAND_MOD = 8.0; // bpm < 65

// Calibration: arrowSpeed = bpm * mod * SPEED_SCALAR.
// bpm*mod is ~440-610 across the bands; 0.9 targets ~450 px/s typical.
const SPEED_SCALAR = 0.9;

export function getSpeedMod(tempo: number): number {
  for (const [minBpm, mod] of SPEED_MOD_BANDS) {
    if (tempo >= minBpm) return mod;
  }
  return LOWEST_BAND_MOD;
}

export function getArrowSpeed(tempo: number): number {
  return tempo * getSpeedMod(tempo) * SPEED_SCALAR;
}

/**
 * Key mapping for player 1 (arrow keys)
 */
export const KEY_MAP: Record<string, Direction> = {
  'ArrowLeft': Direction.LEFT,
  'ArrowDown': Direction.DOWN,
  'ArrowUp': Direction.UP,
  'ArrowRight': Direction.RIGHT
};

// Player 2 uses WASD. e.key is case-sensitive, so cover both letter cases for
// when caps lock is on or shift is held.
export const KEY_MAP_P2: Record<string, Direction> = {
  a: Direction.LEFT,
  A: Direction.LEFT,
  s: Direction.DOWN,
  S: Direction.DOWN,
  w: Direction.UP,
  W: Direction.UP,
  d: Direction.RIGHT,
  D: Direction.RIGHT,
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

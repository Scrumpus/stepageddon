/**
 * Common TypeScript types shared across the entire application
 */

/**
 * Song/Track information
 */
export interface SongInfo {
  title: string;
  artist?: string;
  duration: number;
  tempo: number;
  thumbnail?: string;
  is_preview?: boolean;
}

/**
 * Difficulty levels for gameplay
 */
export type DifficultyLevel = 'beginner' | 'intermediate' | 'expert';

/**
 * Game state machine states
 */
export enum GameState {
  MENU = 'menu',
  LOADING = 'loading',
  READY = 'ready',
  PLAYING = 'playing',
  PAUSED = 'paused',
  FINISHED = 'finished'
}

/**
 * Arrow directions (DDR-style 4-panel)
 */
export enum Direction {
  LEFT = 'left',
  DOWN = 'down',
  UP = 'up',
  RIGHT = 'right'
}

/**
 * Keyboard key codes mapped to directions
 */
export type KeyCode = 'ArrowLeft' | 'ArrowDown' | 'ArrowUp' | 'ArrowRight' | 'Escape';

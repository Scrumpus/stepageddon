// Arrow directions
export const DIRECTIONS = {
  LEFT: 'left',
  DOWN: 'down',
  UP: 'up',
  RIGHT: 'right',
};

// Key mappings
export const KEY_MAP = {
  ArrowLeft: DIRECTIONS.LEFT,
  ArrowDown: DIRECTIONS.DOWN,
  ArrowUp: DIRECTIONS.UP,
  ArrowRight: DIRECTIONS.RIGHT,
};

// Game timing (in milliseconds)
export const TIMING = {
  PERFECT: 50,   // ±50ms
  GOOD: 100,     // ±100ms
  OK: 150,       // ±150ms
  MISS: 200,     // Beyond 150ms
};

// Scoring
export const POINTS = {
  PERFECT: 100,
  GOOD: 50,
  OK: 25,
  MISS: 0,
};

export const COMBO_MULTIPLIER = {
  10: 1.1,
  25: 1.25,
  50: 1.5,
  100: 2.0,
};

// Visual settings
export const ARROW_SPEED = 400; // pixels per second
export const HIT_ZONE_Y = 100; // Distance from top of game area
export const ARROW_SIZE = 80; // pixels

// Game states
export const GAME_STATES = {
  MENU: 'menu',
  LOADING: 'loading',
  READY: 'ready',
  PLAYING: 'playing',
  PAUSED: 'paused',
  FINISHED: 'finished',
};

// Difficulty settings
export const DIFFICULTY_INFO = {
  beginner: {
    name: 'Beginner',
    description: 'Slow pace, single arrows only',
    color: 'text-green-400',
  },
  intermediate: {
    name: 'Intermediate',
    description: 'Moderate speed, some doubles',
    color: 'text-yellow-400',
  },
  expert: {
    name: 'Expert',
    description: 'Fast, complex patterns',
    color: 'text-red-400',
  },
};

/**
 * Keyboard input hook - manages keyboard event listeners
 */

import { useState, useEffect, useCallback } from 'react';
import { GameState, Direction } from '@/types/common.types';
import { KEY_MAP } from '../types/game.types';

interface UseKeyboardInputParams {
  gameState: GameState;
  onArrowPress: (direction: Direction) => void;
  onArrowRelease: (direction: Direction) => void;
  onPause: () => void;
}

interface UseKeyboardInputReturn {
  activeKeys: Record<Direction, boolean>;
}

/**
 * Handle keyboard events for game controls
 * Tracks active keys for visual feedback and hold note detection
 */
export function useKeyboardInput({
  gameState,
  onArrowPress,
  onArrowRelease,
  onPause,
}: UseKeyboardInputParams): UseKeyboardInputReturn {
  const [activeKeys, setActiveKeys] = useState<Record<Direction, boolean>>({
    [Direction.LEFT]: false,
    [Direction.DOWN]: false,
    [Direction.UP]: false,
    [Direction.RIGHT]: false,
  });

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const direction = KEY_MAP[e.key];

      if (direction && gameState === GameState.PLAYING) {
        e.preventDefault();
        setActiveKeys((prev) => ({ ...prev, [direction]: true }));
        onArrowPress(direction);
      } else if (e.key === 'Escape') {
        onPause();
      }
    },
    [gameState, onArrowPress, onPause]
  );

  const handleKeyUp = useCallback(
    (e: KeyboardEvent) => {
      const direction = KEY_MAP[e.key];
      if (direction) {
        setActiveKeys((prev) => ({ ...prev, [direction]: false }));
        if (gameState === GameState.PLAYING) {
          onArrowRelease(direction);
        }
      }
    },
    [gameState, onArrowRelease]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [handleKeyDown, handleKeyUp]);

  return { activeKeys };
}

/**
 * Keyboard input hook - manages keyboard event listeners.
 * keyMap is per-player so two instances can coexist with different bindings.
 */

import { useState, useEffect, useCallback } from 'react';
import { GameState, Direction } from '@/types/common.types';

interface UseKeyboardInputParams {
  gameState: GameState;
  keyMap: Record<string, Direction>;
  onArrowPress: (direction: Direction) => void;
  onArrowRelease: (direction: Direction) => void;
  onPause: () => void;
  // Only one instance should own Escape — otherwise dual-mode pauses twice
  // and cancels itself out. Defaults true so single-player wiring is unchanged.
  enablePause?: boolean;
}

interface UseKeyboardInputReturn {
  activeKeys: Record<Direction, boolean>;
}

export function useKeyboardInput({
  gameState,
  keyMap,
  onArrowPress,
  onArrowRelease,
  onPause,
  enablePause = true,
}: UseKeyboardInputParams): UseKeyboardInputReturn {
  const [activeKeys, setActiveKeys] = useState<Record<Direction, boolean>>({
    [Direction.LEFT]: false,
    [Direction.DOWN]: false,
    [Direction.UP]: false,
    [Direction.RIGHT]: false,
  });

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const direction = keyMap[e.key];

      if (direction && gameState === GameState.PLAYING) {
        e.preventDefault();
        setActiveKeys((prev) => ({ ...prev, [direction]: true }));
        onArrowPress(direction);
      } else if (enablePause && e.key === 'Escape') {
        onPause();
      }
    },
    [gameState, keyMap, onArrowPress, onPause, enablePause]
  );

  const handleKeyUp = useCallback(
    (e: KeyboardEvent) => {
      const direction = keyMap[e.key];
      if (direction) {
        setActiveKeys((prev) => ({ ...prev, [direction]: false }));
        if (gameState === GameState.PLAYING) {
          onArrowRelease(direction);
        }
      }
    },
    [gameState, keyMap, onArrowRelease]
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

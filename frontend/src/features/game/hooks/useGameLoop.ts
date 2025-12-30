/**
 * Game loop hook - manages requestAnimationFrame and visible arrows
 * CRITICAL: Maintains ±50ms timing accuracy
 */

import { useEffect, useRef, useState } from 'react';
import { GameState } from '@/types/common.types';
import { Step, ActiveArrow } from '../types/step.types';
import { VISUAL_CONFIG, getArrowSpeed } from '../types/game.types';

interface UseGameLoopParams {
  audioRef: React.RefObject<HTMLAudioElement>;
  steps: Step[];
  gameState: GameState;
  songDuration: number;
  tempo: number;
  onFinish: () => void;
  onMiss: () => void;
}

interface UseGameLoopReturn {
  currentTime: number;
  activeArrows: ActiveArrow[];
  processedStepsRef: React.MutableRefObject<Set<string>>;
}

/**
 * Core game loop using requestAnimationFrame
 * Updates visible arrows and detects misses
 */
export function useGameLoop({
  audioRef,
  steps,
  gameState,
  songDuration,
  tempo,
  onFinish,
  onMiss,
}: UseGameLoopParams): UseGameLoopReturn {
  const [currentTime, setCurrentTime] = useState(0);
  const [activeArrows, setActiveArrows] = useState<ActiveArrow[]>([]);

  const animationRef = useRef<number | null>(null);
  const processedStepsRef = useRef<Set<string>>(new Set());

  // Calculate arrow speed based on tempo
  const arrowSpeed = getArrowSpeed(tempo);

  useEffect(() => {
    // Only run game loop when playing
    if (gameState !== GameState.PLAYING) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      return;
    }

    const gameLoop = () => {
      if (!audioRef.current) return;

      const currentTime = audioRef.current.currentTime;
      setCurrentTime(currentTime);

      // Update active arrows - 2 second lookahead window
      const visibleWindow = VISUAL_CONFIG.VISIBLE_WINDOW;
      const newActiveArrows: ActiveArrow[] = [];

      steps.forEach((step, stepIndex) => {
        const timeUntilHit = step.time - currentTime;

        // Check for missed arrows (-200ms grace period)
        // Each arrow in the step is tracked separately
        step.arrows.forEach((direction, arrowIndex) => {
          const arrowKey = `${stepIndex}-${arrowIndex}`;

          // Skip if already processed
          if (processedStepsRef.current.has(arrowKey)) return;

          if (timeUntilHit < -0.2) {
            processedStepsRef.current.add(arrowKey);
            onMiss();
            return;
          }

          // Show arrows in visible window (-200ms to +2s)
          // Arrows float UP: spawn at bottom (SPAWN_Y) and move to top (HIT_ZONE_Y)
          if (timeUntilHit >= -0.2 && timeUntilHit <= visibleWindow) {
            const y = VISUAL_CONFIG.HIT_ZONE_Y + (timeUntilHit * arrowSpeed);
            newActiveArrows.push({
              time: step.time,
              direction,
              type: step.type,
              hold_duration: step.hold_duration,
              stepIndex,
              arrowIndex,
              y,
              timeUntilHit,
            });
          }
        });
      });

      setActiveArrows(newActiveArrows);

      // Check if song is finished (0.5s buffer)
      if (currentTime >= songDuration - 0.5) {
        onFinish();
        return;
      }

      animationRef.current = requestAnimationFrame(gameLoop);
    };

    animationRef.current = requestAnimationFrame(gameLoop);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [gameState, steps, songDuration, audioRef, onFinish, onMiss, arrowSpeed]);

  return {
    currentTime,
    activeArrows,
    processedStepsRef,
  };
}

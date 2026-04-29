/**
 * Game loop hook - manages requestAnimationFrame and visible arrows
 * CRITICAL: Maintains ±50ms timing accuracy
 */

import { useEffect, useRef, useState } from 'react';
import { GameState } from '@/types/common.types';
import { Step, ActiveArrow, ActiveHold } from '../types/step.types';
import { VISUAL_CONFIG, getArrowSpeed } from '../types/game.types';

interface UseGameLoopParams {
  audioRef: React.RefObject<HTMLAudioElement>;
  steps: Step[];
  gameState: GameState;
  songDuration: number;
  tempo: number;
  activeHolds: ActiveHold[];
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
  activeHolds,
  onFinish,
  onMiss,
}: UseGameLoopParams): UseGameLoopReturn {
  const [currentTime, setCurrentTime] = useState(0);
  const [activeArrows, setActiveArrows] = useState<ActiveArrow[]>([]);

  const animationRef = useRef<number | null>(null);
  const processedStepsRef = useRef<Set<string>>(new Set());

  // Stable refs for callbacks to avoid restarting the game loop
  const onFinishRef = useRef(onFinish);
  onFinishRef.current = onFinish;
  const onMissRef = useRef(onMiss);
  onMissRef.current = onMiss;
  // Read latest activeHolds inside the rAF loop without restarting it.
  const activeHoldsRef = useRef(activeHolds);
  activeHoldsRef.current = activeHolds;

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
        const isHold = step.type === 'hold' && !!step.hold_duration;
        const holdDuration = step.hold_duration ?? 0;

        // Check for missed arrows (-200ms grace period)
        // Each arrow in the step is tracked separately
        step.arrows.forEach((direction, arrowIndex) => {
          const arrowKey = `${stepIndex}-${arrowIndex}`;
          const isProcessed = processedStepsRef.current.has(arrowKey);
          const isOngoingHold =
            isHold && activeHoldsRef.current.some((h) => h.arrowKey === arrowKey);

          if (!isProcessed && timeUntilHit < -0.2) {
            processedStepsRef.current.add(arrowKey);
            onMissRef.current();
            return;
          }

          // Once processed, only keep hold notes around — and only while the
          // hold is still being tracked (so missed holds vanish, hit holds
          // keep their trail visible until release/end).
          if (isProcessed && !isOngoingHold) return;

          // Hold notes that are being held extend the visible window past 0
          // until the hold's end time, so the trail stays on screen.
          const lowerBound = isOngoingHold ? -(holdDuration + 0.2) : -0.2;

          // Show arrows in visible window (-200ms to +2s for taps; longer for
          // active holds so the trail persists through the hold duration).
          // Arrows float UP: spawn at bottom (SPAWN_Y) and move to top (HIT_ZONE_Y)
          if (timeUntilHit >= lowerBound && timeUntilHit <= visibleWindow) {
            const y = VISUAL_CONFIG.HIT_ZONE_Y + (timeUntilHit * arrowSpeed);
            newActiveArrows.push({
              time: step.time,
              direction,
              type: step.type,
              hold_duration: step.hold_duration,
              beat_subdivision: step.beat_subdivision,
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
        onFinishRef.current();
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
  }, [gameState, steps, songDuration, audioRef, arrowSpeed]);

  return {
    currentTime,
    activeArrows,
    processedStepsRef,
  };
}

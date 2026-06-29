/**
 * Game loop hook - manages requestAnimationFrame and visible arrows.
 * Pure: emits currentTime + activeArrows. Miss detection and per-player
 * processed-step tracking live in useMissTracking / useHitDetection so
 * two players can share one loop while keeping separate per-player state.
 */

import { useEffect, useRef, useState } from 'react';
import { GameState } from '@/types/common.types';
import { useGameStore } from '@/app/store/useGameStore';
import { Step, ActiveArrow, ActiveHold } from '../types/step.types';
import { VISUAL_CONFIG, getArrowSpeed } from '../types/game.types';

interface UseGameLoopParams {
  audioRef: React.RefObject<HTMLAudioElement>;
  steps: Step[];
  gameState: GameState;
  songDuration: number;
  tempo: number;
  // Union of all players' active holds; used only to extend trail visibility
  // past the hit zone while the player is still holding.
  activeHolds: ActiveHold[];
  onFinish: () => void;
}

interface UseGameLoopReturn {
  currentTime: number;
  activeArrows: ActiveArrow[];
}

export function useGameLoop({
  audioRef,
  steps,
  gameState,
  songDuration,
  tempo,
  activeHolds,
  onFinish,
}: UseGameLoopParams): UseGameLoopReturn {
  const [currentTime, setCurrentTime] = useState(0);
  const [activeArrows, setActiveArrows] = useState<ActiveArrow[]>([]);

  const animationRef = useRef<number | null>(null);

  const onFinishRef = useRef(onFinish);
  onFinishRef.current = onFinish;
  const activeHoldsRef = useRef(activeHolds);
  activeHoldsRef.current = activeHolds;

  const speedMult = useGameStore((s) => s.arrowSpeedMultiplier);
  const arrowSpeed = getArrowSpeed(tempo) * speedMult;

  useEffect(() => {
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

      const visibleWindow = VISUAL_CONFIG.VISIBLE_WINDOW;
      const newActiveArrows: ActiveArrow[] = [];

      steps.forEach((step, stepIndex) => {
        const timeUntilHit = step.time - currentTime;
        const isHold = step.type === 'hold' && !!step.hold_duration;
        const holdDuration = step.hold_duration ?? 0;

        step.arrows.forEach((direction, arrowIndex) => {
          const arrowKey = `${stepIndex}-${arrowIndex}`;
          const isOngoingHold =
            isHold && activeHoldsRef.current.some((h) => h.arrowKey === arrowKey);

          // Active holds extend the lower bound past the hit zone so the
          // trail keeps rendering through the hold's duration.
          const lowerBound = isOngoingHold ? -(holdDuration + 0.2) : -0.2;

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
  };
}

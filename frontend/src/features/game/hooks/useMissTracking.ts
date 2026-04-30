/**
 * Per-player miss tracking. Walks the chart against currentTime and marks
 * any arrow that has passed its hit window unhit, calling onMiss for each.
 *
 * Iterates `steps` directly (not activeArrows) because useGameLoop drops
 * arrows from activeArrows the moment their timeUntilHit < -0.2 — so a miss
 * predicate that filters activeArrows would never fire on those drop-offs.
 */

import { useEffect } from 'react';
import { GameState } from '@/types/common.types';
import { Step, ActiveHold } from '../types/step.types';

interface UseMissTrackingParams {
  steps: Step[];
  currentTime: number;
  processedStepsRef: React.MutableRefObject<Set<string>>;
  activeHolds: ActiveHold[];
  gameState: GameState;
  onMiss: () => void;
}

export function useMissTracking({
  steps,
  currentTime,
  processedStepsRef,
  activeHolds,
  gameState,
  onMiss,
}: UseMissTrackingParams): void {
  useEffect(() => {
    if (gameState !== GameState.PLAYING) return;

    steps.forEach((step, stepIndex) => {
      const timeUntilHit = step.time - currentTime;
      if (timeUntilHit >= -0.2) return; // still in hit window or upcoming
      step.arrows.forEach((_direction, arrowIndex) => {
        const arrowKey = `${stepIndex}-${arrowIndex}`;
        if (processedStepsRef.current.has(arrowKey)) return;
        // Don't fire a miss against an in-progress hold this player owns.
        const isOngoingHold = activeHolds.some((h) => h.arrowKey === arrowKey);
        if (isOngoingHold) return;
        processedStepsRef.current.add(arrowKey);
        onMiss();
      });
    });
  }, [steps, currentTime, processedStepsRef, activeHolds, gameState, onMiss]);
}

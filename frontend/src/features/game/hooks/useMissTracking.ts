/**
 * Per-player miss tracking. Watches activeArrows and marks any arrow that
 * has passed its hit window unhit, calling onMiss for each one.
 *
 * Split out from useGameLoop so two players can each have their own
 * processedStepsRef and miss callback while sharing one loop / one chart.
 */

import { useEffect } from 'react';
import { GameState } from '@/types/common.types';
import { ActiveArrow, ActiveHold } from '../types/step.types';

interface UseMissTrackingParams {
  activeArrows: ActiveArrow[];
  processedStepsRef: React.MutableRefObject<Set<string>>;
  activeHolds: ActiveHold[];
  gameState: GameState;
  onMiss: () => void;
}

export function useMissTracking({
  activeArrows,
  processedStepsRef,
  activeHolds,
  gameState,
  onMiss,
}: UseMissTrackingParams): void {
  useEffect(() => {
    if (gameState !== GameState.PLAYING) return;

    for (const arrow of activeArrows) {
      const arrowKey = `${arrow.stepIndex}-${arrow.arrowIndex}`;
      if (processedStepsRef.current.has(arrowKey)) continue;
      if (arrow.timeUntilHit >= -0.2) continue;
      // If this player is currently holding this exact note, the trail is
      // still active for them — don't fire a miss against an in-progress hold.
      const isOngoingHold = activeHolds.some((h) => h.arrowKey === arrowKey);
      if (isOngoingHold) continue;
      processedStepsRef.current.add(arrowKey);
      onMiss();
    }
  }, [activeArrows, processedStepsRef, activeHolds, gameState, onMiss]);
}

/**
 * Hold tracking hook - manages active hold notes and their scoring
 */

import { useState, useCallback, useRef } from 'react';
import { Direction } from '@/types/common.types';
import { ActiveHold } from '../types/step.types';
import { HOLD_SCORING } from '../types/game.types';
import { calculateHoldTickPoints, calculateHoldCompletionBonus } from '../utils/scoring';

interface UseHoldTrackingParams {
  combo: number;
  onScoreUpdate: (points: number) => void;
}

interface UseHoldTrackingReturn {
  activeHolds: ActiveHold[];
  startHold: (hold: ActiveHold) => void;
  releaseHold: (direction: Direction) => void;
  updateHolds: (currentTime: number) => void;
}

/**
 * Track active hold notes and award points over time
 */
export function useHoldTracking({
  combo,
  onScoreUpdate,
}: UseHoldTrackingParams): UseHoldTrackingReturn {
  const [activeHolds, setActiveHolds] = useState<ActiveHold[]>([]);
  const comboRef = useRef(combo);
  comboRef.current = combo;

  const startHold = useCallback((hold: ActiveHold) => {
    setActiveHolds((prev) => [...prev, hold]);
  }, []);

  const releaseHold = useCallback((direction: Direction) => {
    setActiveHolds((prev) => prev.filter((h) => h.direction !== direction));
  }, []);

  const updateHolds = useCallback(
    (currentTime: number) => {
      setActiveHolds((prev) => {
        const updated: ActiveHold[] = [];
        let totalPoints = 0;

        for (const hold of prev) {
          // Check if hold is completed
          if (currentTime >= hold.endTime) {
            // Award completion bonus
            totalPoints += calculateHoldCompletionBonus(comboRef.current);
            // Don't add to updated list - hold is done
            continue;
          }

          // Calculate progress
          const elapsed = currentTime - hold.startTime;
          const progress = Math.min(elapsed / hold.totalDuration, 1);

          // Check if we should award tick points
          const timeSinceLastTick = currentTime - hold.lastTickTime;
          let newLastTickTime = hold.lastTickTime;

          if (timeSinceLastTick >= HOLD_SCORING.TICK_INTERVAL) {
            // Calculate number of ticks since last update
            const tickCount = Math.floor(timeSinceLastTick / HOLD_SCORING.TICK_INTERVAL);
            totalPoints += calculateHoldTickPoints(comboRef.current) * tickCount;
            newLastTickTime = hold.lastTickTime + tickCount * HOLD_SCORING.TICK_INTERVAL;
          }

          updated.push({
            ...hold,
            lastTickTime: newLastTickTime,
            holdProgress: progress,
          });
        }

        // Award accumulated points
        if (totalPoints > 0) {
          onScoreUpdate(totalPoints);
        }

        return updated;
      });
    },
    [onScoreUpdate]
  );

  return {
    activeHolds,
    startHold,
    releaseHold,
    updateHolds,
  };
}

/**
 * Hit detection hook - handles arrow hit detection and scoring
 * CRITICAL: Fixes state mutation bug from original (closest.hit = true)
 */

import { useCallback } from 'react';
import { Direction } from '@/types/common.types';
import { ActiveArrow } from '../types/step.types';
import { Judgment, JudgmentDisplay, TIMING } from '../types/game.types';
import { evaluateHit, calculatePoints } from '../utils/scoring';

interface UseHitDetectionParams {
  activeArrows: ActiveArrow[];
  processedStepsRef: React.MutableRefObject<Set<string>>;
  combo: number;
  onScoreUpdate: (points: number) => void;
  onComboUpdate: (newCombo: number) => void;
  onHitAccuracyUpdate: (judgment: Judgment) => void;
  onJudgmentDisplay: (display: JudgmentDisplay) => void;
  onMiss: () => void;
}

interface UseHitDetectionReturn {
  checkHit: (direction: Direction) => void;
}

/**
 * Check for hits and update score/combo
 * Uses immutable updates (fixes mutation bug from original)
 */
export function useHitDetection({
  activeArrows,
  processedStepsRef,
  combo,
  onScoreUpdate,
  onComboUpdate,
  onHitAccuracyUpdate,
  onJudgmentDisplay,
  onMiss,
}: UseHitDetectionParams): UseHitDetectionReturn {
  const checkHit = useCallback(
    (direction: Direction) => {
      // Find hittable arrows for this direction
      const hittableArrows = activeArrows.filter((arrow) => {
        const arrowKey = `${arrow.stepIndex}-${arrow.arrowIndex}`;
        return (
          arrow.direction === direction &&
          !processedStepsRef.current.has(arrowKey) &&
          Math.abs(arrow.timeUntilHit) <= TIMING.MISS / 1000 // 200ms window
        );
      });

      if (hittableArrows.length === 0) return;

      // Find closest arrow by absolute time difference
      const closest = hittableArrows.reduce((prev, curr) =>
        Math.abs(curr.timeUntilHit) < Math.abs(prev.timeUntilHit) ? curr : prev
      );

      const timingMs = closest.timeUntilHit * 1000;
      const judgment = evaluateHit(timingMs);

      // Mark as processed
      const arrowKey = `${closest.stepIndex}-${closest.arrowIndex}`;
      processedStepsRef.current.add(arrowKey);

      // Update score and combo
      if (judgment !== Judgment.MISS) {
        const newCombo = combo + 1;
        const points = calculatePoints(judgment, newCombo);

        onScoreUpdate(points);
        onComboUpdate(newCombo);
        onHitAccuracyUpdate(judgment);

        // Show judgment feedback
        onJudgmentDisplay({ judgment, points });
      } else {
        onMiss();
      }
    },
    [
      activeArrows,
      processedStepsRef,
      combo,
      onScoreUpdate,
      onComboUpdate,
      onHitAccuracyUpdate,
      onJudgmentDisplay,
      onMiss,
    ]
  );

  return { checkHit };
}

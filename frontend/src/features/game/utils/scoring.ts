/**
 * Scoring utilities for game play
 */

import { TIMING, POINTS, COMBO_MULTIPLIER, HOLD_SCORING, Judgment, HitAccuracy } from '../types';

/**
 * Evaluate hit timing and return judgment
 * @param timingMs - Timing difference in milliseconds (can be negative)
 * @returns Judgment level (PERFECT, GOOD, OK, or MISS)
 */
export const evaluateHit = (timingMs: number): Judgment => {
  const absTime = Math.abs(timingMs);

  if (absTime <= TIMING.PERFECT) {
    return Judgment.PERFECT;
  } else if (absTime <= TIMING.GOOD) {
    return Judgment.GOOD;
  } else if (absTime <= TIMING.OK) {
    return Judgment.OK;
  } else {
    return Judgment.MISS;
  }
};

/**
 * Calculate points for a hit with combo multiplier
 * @param judgment - Hit judgment level
 * @param combo - Current combo count
 * @returns Points awarded
 */
export const calculatePoints = (judgment: Judgment, combo: number): number => {
  const basePoints = POINTS[judgment];
  const multiplier = getComboMultiplier(combo);
  return Math.floor(basePoints * multiplier);
};

/**
 * Get combo multiplier based on current combo
 * @param combo - Current combo count
 * @returns Multiplier value
 */
export const getComboMultiplier = (combo: number): number => {
  if (combo >= 100) return COMBO_MULTIPLIER[100];
  if (combo >= 50) return COMBO_MULTIPLIER[50];
  if (combo >= 25) return COMBO_MULTIPLIER[25];
  if (combo >= 10) return COMBO_MULTIPLIER[10];
  return 1.0;
};

/**
 * Calculate weighted accuracy percentage from hit breakdown
 * @param hitAccuracy - Hit accuracy breakdown
 * @returns Accuracy percentage (0-100)
 */
export const calculateAccuracy = (hitAccuracy: HitAccuracy): number => {
  const total = Object.values(hitAccuracy).reduce((sum, val) => sum + val, 0);
  if (total === 0) return 0;

  const weighted =
    (hitAccuracy.perfect * 100) +
    (hitAccuracy.good * 66) +
    (hitAccuracy.ok * 33) +
    (hitAccuracy.miss * 0);

  return (weighted / (total * 100)) * 100;
};

/**
 * Calculate hold tick points with combo multiplier
 * @param combo - Current combo count
 * @returns Points for this hold tick
 */
export const calculateHoldTickPoints = (combo: number): number => {
  const multiplier = getComboMultiplier(combo);
  return Math.floor(HOLD_SCORING.POINTS_PER_TICK * multiplier);
};

/**
 * Calculate hold completion bonus with combo multiplier
 * @param combo - Current combo count
 * @returns Bonus points for completing the hold
 */
export const calculateHoldCompletionBonus = (combo: number): number => {
  const multiplier = getComboMultiplier(combo);
  return Math.floor(HOLD_SCORING.COMPLETION_BONUS * multiplier);
};

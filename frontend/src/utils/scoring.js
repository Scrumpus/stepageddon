import { TIMING, POINTS, COMBO_MULTIPLIER } from './gameConstants';

export const evaluateHit = (timingMs) => {
  const absTime = Math.abs(timingMs);
  
  if (absTime <= TIMING.PERFECT) {
    return 'PERFECT';
  } else if (absTime <= TIMING.GOOD) {
    return 'GOOD';
  } else if (absTime <= TIMING.OK) {
    return 'OK';
  } else {
    return 'MISS';
  }
};

export const calculatePoints = (judgment, combo) => {
  const basePoints = POINTS[judgment];
  const multiplier = getComboMultiplier(combo);
  return Math.floor(basePoints * multiplier);
};

export const getComboMultiplier = (combo) => {
  if (combo >= 100) return COMBO_MULTIPLIER[100];
  if (combo >= 50) return COMBO_MULTIPLIER[50];
  if (combo >= 25) return COMBO_MULTIPLIER[25];
  if (combo >= 10) return COMBO_MULTIPLIER[10];
  return 1.0;
};

export const calculateGrade = (accuracy) => {
  if (accuracy >= 95) return 'S';
  if (accuracy >= 90) return 'A';
  if (accuracy >= 80) return 'B';
  if (accuracy >= 70) return 'C';
  if (accuracy >= 60) return 'D';
  return 'F';
};

export const calculateAccuracy = (hitAccuracy) => {
  const total = Object.values(hitAccuracy).reduce((sum, val) => sum + val, 0);
  if (total === 0) return 0;
  
  const weighted = 
    (hitAccuracy.perfect * 100) +
    (hitAccuracy.good * 66) +
    (hitAccuracy.ok * 33) +
    (hitAccuracy.miss * 0);
  
  return (weighted / (total * 100)) * 100;
};

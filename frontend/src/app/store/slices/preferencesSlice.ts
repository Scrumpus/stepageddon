import { StateCreator } from 'zustand';
import { DifficultyLevel, GameMode } from '@/types/common.types';

export type ArrowSpeedMultiplier = 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 | 1.75 | 2;

export interface PreferencesSlice {
  difficulty: DifficultyLevel;
  gameMode: GameMode;
  arrowSpeedMultiplier: ArrowSpeedMultiplier;
  setDifficulty: (difficulty: DifficultyLevel) => void;
  setGameMode: (mode: GameMode) => void;
  setArrowSpeedMultiplier: (mult: ArrowSpeedMultiplier) => void;
}

export const createPreferencesSlice: StateCreator<PreferencesSlice, [], [], PreferencesSlice> = (
  set,
) => ({
  difficulty: 'medium',
  gameMode: 'single',
  arrowSpeedMultiplier: 1,
  setDifficulty: (difficulty) => set({ difficulty }),
  setGameMode: (gameMode) => set({ gameMode }),
  setArrowSpeedMultiplier: (arrowSpeedMultiplier) => set({ arrowSpeedMultiplier }),
});

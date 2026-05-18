import { StateCreator } from 'zustand';
import { DifficultyLevel, GameMode } from '@/types/common.types';

export interface PreferencesSlice {
  difficulty: DifficultyLevel;
  gameMode: GameMode;
  setDifficulty: (difficulty: DifficultyLevel) => void;
  setGameMode: (mode: GameMode) => void;
}

export const createPreferencesSlice: StateCreator<PreferencesSlice, [], [], PreferencesSlice> = (
  set,
) => ({
  difficulty: 'medium',
  gameMode: 'single',
  setDifficulty: (difficulty) => set({ difficulty }),
  setGameMode: (gameMode) => set({ gameMode }),
});

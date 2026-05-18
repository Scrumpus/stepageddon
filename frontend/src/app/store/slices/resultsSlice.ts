import { StateCreator } from 'zustand';
import { GameResults } from '@/features/results/types/results.types';

export interface ResultsSlice {
  gameResults: GameResults | null;
  setGameResults: (results: GameResults | null) => void;
}

export const createResultsSlice: StateCreator<ResultsSlice, [], [], ResultsSlice> = (set) => ({
  gameResults: null,
  setGameResults: (gameResults) => set({ gameResults }),
});

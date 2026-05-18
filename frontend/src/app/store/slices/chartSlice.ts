import { StateCreator } from 'zustand';
import { DifficultyLevel, SongInfo } from '@/types/common.types';
import { Step } from '@/features/game/types/step.types';

export interface ChartSlice {
  songData: SongInfo | null;
  steps: Step[];
  stepsByDifficulty: Partial<Record<DifficultyLevel, Step[]>> | null;
  audioUrl: string | null;
  setSongData: (data: SongInfo | null) => void;
  setSteps: (steps: Step[]) => void;
  setStepsByDifficulty: (map: Partial<Record<DifficultyLevel, Step[]>> | null) => void;
  setAudioUrl: (url: string | null) => void;
}

export const createChartSlice: StateCreator<ChartSlice, [], [], ChartSlice> = (set) => ({
  songData: null,
  steps: [],
  stepsByDifficulty: null,
  audioUrl: null,
  setSongData: (songData) => set({ songData }),
  setSteps: (steps) => set({ steps }),
  setStepsByDifficulty: (stepsByDifficulty) => set({ stepsByDifficulty }),
  setAudioUrl: (audioUrl) => set({ audioUrl }),
});

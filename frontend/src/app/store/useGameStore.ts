import { create } from 'zustand';
import { DifficultyLevel, GameState, SongInfo } from '@/types/common.types';
import { Step } from '@/features/game/types/step.types';
import { GameResults } from '@/features/results/types/results.types';
import { StepGenerationResponse } from '@/features/menu/types/menu.types';
import { ChartSlice, createChartSlice } from './slices/chartSlice';
import { FlowSlice, createFlowSlice } from './slices/flowSlice';
import { LoadingSlice, createLoadingSlice } from './slices/loadingSlice';
import { PreferencesSlice, createPreferencesSlice } from './slices/preferencesSlice';
import { ResultsSlice, createResultsSlice } from './slices/resultsSlice';

interface GameStoreActions {
  generationStarted: (params: { message: string; progress: number }) => void;
  generationCompleted: (result: StepGenerationResponse) => void;
  enterDifficultySelect: () => void;
  chartLoadStarted: (params: { message: string; progress: number }) => void;
  chartLoaded: (params: {
    songData: SongInfo;
    steps: Step[];
    audioUrl: string | null;
  }) => void;
  difficultyPicked: (level: DifficultyLevel) => void;
  gameFinished: (results: GameResults) => void;
  resetGame: () => void;
}

export type GameStore = FlowSlice &
  ChartSlice &
  LoadingSlice &
  ResultsSlice &
  PreferencesSlice &
  GameStoreActions;

function chartsToStepsByDifficulty(
  result: StepGenerationResponse,
): Partial<Record<DifficultyLevel, Step[]>> {
  const out: Partial<Record<DifficultyLevel, Step[]>> = {};
  if (!result.charts) return out;
  for (const [key, payload] of Object.entries(result.charts)) {
    if (!payload || !Array.isArray(payload.steps)) continue;
    out[key as DifficultyLevel] = payload.steps as Step[];
  }
  return out;
}

export const useGameStore = create<GameStore>((set, get, store) => ({
  ...createFlowSlice(set, get, store),
  ...createChartSlice(set, get, store),
  ...createLoadingSlice(set, get, store),
  ...createResultsSlice(set, get, store),
  ...createPreferencesSlice(set, get, store),

  generationStarted: ({ message, progress }) =>
    set({
      gameState: GameState.LOADING,
      loadingMessage: message,
      loadingProgress: progress,
    }),

  generationCompleted: (result) =>
    set({
      loadingMessage: 'Generation complete!',
      loadingProgress: 100,
      songData: result.song_info,
      steps: [],
      stepsByDifficulty: chartsToStepsByDifficulty(result),
      audioUrl: result.audio_url ?? null,
    }),

  enterDifficultySelect: () => set({ gameState: GameState.DIFFICULTY_SELECT }),

  chartLoadStarted: ({ message, progress }) =>
    set({
      gameState: GameState.LOADING,
      loadingMessage: message,
      loadingProgress: progress,
    }),

  chartLoaded: ({ songData, steps, audioUrl }) =>
    set({
      loadingMessage: 'Ready!',
      loadingProgress: 100,
      songData,
      steps,
      audioUrl,
    }),

  difficultyPicked: (level) => {
    const map = get().stepsByDifficulty;
    const steps = map?.[level];
    if (!steps) return;
    set({
      difficulty: level,
      steps,
      gameState: GameState.READY,
    });
  },

  gameFinished: (results) =>
    set({
      gameResults: results,
      gameState: GameState.FINISHED,
    }),

  resetGame: () =>
    set({
      gameState: GameState.MENU,
      songData: null,
      steps: [],
      stepsByDifficulty: null,
      audioUrl: null,
      gameResults: null,
    }),
}));

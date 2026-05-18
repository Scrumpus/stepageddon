import { StateCreator } from 'zustand';
import { GameState } from '@/types/common.types';

export interface FlowSlice {
  gameState: GameState;
  setGameState: (state: GameState) => void;
}

export const createFlowSlice: StateCreator<FlowSlice, [], [], FlowSlice> = (set) => ({
  gameState: GameState.MENU,
  setGameState: (gameState) => set({ gameState }),
});

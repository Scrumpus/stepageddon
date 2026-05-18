import { StateCreator } from 'zustand';

export interface LoadingSlice {
  loadingMessage: string;
  loadingProgress: number;
  setLoadingMessage: (message: string) => void;
  setLoadingProgress: (progress: number) => void;
}

export const createLoadingSlice: StateCreator<LoadingSlice, [], [], LoadingSlice> = (set) => ({
  loadingMessage: '',
  loadingProgress: 0,
  setLoadingMessage: (loadingMessage) => set({ loadingMessage }),
  setLoadingProgress: (loadingProgress) => set({ loadingProgress }),
});

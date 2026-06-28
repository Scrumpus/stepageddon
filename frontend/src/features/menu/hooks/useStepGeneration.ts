/**
 * Hook for handling step generation with loading states and error handling.
 *
 * Generation now produces a chart for every difficulty in a single pass; the
 * user picks which to play on the DIFFICULTY_SELECT screen that follows.
 */

import { useState, useCallback } from 'react';
import { GameState } from '@/types/common.types';
import { generateStepsFromFile, generateStepsFromUrl, getAudioUrl } from '../api';
import { useGameStore } from '@/app/store/useGameStore';
import { useToastApi } from '@/app/providers/ToastProvider';
import { StepGenerationResponse } from '../types/menu.types';

export function useStepGeneration() {
  const generationStarted = useGameStore((s) => s.generationStarted);
  const generationCompleted = useGameStore((s) => s.generationCompleted);
  const enterDifficultySelect = useGameStore((s) => s.enterDifficultySelect);
  const setGameState = useGameStore((s) => s.setGameState);
  const setLoadingMessage = useGameStore((s) => s.setLoadingMessage);
  const setLoadingProgress = useGameStore((s) => s.setLoadingProgress);
  const { showToast } = useToastApi();

  const [isLoading, setIsLoading] = useState(false);

  const finishGeneration = useCallback(
    (result: StepGenerationResponse) => {
      generationCompleted({
        ...result,
        audio_url: getAudioUrl(result.audio_url),
      });

      setTimeout(() => {
        enterDifficultySelect();
        setIsLoading(false);
      }, 500);
    },
    [generationCompleted, enterDifficultySelect],
  );

  const handleFileUpload = useCallback(
    async (file: File, style: string = 'auto') => {
      try {
        setIsLoading(true);
        generationStarted({ message: 'Uploading audio...', progress: 25 });

        const result = await generateStepsFromFile(file, style);
        finishGeneration(result);
      } catch (error: any) {
        console.error('Upload failed:', error);
        const errorMessage =
          error.response?.data?.detail || error.message || 'Failed to generate steps';
        showToast(errorMessage, 'error');
        setGameState(GameState.MENU);
        setIsLoading(false);
      }
    },
    [finishGeneration, generationStarted, setGameState, showToast],
  );

  const handleUrlSubmit = useCallback(
    async (url: string, style: string = 'auto') => {
      try {
        setIsLoading(true);
        generationStarted({ message: 'Downloading audio...', progress: 20 });

        // Simulated progress mid-fetch
        setTimeout(() => {
          setLoadingMessage('Analyzing music...');
          setLoadingProgress(50);
        }, 1000);

        const result = await generateStepsFromUrl(url, style);
        finishGeneration(result);
      } catch (error: any) {
        console.error('URL processing failed:', error);
        const errorMessage =
          error.response?.data?.detail || error.message || 'Failed to generate steps';
        showToast(errorMessage, 'error');
        setGameState(GameState.MENU);
        setIsLoading(false);
      }
    },
    [
      finishGeneration,
      generationStarted,
      setGameState,
      setLoadingMessage,
      setLoadingProgress,
      showToast,
    ],
  );

  return {
    isLoading,
    handleFileUpload,
    handleUrlSubmit,
  };
}

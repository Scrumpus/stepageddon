/**
 * Hook for handling step generation with loading states and error handling.
 *
 * Generation now produces a chart for every difficulty in a single pass; the
 * user picks which to play on the DIFFICULTY_SELECT screen that follows.
 */

import { useState, useCallback } from 'react';
import { DifficultyLevel, GameState } from '@/types/common.types';
import { generateStepsFromFile, generateStepsFromUrl, getAudioUrl } from '../api';
import { useApp } from '@/app/providers/AppProvider';
import { Step } from '@/features/game/types/step.types';
import { StepGenerationResponse } from '../types/menu.types';

/**
 * Convert the response's `charts` map into a `{difficulty: Step[]}` map for
 * AppProvider. Drops difficulties the backend didn't return.
 */
function chartsToStepsByDifficulty(
  response: StepGenerationResponse,
): Partial<Record<DifficultyLevel, Step[]>> {
  const out: Partial<Record<DifficultyLevel, Step[]>> = {};
  if (!response.charts) return out;
  for (const [key, payload] of Object.entries(response.charts)) {
    if (!payload || !Array.isArray(payload.steps)) continue;
    out[key as DifficultyLevel] = payload.steps as Step[];
  }
  return out;
}

export function useStepGeneration() {
  const {
    setSongData,
    setStepsByDifficulty,
    setSteps,
    setAudioUrl,
    setGameState,
    setLoadingMessage,
    setLoadingProgress,
    showToast,
  } = useApp();

  const [isLoading, setIsLoading] = useState(false);

  const finishGeneration = useCallback(
    (result: StepGenerationResponse) => {
      setLoadingProgress(100);
      setLoadingMessage('Generation complete!');

      setSongData(result.song_info);
      setSteps([]); // cleared until user picks a difficulty
      setStepsByDifficulty(chartsToStepsByDifficulty(result));
      setAudioUrl(getAudioUrl(result.audio_url));

      setTimeout(() => {
        setGameState(GameState.DIFFICULTY_SELECT);
        setIsLoading(false);
      }, 500);
    },
    [
      setAudioUrl,
      setGameState,
      setLoadingMessage,
      setLoadingProgress,
      setSongData,
      setSteps,
      setStepsByDifficulty,
    ],
  );

  /**
   * Handle file upload and step generation
   */
  const handleFileUpload = useCallback(
    async (file: File) => {
      try {
        setIsLoading(true);
        setGameState(GameState.LOADING);
        setLoadingMessage('Uploading audio...');
        setLoadingProgress(25);

        const result = await generateStepsFromFile(file);
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
    [finishGeneration, setGameState, setLoadingMessage, setLoadingProgress, showToast],
  );

  /**
   * Handle URL submission and step generation
   */
  const handleUrlSubmit = useCallback(
    async (url: string) => {
      try {
        setIsLoading(true);
        setGameState(GameState.LOADING);
        setLoadingMessage('Downloading audio...');
        setLoadingProgress(20);

        // Simulate progress updates
        setTimeout(() => {
          setLoadingMessage('Analyzing music...');
          setLoadingProgress(50);
        }, 1000);

        const result = await generateStepsFromUrl(url);
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
    [finishGeneration, setGameState, setLoadingMessage, setLoadingProgress, showToast],
  );

  return {
    isLoading,
    handleFileUpload,
    handleUrlSubmit,
  };
}

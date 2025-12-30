/**
 * Hook for handling step generation with loading states and error handling
 */

import { useState, useCallback } from 'react';
import { GameState } from '@/types/common.types';
import { generateStepsFromFile, generateStepsFromUrl, getAudioUrl } from '../api';
import { useApp } from '@/app/providers/AppProvider';
import { Step } from '@/features/game/types/step.types';
import { StepGenerationResponse } from '../types/menu.types';

/**
 * Extract steps from API response (prefers new_steps format)
 */
function extractSteps(response: StepGenerationResponse): Step[] {
  // Use new_steps if available
  if (response.new_steps?.steps && Array.isArray(response.new_steps.steps)) {
    return response.new_steps.steps as Step[];
  }

  // Fallback: shouldn't happen with new backend
  console.warn('No new_steps found in response, steps may be empty');
  return [];
}

export function useStepGeneration() {
  const {
    difficulty,
    setSongData,
    setSteps,
    setAudioUrl,
    setGameState,
    setLoadingMessage,
    setLoadingProgress,
    showToast,
  } = useApp();

  const [isLoading, setIsLoading] = useState(false);

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

        const result = await generateStepsFromFile(file, difficulty);

        setLoadingProgress(100);
        setLoadingMessage('Generation complete!');

        // Set data
        setSongData(result.song_info);
        setSteps(extractSteps(result));
        setAudioUrl(getAudioUrl(result.audio_url));

        // Move to ready screen
        setTimeout(() => {
          setGameState(GameState.READY);
          setIsLoading(false);
        }, 500);
      } catch (error: any) {
        console.error('Upload failed:', error);
        const errorMessage = error.response?.data?.detail || error.message || 'Failed to generate steps';
        showToast(errorMessage, 'error');
        setGameState(GameState.MENU);
        setIsLoading(false);
      }
    },
    [
      difficulty,
      setGameState,
      setLoadingMessage,
      setLoadingProgress,
      setSongData,
      setSteps,
      setAudioUrl,
      showToast,
    ]
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

        const result = await generateStepsFromUrl(url, difficulty);

        setLoadingProgress(100);
        setLoadingMessage('Generation complete!');

        // Set data
        setSongData(result.song_info);
        setSteps(extractSteps(result));
        setAudioUrl(getAudioUrl(result.audio_url));

        // Move to ready screen
        setTimeout(() => {
          setGameState(GameState.READY);
          setIsLoading(false);
        }, 500);
      } catch (error: any) {
        console.error('URL processing failed:', error);
        const errorMessage = error.response?.data?.detail || error.message || 'Failed to generate steps';
        showToast(errorMessage, 'error');
        setGameState(GameState.MENU);
        setIsLoading(false);
      }
    },
    [
      difficulty,
      setGameState,
      setLoadingMessage,
      setLoadingProgress,
      setSongData,
      setSteps,
      setAudioUrl,
      showToast,
    ]
  );

  return {
    isLoading,
    handleFileUpload,
    handleUrlSubmit,
  };
}

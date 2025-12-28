/**
 * Hook for handling step generation with loading states and error handling
 */

import { useState, useCallback } from 'react';
import { GameState } from '@/types/common.types';
import { generateStepsFromFile, generateStepsFromUrl, getAudioUrl } from '../api';
import { useApp } from '@/app/providers';

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
        setSteps(result.steps);
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
        setSteps(result.steps);
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

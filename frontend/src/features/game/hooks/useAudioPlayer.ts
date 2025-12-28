/**
 * Audio player hook - manages audio element lifecycle
 */

import { useEffect } from 'react';
import { GameState } from '@/types/common.types';

interface UseAudioPlayerParams {
  audioRef: React.RefObject<HTMLAudioElement>;
  gameState: GameState;
  shouldAutoPlay?: boolean;
}

/**
 * Manage audio playback based on game state
 */
export function useAudioPlayer({
  audioRef,
  gameState,
  shouldAutoPlay = true,
}: UseAudioPlayerParams): void {
  // Initialize and auto-play
  useEffect(() => {
    if (shouldAutoPlay && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play();
    }

    // Cleanup on unmount
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, [audioRef, shouldAutoPlay]);

  // Handle pause/resume based on game state
  useEffect(() => {
    if (!audioRef.current) return;

    if (gameState === GameState.PLAYING) {
      audioRef.current.play();
    } else if (gameState === GameState.PAUSED) {
      audioRef.current.pause();
    }
  }, [gameState, audioRef]);
}

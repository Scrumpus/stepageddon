/**
 * GameFlow — renders the correct game-state screen based on Zustand state.
 *
 * When the user presses back or finishes, we navigate back to the menu.
 * This component lives at the /game route and delegates to the existing
 * screen components.
 */

import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { GameState } from '@/types/common.types';
import { useGameStore } from '@/app/store/useGameStore';
import LoadingScreen from './LoadingScreen';
import DifficultySelectScreen from './DifficultySelectScreen';
import ReadyScreen from './ReadyScreen';
import GameScreen from '@/features/game/components/GameScreen';
import ResultsScreen from '@/features/results/components/ResultsScreen';

/**
 * When the game flow ends (back to menu), navigate to /playlists.
 * We watch for MENU state — it signals the user pressed back or finished.
 */
function useDetectMenuReturn() {
  const gameState = useGameStore((s) => s.gameState);
  const navigate = useNavigate();

  useEffect(() => {
    if (gameState === GameState.MENU) {
      navigate('/playlists', { replace: true });
    }
  }, [gameState, navigate]);
}

function GameFlow() {
  const gameState = useGameStore((s) => s.gameState);

  useDetectMenuReturn();

  switch (gameState) {
    case GameState.LOADING:
      return <LoadingScreen />;
    case GameState.DIFFICULTY_SELECT:
      return <DifficultySelectScreen />;
    case GameState.READY:
      return <ReadyScreen />;
    case GameState.PLAYING:
    case GameState.PAUSED:
      return <GameScreen />;
    case GameState.FINISHED:
      return <ResultsScreen />;
    case GameState.MENU:
      // Will be redirected by useDetectMenuReturn — render nothing briefly.
      return null;
    default:
      return null;
  }
}

export default GameFlow;

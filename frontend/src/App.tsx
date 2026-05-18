/**
 * Main App component - simplified orchestrator using feature modules
 */

import { useGameStore } from '@/app/store/useGameStore';
import { GameState } from '@/types/common.types';
import MenuScreen from '@/features/menu/components/MenuScreen';
import LoadingScreen from '@/features/menu/components/LoadingScreen';
import DifficultySelectScreen from '@/features/menu/components/DifficultySelectScreen';
import ReadyScreen from '@/features/menu/components/ReadyScreen';
import GameScreen from '@/features/game/components/GameScreen';
import ResultsScreen from '@/features/results/components/ResultsScreen';

function App() {
  const gameState = useGameStore((s) => s.gameState);

  return (
    <div className="min-h-screen bg-game-bg">
      {gameState === GameState.MENU && <MenuScreen />}
      {gameState === GameState.LOADING && <LoadingScreen />}
      {gameState === GameState.DIFFICULTY_SELECT && <DifficultySelectScreen />}
      {gameState === GameState.READY && <ReadyScreen />}
      {(gameState === GameState.PLAYING || gameState === GameState.PAUSED) && <GameScreen />}
      {gameState === GameState.FINISHED && <ResultsScreen />}
    </div>
  );
}

export default App;

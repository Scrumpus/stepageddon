/**
 * Main App component - simplified orchestrator using feature modules
 */

import { useApp } from '@/app/providers/AppProvider';
import { GameState } from '@/types/common.types';
import MenuScreen from '@/features/menu/components/MenuScreen';
import LoadingScreen from '@/features/menu/components/LoadingScreen';
import ReadyScreen from '@/features/menu/components/ReadyScreen';
import GameScreen from '@/features/game/components/GameScreen';
import ResultsScreen from '@/features/results/components/ResultsScreen';

function App() {
  const { gameState } = useApp();

  return (
    <div className="min-h-screen bg-game-bg">
      {/* Render current screen */}
      {gameState === GameState.MENU && <MenuScreen />}
      {gameState === GameState.LOADING && <LoadingScreen />}
      {gameState === GameState.READY && <ReadyScreen />}
      {(gameState === GameState.PLAYING || gameState === GameState.PAUSED) && <GameScreen />}
      {gameState === GameState.FINISHED && <ResultsScreen />}
    </div>
  );
}

export default App;

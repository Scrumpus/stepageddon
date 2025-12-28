/**
 * Main App component - simplified orchestrator using feature modules
 */

import { useApp } from '@/app/providers';
import { GameState } from '@/types/common.types';
import { MenuScreen, LoadingScreen, ReadyScreen } from '@/features/menu';
import { GameScreen } from '@/features/game';
import { ResultsScreen } from '@/features/results';

function App() {
  const { gameState } = useApp();

  return (
    <div className="min-h-screen bg-gradient-to-br from-game-bg via-purple-900/20 to-game-bg">
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

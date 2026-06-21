/**
 * Results Screen - displays game results
 * Composes GradeDisplay and StatsBreakdown components
 */

import { RotateCcw, Home, ListMusic } from 'lucide-react';
import { useGameStore } from '@/app/store/useGameStore';
import { GameState } from '@/types/common.types';
import { calculateGrade } from '@/features/results/utils/gradeCalculation';
import GradeDisplay from './GradeDisplay';
import StatsBreakdown from './StatsBreakdown';

function ResultsScreen() {
  const gameResults = useGameStore((s) => s.gameResults);
  const songData = useGameStore((s) => s.songData);
  const stepsByDifficulty = useGameStore((s) => s.stepsByDifficulty);
  const setGameState = useGameStore((s) => s.setGameState);
  const enterDifficultySelect = useGameStore((s) => s.enterDifficultySelect);
  const resetGame = useGameStore((s) => s.resetGame);

  if (!gameResults || !songData) return null;

  const grade = calculateGrade(gameResults.accuracy);
  // Only generated songs keep every difficulty's chart in memory; library
  // songs load a single chart, so there's nothing to switch back to.
  const canChangeDifficulty = !!stepsByDifficulty;

  const handlePlayAgain = () => {
    setGameState(GameState.READY);
  };

  const handleReturnToMenu = () => {
    resetGame();
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Results Card */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/10">
          <GradeDisplay grade={grade} songTitle={songData.title} />

          <StatsBreakdown results={gameResults} />

          {/* Action Buttons */}
          {canChangeDifficulty && (
            <button
              onClick={enterDifficultySelect}
              className="w-full mb-4 py-3 bg-white/10 rounded-lg font-semibold hover:bg-white/20 transition-all flex items-center justify-center gap-2"
            >
              <ListMusic className="w-5 h-5" />
              Return to Difficulty Selection
            </button>
          )}

          <div className="flex gap-4">
            <button
              onClick={handleReturnToMenu}
              className="flex-1 py-4 bg-white/10 rounded-lg font-semibold hover:bg-white/20 transition-all flex items-center justify-center gap-2"
            >
              <Home className="w-5 h-5" />
              Main Menu
            </button>

            <button
              onClick={handlePlayAgain}
              className="flex-1 py-4 bg-game-primary rounded-lg font-semibold hover:shadow-lg hover:shadow-game-accent/50 transition-all flex items-center justify-center gap-2"
            >
              <RotateCcw className="w-5 h-5" />
              Play Again
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResultsScreen;

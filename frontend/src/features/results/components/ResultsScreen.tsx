/**
 * Results Screen - displays game results
 * Composes GradeDisplay and StatsBreakdown components
 */

import { RotateCcw, Home } from 'lucide-react';
import { useApp } from '@/app/providers/AppProvider';
import { GameState } from '@/types/common.types';
import { calculateGrade } from '@/features/results/utils/gradeCalculation';
import GradeDisplay from './GradeDisplay';
import StatsBreakdown from './StatsBreakdown';

function ResultsScreen() {
  const { gameResults, songData, setGameState, resetGame } = useApp();

  if (!gameResults || !songData) return null;

  const grade = calculateGrade(gameResults.accuracy);

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
              className="flex-1 py-4 bg-gradient-to-r from-game-primary to-game-accent rounded-lg font-semibold hover:shadow-lg hover:shadow-game-accent/50 transition-all flex items-center justify-center gap-2"
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

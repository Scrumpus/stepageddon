import { Trophy, Target, Zap, RotateCcw, Home } from 'lucide-react';
import { calculateGrade } from '../utils/scoring';

function ResultsScreen({ results, songInfo, onPlayAgain, onReturnToMenu }) {
  const { score, maxCombo, hitAccuracy, accuracy, totalNotes } = results;
  const grade = calculateGrade(accuracy);
  
  const gradeColors = {
    'S': 'text-yellow-400',
    'A': 'text-green-400',
    'B': 'text-blue-400',
    'C': 'text-purple-400',
    'D': 'text-orange-400',
    'F': 'text-red-400',
  };
  
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Results Card */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/10">
          {/* Grade */}
          <div className="text-center mb-8">
            <div className={`text-8xl font-bold mb-4 ${gradeColors[grade]}`}>
              {grade}
            </div>
            <h2 className="text-2xl font-semibold mb-2">Song Complete!</h2>
            <p className="text-gray-400">{songInfo.title}</p>
          </div>
          
          {/* Main Stats */}
          <div className="grid grid-cols-3 gap-4 mb-8">
            <div className="bg-white/5 rounded-lg p-4 text-center">
              <Trophy className="w-8 h-8 mx-auto mb-2 text-yellow-400" />
              <div className="text-sm text-gray-400 mb-1">Score</div>
              <div className="text-2xl font-bold">{score.toLocaleString()}</div>
            </div>
            
            <div className="bg-white/5 rounded-lg p-4 text-center">
              <Zap className="w-8 h-8 mx-auto mb-2 text-game-accent" />
              <div className="text-sm text-gray-400 mb-1">Max Combo</div>
              <div className="text-2xl font-bold">{maxCombo}x</div>
            </div>
            
            <div className="bg-white/5 rounded-lg p-4 text-center">
              <Target className="w-8 h-8 mx-auto mb-2 text-blue-400" />
              <div className="text-sm text-gray-400 mb-1">Accuracy</div>
              <div className="text-2xl font-bold">{accuracy.toFixed(1)}%</div>
            </div>
          </div>
          
          {/* Hit Accuracy Breakdown */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold mb-4">Hit Breakdown</h3>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                  <span>Perfect</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-gray-400">
                    {((hitAccuracy.perfect / totalNotes) * 100).toFixed(1)}%
                  </div>
                  <div className="font-bold w-16 text-right">{hitAccuracy.perfect}</div>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-400"></div>
                  <span>Good</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-gray-400">
                    {((hitAccuracy.good / totalNotes) * 100).toFixed(1)}%
                  </div>
                  <div className="font-bold w-16 text-right">{hitAccuracy.good}</div>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-400"></div>
                  <span>OK</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-gray-400">
                    {((hitAccuracy.ok / totalNotes) * 100).toFixed(1)}%
                  </div>
                  <div className="font-bold w-16 text-right">{hitAccuracy.ok}</div>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-400"></div>
                  <span>Miss</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-gray-400">
                    {((hitAccuracy.miss / totalNotes) * 100).toFixed(1)}%
                  </div>
                  <div className="font-bold w-16 text-right">{hitAccuracy.miss}</div>
                </div>
              </div>
            </div>
            
            {/* Total */}
            <div className="mt-4 pt-4 border-t border-white/10 flex justify-between font-semibold">
              <span>Total Notes</span>
              <span>{totalNotes}</span>
            </div>
          </div>
          
          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              onClick={onReturnToMenu}
              className="flex-1 py-4 bg-white/10 rounded-lg font-semibold hover:bg-white/20 transition-all flex items-center justify-center gap-2"
            >
              <Home className="w-5 h-5" />
              Main Menu
            </button>
            
            <button
              onClick={onPlayAgain}
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

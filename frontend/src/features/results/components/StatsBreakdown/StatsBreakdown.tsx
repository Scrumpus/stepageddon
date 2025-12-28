/**
 * Stats breakdown - displays score, combo, accuracy and hit distribution
 */

import { Trophy, Target, Zap } from 'lucide-react';
import { GameResults } from '../../types/results.types';

interface StatsBreakdownProps {
  results: GameResults;
}

export function StatsBreakdown({ results }: StatsBreakdownProps) {
  const { score, maxCombo, hitAccuracy, accuracy, totalNotes } = results;

  return (
    <div>
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
    </div>
  );
}

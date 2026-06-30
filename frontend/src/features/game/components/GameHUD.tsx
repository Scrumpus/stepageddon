/**
 * Game HUD - displays score(s), song info, and controls.
 * Combo lives under the per-player judgment, not in the HUD.
 */

import { Pause, Play, Home } from 'lucide-react';
import { GameState, SongInfo } from '@/types/common.types';

interface GameHUDProps {
  score: number;
  // Optional P2 score; when present, the score block shows P1/P2 side by side.
  score2?: number;
  currentTime: number;
  songInfo: SongInfo;
  gameState: GameState;
  onPause: () => void;
  onReturnToMenu: () => void;
}

function GameHUD({
  score,
  score2,
  currentTime,
  songInfo,
  gameState,
  onPause,
  onReturnToMenu,
}: GameHUDProps) {
  const dualMode = score2 !== undefined;
  // Clamp to 0-100 and guard against a missing/zero duration (which would yield
  // NaN/Infinity and break the bar). currentTime can overrun the API-reported
  // duration, so cap at 100.
  const progressPercent =
    songInfo.duration > 0
      ? Math.min(100, Math.max(0, (currentTime / songInfo.duration) * 100))
      : 0;
  return (
    <>
      {/* Header */}
      <div className="bg-black/50 backdrop-blur-sm p-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex gap-2">
            <button
              onClick={onPause}
              className="p-3 bg-white/10 hover:bg-white/20 rounded-lg transition-all"
            >
              {gameState === GameState.PAUSED ? (
                <Play className="w-5 h-5" />
              ) : (
                <Pause className="w-5 h-5" />
              )}
            </button>

            <button
              onClick={onReturnToMenu}
              className="p-3 bg-white/10 hover:bg-white/20 rounded-lg transition-all"
            >
              <Home className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="text-center">
          <div className="text-lg font-semibold">{songInfo.title}</div>
          <div className="text-sm text-gray-400">
            {Math.floor(currentTime)}s / {Math.floor(songInfo.duration)}s
          </div>
        </div>

        {/* Right side intentionally left empty — Create/Settings occupy the top-right corner */}
        <div />
      </div>

      {/* Progress Bar */}
      <div className="h-1">
        <div
          className="h-full bg-game-primary"
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Footer — scores at the bottom */}
      <div className="fixed bottom-4 left-4 right-4 z-40 flex justify-between pointer-events-none">
        <div className="bg-black/50 backdrop-blur-sm rounded-xl px-6 py-3 border border-white/10">
          <div className="text-sm text-gray-400">{dualMode ? 'P1 Score' : 'Score'}</div>
          <div className="text-3xl font-bold">{score.toLocaleString()}</div>
        </div>

        {dualMode && (
          <div className="bg-black/50 backdrop-blur-sm rounded-xl px-6 py-3 border border-white/10 text-right">
            <div className="text-sm text-gray-400">P2 Score</div>
            <div className="text-3xl font-bold">{score2!.toLocaleString()}</div>
          </div>
        )}
      </div>
    </>
  );
}

export default GameHUD;

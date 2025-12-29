/**
 * Game HUD - displays score, combo, song info, and controls
 */

import { Pause, Play, Home } from 'lucide-react';
import { GameState, SongInfo } from '@/types/common.types';

interface GameHUDProps {
  score: number;
  combo: number;
  currentTime: number;
  songInfo: SongInfo;
  gameState: GameState;
  onPause: () => void;
  onReturnToMenu: () => void;
}

function GameHUD({
  score,
  combo,
  currentTime,
  songInfo,
  gameState,
  onPause,
  onReturnToMenu,
}: GameHUDProps) {
  return (
    <>
      {/* Header */}
      <div className="bg-black/50 backdrop-blur-sm p-4 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <div className="text-sm text-gray-400">Score</div>
            <div className="text-2xl font-bold">{score.toLocaleString()}</div>
          </div>

          <div>
            <div className="text-sm text-gray-400">Combo</div>
            <div className="text-2xl font-bold text-game-accent">{combo}x</div>
          </div>
        </div>

        <div className="text-center">
          <div className="text-lg font-semibold">{songInfo.title}</div>
          <div className="text-sm text-gray-400">
            {Math.floor(currentTime)}s / {Math.floor(songInfo.duration)}s
          </div>
        </div>

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

      {/* Progress Bar */}
      <div className="h-1 bg-white/10">
        <div
          className="h-full bg-gradient-to-r from-game-primary to-game-accent transition-all"
          style={{ width: `${(currentTime / songInfo.duration) * 100}%` }}
        />
      </div>
    </>
  );
}

export default GameHUD;

/**
 * Ready screen - displays song info before game starts
 */

import { Play, ArrowLeft, Music, Clock, Zap } from 'lucide-react';
import { GameState } from '@/types/common.types';
import { useApp } from '@/app/providers';
import { DIFFICULTY_INFO } from '../types/menu.types';

export function ReadyScreen() {
  const { songData, difficulty, setGameState, resetGame } = useApp();

  if (!songData) {
    return null;
  }

  const difficultyInfo = DIFFICULTY_INFO[difficulty];

  const handleStart = () => {
    setGameState(GameState.PLAYING);
  };

  const handleBack = () => {
    resetGame();
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Song Info Card */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/10">
          {/* Thumbnail (if available) */}
          {songData.thumbnail && (
            <div className="mb-6 rounded-lg overflow-hidden">
              <img
                src={songData.thumbnail}
                alt={songData.title}
                className="w-full h-48 object-cover"
              />
            </div>
          )}

          {/* Song Details */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">{songData.title}</h1>
            {songData.artist && (
              <p className="text-xl text-gray-400 mb-4">by {songData.artist}</p>
            )}

            <div className="flex flex-wrap gap-4 mt-6">
              <div className="flex items-center gap-2 bg-white/5 px-4 py-2 rounded-lg">
                <Clock className="w-4 h-4 text-game-accent" />
                <span>
                  {Math.floor(songData.duration / 60)}:
                  {String(Math.floor(songData.duration % 60)).padStart(2, '0')}
                </span>
              </div>

              <div className="flex items-center gap-2 bg-white/5 px-4 py-2 rounded-lg">
                <Zap className="w-4 h-4 text-game-accent" />
                <span>{Math.round(songData.tempo)} BPM</span>
              </div>

              <div
                className={`flex items-center gap-2 bg-white/5 px-4 py-2 rounded-lg ${difficultyInfo.color}`}
              >
                <Music className="w-4 h-4" />
                <span>{difficultyInfo.name}</span>
              </div>
            </div>

            {songData.is_preview && (
              <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <p className="text-yellow-400 text-sm">
                  ℹ️ This is a 30-second preview from Spotify
                </p>
              </div>
            )}
          </div>

          {/* Instructions */}
          <div className="mb-8 p-4 bg-white/5 rounded-lg">
            <h3 className="font-semibold mb-3">How to Play</h3>
            <div className="space-y-2 text-sm text-gray-300">
              <p>• Press arrow keys (←↓↑→) when arrows reach the target zone</p>
              <p>• Hit perfectly timed notes for maximum points</p>
              <p>• Build combos for score multipliers</p>
              <p>• Press ESC to pause during gameplay</p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              onClick={handleBack}
              className="flex-1 py-4 bg-white/10 rounded-lg font-semibold hover:bg-white/20 transition-all flex items-center justify-center gap-2"
            >
              <ArrowLeft className="w-5 h-5" />
              Back
            </button>

            <button
              onClick={handleStart}
              className="flex-1 py-4 bg-gradient-to-r from-game-primary to-game-accent rounded-lg font-semibold hover:shadow-lg hover:shadow-game-accent/50 transition-all flex items-center justify-center gap-2"
            >
              <Play className="w-5 h-5" />
              Start Game
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

import { Play, ArrowLeft, Music, Clock, Zap } from 'lucide-react';
import { DIFFICULTY_INFO } from '../utils/gameConstants';

function ReadyScreen({ songInfo, difficulty, onStart, onBack }) {
  const difficultyInfo = DIFFICULTY_INFO[difficulty];
  
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Song Info Card */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/10">
          {/* Thumbnail (if available) */}
          {songInfo.thumbnail && (
            <div className="mb-6 rounded-lg overflow-hidden">
              <img
                src={songInfo.thumbnail}
                alt={songInfo.title}
                className="w-full h-48 object-cover"
              />
            </div>
          )}
          
          {/* Song Details */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">{songInfo.title}</h1>
            {songInfo.artist && (
              <p className="text-xl text-gray-400 mb-4">by {songInfo.artist}</p>
            )}
            
            <div className="flex flex-wrap gap-4 mt-6">
              <div className="flex items-center gap-2 bg-white/5 px-4 py-2 rounded-lg">
                <Clock className="w-4 h-4 text-game-accent" />
                <span>{Math.floor(songInfo.duration / 60)}:{String(Math.floor(songInfo.duration % 60)).padStart(2, '0')}</span>
              </div>
              
              <div className="flex items-center gap-2 bg-white/5 px-4 py-2 rounded-lg">
                <Zap className="w-4 h-4 text-game-accent" />
                <span>{Math.round(songInfo.tempo)} BPM</span>
              </div>
              
              <div className={`flex items-center gap-2 bg-white/5 px-4 py-2 rounded-lg ${difficultyInfo.color}`}>
                <Music className="w-4 h-4" />
                <span>{difficultyInfo.name}</span>
              </div>
            </div>
            
            {songInfo.is_preview && (
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
              onClick={onBack}
              className="flex-1 py-4 bg-white/10 rounded-lg font-semibold hover:bg-white/20 transition-all flex items-center justify-center gap-2"
            >
              <ArrowLeft className="w-5 h-5" />
              Back
            </button>
            
            <button
              onClick={onStart}
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

export default ReadyScreen;

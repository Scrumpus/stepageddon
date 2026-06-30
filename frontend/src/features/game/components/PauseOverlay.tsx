/**
 * Pause overlay - shown when game is paused.
 * Single/dual mode is now toggled via the Settings gear (top-right).
 */

import { Play, ListMusic } from 'lucide-react';

interface PauseOverlayProps {
  onResume: () => void;
  onChangeDifficulty?: () => void;
}

function PauseOverlay({ onResume, onChangeDifficulty }: PauseOverlayProps) {
  return (
    <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="text-center">
        <h2 className="text-4xl font-bold mb-4">Paused</h2>
        <p className="text-gray-400 mb-8">Press ESC or click Play to resume</p>

        <div className="flex flex-col items-center gap-3">
          <button
            onClick={onResume}
            className="px-8 py-4 bg-game-primary rounded-lg font-semibold hover:shadow-lg transition-all"
          >
            <Play className="w-6 h-6 inline mr-2" />
            Resume
          </button>

          {onChangeDifficulty && (
            <button
              onClick={onChangeDifficulty}
              className="px-8 py-3 bg-white/10 rounded-lg font-semibold hover:bg-white/20 transition-all"
            >
              <ListMusic className="w-5 h-5 inline mr-2" />
              Return to Difficulty Selection
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default PauseOverlay;

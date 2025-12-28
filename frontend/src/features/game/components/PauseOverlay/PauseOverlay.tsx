/**
 * Pause overlay - shown when game is paused
 */

import { Play } from 'lucide-react';

interface PauseOverlayProps {
  onResume: () => void;
}

export function PauseOverlay({ onResume }: PauseOverlayProps) {
  return (
    <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="text-center">
        <h2 className="text-4xl font-bold mb-4">Paused</h2>
        <p className="text-gray-400 mb-8">Press ESC or click Play to resume</p>
        <button
          onClick={onResume}
          className="px-8 py-4 bg-gradient-to-r from-game-primary to-game-accent rounded-lg font-semibold hover:shadow-lg transition-all"
        >
          <Play className="w-6 h-6 inline mr-2" />
          Resume
        </button>
      </div>
    </div>
  );
}

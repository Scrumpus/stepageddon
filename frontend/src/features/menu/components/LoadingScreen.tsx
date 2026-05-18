/**
 * Loading screen with progress indicator
 */

import { Loader2 } from 'lucide-react';
import { useGameStore } from '@/app/store/useGameStore';
import ProgressBar from '@/components/ProgressBar';

function LoadingScreen() {
  const loadingMessage = useGameStore((s) => s.loadingMessage);
  const loadingProgress = useGameStore((s) => s.loadingProgress);

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="w-16 h-16 mx-auto mb-6 animate-spin text-game-accent" />

        <h2 className="text-2xl font-bold mb-4">{loadingMessage}</h2>

        {/* Progress Bar */}
        <div className="w-80 mx-auto">
          <ProgressBar progress={loadingProgress} />
        </div>

        <p className="mt-4 text-gray-400">
          This may take up to 30 seconds...
        </p>
      </div>
    </div>
  );
}

export default LoadingScreen;

/**
 * Loading screen with progress indicator
 */

import { Loader2 } from 'lucide-react';
import { useApp } from '@/app/providers';
import { ProgressBar } from '@/components/ui';

export function LoadingScreen() {
  const { loadingMessage, loadingProgress } = useApp();

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

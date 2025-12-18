import { Loader2 } from 'lucide-react';

function LoadingScreen({ message, progress }) {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="w-16 h-16 mx-auto mb-6 animate-spin text-game-accent" />
        
        <h2 className="text-2xl font-bold mb-4">{message}</h2>
        
        {/* Progress Bar */}
        <div className="w-80 h-2 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-game-primary to-game-accent transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
        
        <p className="mt-4 text-gray-400">
          This may take up to 30 seconds...
        </p>
      </div>
    </div>
  );
}

export default LoadingScreen;

/**
 * Main menu screen
 */

import { useApp } from '@/app/providers/AppProvider';
import { useStepGeneration } from '../hooks';
import DifficultySelector from './DifficultySelector';
import FileUploader from './FileUploader';
import UrlInput from './UrlInput';

function MenuScreen() {
  const { difficulty, setDifficulty } = useApp();
  const { handleFileUpload, handleUrlSubmit } = useStepGeneration();

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Title */}
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-game-primary via-game-secondary to-game-accent bg-clip-text text-transparent">
            Stepageddon
          </h1>
        </div>

        {/* Main Card */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/10">
          {/* Difficulty Selection */}
          <DifficultySelector
            difficulty={difficulty}
            onDifficultyChange={setDifficulty}
          />

          {/* Song Sources */}
          <div className="space-y-6">
            {/* URL Sources */}
            <div className="space-y-4">
              <UrlInput source="youtube" onUrlSubmit={handleUrlSubmit} />
              <UrlInput source="spotify" onUrlSubmit={handleUrlSubmit} />
            </div>

            {/* Divider */}
            <div className="flex items-center gap-4">
              <div className="flex-1 border-t border-white/10" />
              <span className="text-xs text-gray-500 uppercase tracking-wider">or paste a link</span>
              <div className="flex-1 border-t border-white/10" />
            </div>

            {/* File Upload */}
            <FileUploader onFileSelect={handleFileUpload} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default MenuScreen;

/**
 * Main menu screen
 */

import { useState } from 'react';
import { Upload, Globe } from 'lucide-react';
import { useApp } from '@/app/providers/AppProvider';
import { useStepGeneration } from '../hooks';
import DifficultySelector from './DifficultySelector';
import FileUploader from './FileUploader';
import UrlInput from './UrlInput';
import type { UploadMethod } from '../types/menu.types';

function MenuScreen() {
  const { difficulty, setDifficulty } = useApp();
  const { handleFileUpload, handleUrlSubmit } = useStepGeneration();
  const [uploadMethod, setUploadMethod] = useState<UploadMethod>('file');

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Title */}
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-game-primary via-game-secondary to-game-accent bg-clip-text text-transparent">
            Beat Sync
          </h1>
        </div>

        {/* Main Card */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/10">
          {/* Difficulty Selection */}
          <DifficultySelector
            difficulty={difficulty}
            onDifficultyChange={setDifficulty}
          />

          {/* Upload Method Toggle */}
          <div className="mb-6">
            <div className="flex gap-2 bg-white/5 p-1 rounded-lg">
              <button
                onClick={() => setUploadMethod('file')}
                className={`flex-1 py-2 px-4 rounded-md transition-all ${
                  uploadMethod === 'file'
                    ? 'bg-game-primary text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Upload className="w-4 h-4 inline mr-2" />
                Upload File
              </button>
              <button
                onClick={() => setUploadMethod('url')}
                className={`flex-1 py-2 px-4 rounded-md transition-all ${
                  uploadMethod === 'url'
                    ? 'bg-game-primary text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Globe className="w-4 h-4 inline mr-2" />
                URL
              </button>
            </div>
          </div>

          {/* File Upload or URL Input */}
          {uploadMethod === 'file' ? (
            <FileUploader onFileSelect={handleFileUpload} />
          ) : (
            <UrlInput onUrlSubmit={handleUrlSubmit} />
          )}
        </div>
      </div>
    </div>
  );
}

export default MenuScreen;

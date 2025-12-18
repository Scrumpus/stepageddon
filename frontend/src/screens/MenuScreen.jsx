import { useState } from 'react';
import { Upload, Music, Youtube, Globe } from 'lucide-react';
import { DIFFICULTY_INFO } from '../utils/gameConstants';

function MenuScreen({ difficulty, onDifficultyChange, onFileUpload, onUrlSubmit }) {
  const [url, setUrl] = useState('');
  const [uploadMethod, setUploadMethod] = useState('file'); // 'file' or 'url'
  
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onFileUpload(file);
    }
  };
  
  const handleUrlSubmit = (e) => {
    e.preventDefault();
    if (url.trim()) {
      onUrlSubmit(url.trim());
    }
  };
  
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Title */}
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-game-primary via-game-secondary to-game-accent bg-clip-text text-transparent">
            Beat Sync
          </h1>
          <p className="text-gray-400 text-lg">
            AI-Powered Rhythm Game
          </p>
        </div>
        
        {/* Main Card */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/10">
          {/* Difficulty Selection */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Music className="w-5 h-5" />
              Select Difficulty
            </h2>
            
            <div className="grid grid-cols-3 gap-3">
              {Object.entries(DIFFICULTY_INFO).map(([key, info]) => (
                <button
                  key={key}
                  onClick={() => onDifficultyChange(key)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    difficulty === key
                      ? 'border-game-accent bg-game-accent/20 scale-105'
                      : 'border-white/20 hover:border-white/40'
                  }`}
                >
                  <div className={`font-bold mb-1 ${info.color}`}>
                    {info.name}
                  </div>
                  <div className="text-xs text-gray-400">
                    {info.description}
                  </div>
                </button>
              ))}
            </div>
          </div>
          
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
          
          {/* File Upload */}
          {uploadMethod === 'file' && (
            <div>
              <label className="block cursor-pointer">
                <div className="border-2 border-dashed border-white/30 rounded-lg p-12 text-center hover:border-game-accent hover:bg-game-accent/5 transition-all">
                  <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                  <p className="text-lg mb-2">Drop your audio file here</p>
                  <p className="text-sm text-gray-400">
                    or click to browse
                  </p>
                  <p className="text-xs text-gray-500 mt-2">
                    Supported: MP3, WAV, OGG, FLAC (Max 50MB, 10 min)
                  </p>
                </div>
                <input
                  type="file"
                  accept=".mp3,.wav,.ogg,.flac"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>
            </div>
          )}
          
          {/* URL Input */}
          {uploadMethod === 'url' && (
            <form onSubmit={handleUrlSubmit}>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Paste YouTube or Spotify URL
                  </label>
                  <div className="relative">
                    <input
                      type="url"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      placeholder="https://youtube.com/watch?v=..."
                      className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-lg focus:outline-none focus:border-game-accent text-white"
                      required
                    />
                  </div>
                </div>
                
                <button
                  type="submit"
                  disabled={!url.trim()}
                  className="w-full py-3 bg-gradient-to-r from-game-primary to-game-accent rounded-lg font-semibold hover:shadow-lg hover:shadow-game-accent/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Generate Steps
                </button>
              </div>
              
              <div className="mt-4 flex items-center gap-4 text-sm text-gray-400">
                <div className="flex items-center gap-1">
                  <Youtube className="w-4 h-4 text-red-500" />
                  YouTube
                </div>
                <div className="flex items-center gap-1">
                  <Music className="w-4 h-4 text-green-500" />
                  Spotify
                </div>
              </div>
            </form>
          )}
        </div>
        
        {/* Info */}
        <div className="mt-6 text-center text-sm text-gray-500">
          <p>🎮 Use arrow keys (←↓↑→) to play</p>
          <p className="mt-1">Built with AI-powered step generation using Claude</p>
        </div>
      </div>
    </div>
  );
}

export default MenuScreen;

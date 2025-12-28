/**
 * URL input component for YouTube/Spotify links
 */

import { useState } from 'react';
import { Youtube, Music } from 'lucide-react';

interface UrlInputProps {
  onUrlSubmit: (url: string) => void;
}

export function UrlInput({ onUrlSubmit }: UrlInputProps) {
  const [url, setUrl] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) {
      onUrlSubmit(url.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit}>
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
  );
}

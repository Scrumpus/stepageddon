/**
 * URL input component for a specific audio source
 */

import { useState } from 'react';
import { Youtube, Music, Send } from 'lucide-react';
import type { AudioSource } from '../types/menu.types';

const SOURCE_CONFIG = {
  youtube: {
    label: 'YouTube',
    icon: Youtube,
    iconColor: 'text-red-500',
    placeholder: 'https://youtube.com/watch?v=...',
    borderColor: 'border-red-500/30',
    hoverBorder: 'hover:border-red-500/60',
    focusBorder: 'focus:border-red-500',
    buttonGradient: 'from-red-600 to-red-500',
    buttonShadow: 'hover:shadow-red-500/30',
  },
  spotify: {
    label: 'Spotify',
    icon: Music,
    iconColor: 'text-green-500',
    placeholder: 'https://open.spotify.com/track/...',
    borderColor: 'border-green-500/30',
    hoverBorder: 'hover:border-green-500/60',
    focusBorder: 'focus:border-green-500',
    buttonGradient: 'from-green-600 to-green-500',
    buttonShadow: 'hover:shadow-green-500/30',
  },
} as const;

interface UrlInputProps {
  source: Exclude<AudioSource, 'file'>;
  onUrlSubmit: (url: string) => void;
}

function UrlInput({ source, onUrlSubmit }: UrlInputProps) {
  const [url, setUrl] = useState('');
  const config = SOURCE_CONFIG[source];
  const Icon = config.icon;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) {
      onUrlSubmit(url.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="flex items-center gap-2 mb-3">
        <Icon className={`w-5 h-5 ${config.iconColor}`} />
        <span className="text-sm font-medium text-gray-300">{config.label}</span>
      </div>
      <div className="flex gap-2">
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder={config.placeholder}
          className={`flex-1 px-4 py-3 bg-white/5 border ${config.borderColor} rounded-lg focus:outline-none ${config.focusBorder} text-white text-sm`}
          required
        />
        <button
          type="submit"
          disabled={!url.trim()}
          className={`px-4 py-3 bg-gradient-to-r ${config.buttonGradient} rounded-lg font-semibold hover:shadow-lg ${config.buttonShadow} transition-all disabled:opacity-30 disabled:cursor-not-allowed`}
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </form>
  );
}

export default UrlInput;

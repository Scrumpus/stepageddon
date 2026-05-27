/**
 * URL input component for a specific audio source
 */

import { useState } from 'react';
import { Music, Disc3, Send } from 'lucide-react';
import type { AudioSource } from '../types/menu.types';

const SOURCE_CONFIG = {
  audius: {
    label: 'Audius',
    icon: Music,
    iconColor: 'text-purple-400',
    placeholder: 'https://audius.co/artist/track-name',
    borderColor: 'border-purple-500/30',
    hoverBorder: 'hover:border-purple-500/60',
    focusBorder: 'focus:border-purple-500',
    buttonGradient: 'from-purple-600 to-purple-500',
    buttonShadow: 'hover:shadow-purple-500/30',
  },
  jamendo: {
    label: 'Jamendo',
    icon: Disc3,
    iconColor: 'text-orange-400',
    placeholder: 'https://www.jamendo.com/track/...',
    borderColor: 'border-orange-500/30',
    hoverBorder: 'hover:border-orange-500/60',
    focusBorder: 'focus:border-orange-500',
    buttonGradient: 'from-orange-600 to-orange-500',
    buttonShadow: 'hover:shadow-orange-500/30',
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

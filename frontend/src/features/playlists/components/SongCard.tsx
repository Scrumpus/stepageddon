/**
 * Discover-style song card.
 *
 * Shows a song's jacket + details in a grid tile (mirrors the Discover tab's
 * TrackCard). Clicking it hands the song up so the parent can route to the
 * difficulty selector.
 */

import { Music } from 'lucide-react';
import { getAudioUrl } from '@/lib/axios';
import { SongSummaryDTO } from '../types';

interface Props {
  song: SongSummaryDTO;
  onSelect: (song: SongSummaryDTO) => void;
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function SongCard({ song, onSelect }: Props) {
  const thumb = song.jacket_url || song.banner_url;
  return (
    <button
      type="button"
      onClick={() => onSelect(song)}
      className="group text-left bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg overflow-hidden transition-colors"
    >
      <div className="aspect-square w-full bg-white/10 overflow-hidden">
        {thumb ? (
          <img
            src={getAudioUrl(thumb)}
            alt=""
            className="w-full h-full object-cover group-hover:scale-105 transition-transform"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <Music className="w-8 h-8 text-gray-500" />
          </div>
        )}
      </div>
      <div className="p-2">
        <div className="text-sm font-medium text-white truncate">
          {song.title}
        </div>
        {song.artist && (
          <div className="text-xs text-gray-400 truncate mt-0.5">
            {song.artist}
          </div>
        )}
        <div className="text-[10px] font-mono text-gray-500 mt-1">
          {Math.round(song.tempo)} BPM · {formatDuration(song.duration)}
        </div>
      </div>
    </button>
  );
}

export default SongCard;

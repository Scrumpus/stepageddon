import { ChevronRight, Music } from 'lucide-react';
import { getAudioUrl } from '@/lib/axios';
import { SongSummaryDTO } from '@/features/playlists/types';

interface Props {
  song: SongSummaryDTO;
  onSelect: (songId: string) => void;
}

function SongListItem({ song, onSelect }: Props) {
  const thumb = song.jacket_url || song.banner_url;
  return (
    <button
      onClick={() => onSelect(song.id)}
      className="w-full text-left flex items-center gap-4 p-3 bg-white/5 hover:bg-white/[0.08] rounded-xl border border-white/5 hover:border-game-primary/30 transition-colors"
    >
      <div className="w-14 h-14 rounded-lg overflow-hidden bg-white/10 flex items-center justify-center shrink-0">
        {thumb ? (
          <img
            src={getAudioUrl(thumb)}
            alt={song.title}
            className="w-full h-full object-cover"
            loading="lazy"
          />
        ) : (
          <Music className="w-6 h-6 text-white/40" />
        )}
      </div>

      <div className="flex-1 min-w-0">
        <div className="text-white font-semibold truncate">{song.title}</div>
        {song.artist && (
          <div className="text-xs text-white/50 truncate">{song.artist}</div>
        )}
        <div className="text-[11px] text-white/40 mt-0.5">
          {Math.round(song.tempo)} BPM · {Math.round(song.duration)}s
        </div>
      </div>

      <ChevronRight className="w-5 h-5 text-white/40 shrink-0" />
    </button>
  );
}

export default SongListItem;

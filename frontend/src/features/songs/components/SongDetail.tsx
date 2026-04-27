import { ArrowLeft, Loader2 } from 'lucide-react';
import { SongRow } from '@/features/playlists/components';
import { usePlayChart } from '@/features/playlists/hooks';
import { useSongDetail } from '../hooks';

interface Props {
  songId: string;
  onBack: () => void;
}

function SongDetail({ songId, onBack }: Props) {
  const { song, charts, isLoading, error } = useSongDetail(songId);
  const { playChart, isLoading: isPlaying } = usePlayChart();

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="p-2 rounded-lg hover:bg-white/10 text-white/80 transition-colors"
          aria-label="Back to songs"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold text-white truncate">
            {song?.title ?? 'Song'}
          </h2>
          {song?.artist && (
            <p className="text-sm text-white/60 truncate">{song.artist}</p>
          )}
        </div>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12 text-white/60">
          <Loader2 className="w-5 h-5 animate-spin mr-2" />
          Loading song…
        </div>
      )}

      {error && (
        <div className="py-8 text-center text-rose-300">
          Couldn't load song: {error}
        </div>
      )}

      {!isLoading && !error && song && (
        <SongRow
          song={song}
          charts={charts}
          onSelectChart={playChart}
          disabled={isPlaying}
        />
      )}
    </div>
  );
}

export default SongDetail;

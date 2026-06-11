import { useState } from 'react';
import { ArrowLeft, Loader2 } from 'lucide-react';
import { usePlayChart, usePlaylist } from '../hooks';
import ChartSelect from './ChartSelect';
import SongCard from './SongCard';

interface Props {
  playlistId: string;
  onBack: () => void;
}

function PlaylistDetail({ playlistId, onBack }: Props) {
  const { playlist, songsWithCharts, isLoading, error } = usePlaylist(playlistId);
  const { playChart, isLoading: isPlaying } = usePlayChart();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const selected = songsWithCharts.find((s) => s.song.id === selectedId);

  if (selected) {
    return (
      <ChartSelect
        song={selected.song}
        charts={selected.charts}
        onSelectChart={playChart}
        onBack={() => setSelectedId(null)}
        disabled={isPlaying}
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="p-2 rounded-lg hover:bg-white/10 text-white/80 transition-colors"
          aria-label="Back to playlists"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold text-white truncate">
            {playlist?.name ?? 'Playlist'}
          </h2>
          {playlist?.description && (
            <p className="text-sm text-white/60 truncate">{playlist.description}</p>
          )}
        </div>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12 text-white/60">
          <Loader2 className="w-5 h-5 animate-spin mr-2" />
          Loading songs…
        </div>
      )}

      {error && (
        <div className="py-8 text-center text-rose-300">
          Couldn't load playlist: {error}
        </div>
      )}

      {!isLoading && !error && songsWithCharts.length === 0 && (
        <div className="py-12 text-center text-white/60">
          This playlist is empty.
        </div>
      )}

      {!isLoading && !error && songsWithCharts.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
          {songsWithCharts.map(({ song }) => (
            <SongCard key={song.id} song={song} onSelect={() => setSelectedId(song.id)} />
          ))}
        </div>
      )}
    </div>
  );
}

export default PlaylistDetail;

import { Loader2 } from 'lucide-react';
import { usePlaylists } from '../hooks';
import PlaylistCard from './PlaylistCard';

interface Props {
  onSelect: (playlistId: string) => void;
}

function PlaylistList({ onSelect }: Props) {
  const { playlists, isLoading, error } = usePlaylists();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12 text-white/60">
        <Loader2 className="w-5 h-5 animate-spin mr-2" />
        Loading playlists…
      </div>
    );
  }

  if (error) {
    return (
      <div className="py-8 text-center text-rose-300">
        Couldn't load playlists: {error}
      </div>
    );
  }

  if (playlists.length === 0) {
    return (
      <div className="py-12 text-center text-white/60">
        <p className="font-semibold">No playlists yet.</p>
        <p className="text-sm mt-2 text-white/40">
          Run <code className="px-1.5 py-0.5 rounded bg-white/10 text-white/80">
            python -m scripts.seed_playlist_from_game --game "[01] DDR 1st Mix"
          </code> in the backend to create one.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {playlists.map((p) => (
        <PlaylistCard key={p.id} playlist={p} onClick={() => onSelect(p.id)} />
      ))}
    </div>
  );
}

export default PlaylistList;

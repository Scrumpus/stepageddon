import { ListMusic } from 'lucide-react';
import { PlaylistDTO } from '../types';

interface Props {
  playlist: PlaylistDTO;
  onClick: () => void;
}

function PlaylistCard({ playlist, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left p-4 bg-white/5 hover:bg-white/[0.08] border border-white/10 hover:border-game-primary/40 rounded-xl transition-colors"
    >
      <div className="flex items-start gap-3">
        <div className="w-12 h-12 rounded-lg bg-game-primary/15 text-game-primary flex items-center justify-center shrink-0">
          <ListMusic className="w-6 h-6" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-white truncate">{playlist.name}</div>
          <div className="text-[11px] text-white/40 mt-1">
            {playlist.song_count} {playlist.song_count === 1 ? 'song' : 'songs'}
          </div>
        </div>
      </div>
    </button>
  );
}

export default PlaylistCard;

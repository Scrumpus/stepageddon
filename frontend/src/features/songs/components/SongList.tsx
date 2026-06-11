import { Loader2, Search, X } from 'lucide-react';
import { SongCard } from '@/features/playlists/components';
import { SongSummaryDTO } from '@/features/playlists/types';
import { useSongSearch } from '../hooks';

interface Props {
  onSelect: (song: SongSummaryDTO) => void;
}

function SongList({ onSelect }: Props) {
  const {
    query,
    setQuery,
    songs,
    isLoading,
    isLoadingMore,
    error,
    hasMore,
    loadMore,
  } = useSongSearch();

  return (
    <div className="space-y-3">
      <div className="relative">
        <Search className="w-4 h-4 text-white/40 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search songs by title or artist…"
          className="w-full pl-9 pr-9 py-2.5 bg-white/5 border border-white/10 focus:border-game-primary/50 focus:bg-white/[0.08] rounded-lg text-white placeholder:text-white/40 outline-none transition-colors"
        />
        {query && (
          <button
            onClick={() => setQuery('')}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded text-white/50 hover:text-white hover:bg-white/10"
            aria-label="Clear search"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12 text-white/60">
          <Loader2 className="w-5 h-5 animate-spin mr-2" />
          Loading songs…
        </div>
      )}

      {!isLoading && error && (
        <div className="py-8 text-center text-rose-300">
          Couldn't load songs: {error}
        </div>
      )}

      {!isLoading && !error && songs.length === 0 && (
        <div className="py-12 text-center text-white/60">
          {query ? (
            <>No songs match "{query}".</>
          ) : (
            <>No songs in the catalog yet.</>
          )}
        </div>
      )}

      {!isLoading && !error && songs.length > 0 && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
            {songs.map((song) => (
              <SongCard key={song.id} song={song} onSelect={onSelect} />
            ))}
          </div>

          {hasMore && (
            <div className="pt-2">
              <button
                onClick={loadMore}
                disabled={isLoadingMore}
                className="w-full py-2.5 rounded-lg bg-white/5 hover:bg-white/10 text-white/80 text-sm font-medium border border-white/10 transition-colors disabled:opacity-50"
              >
                {isLoadingMore ? (
                  <span className="inline-flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" /> Loading…
                  </span>
                ) : (
                  'Load more'
                )}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default SongList;

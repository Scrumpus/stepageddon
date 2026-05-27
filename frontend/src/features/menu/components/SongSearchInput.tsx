/**
 * Search-as-you-type widget backed by Audius + Jamendo (merged).
 *
 * Clicking a result calls `onSelect(url)` with the provider's track URL — the
 * generation pipeline re-resolves and downloads it.
 */

import { useEffect, useRef, useState } from 'react';
import { Search, Loader2, Music, X } from 'lucide-react';
import { useSongSearch } from '../hooks';
import type { SongSearchResult } from '../types/menu.types';

interface SongSearchInputProps {
  onSelect: (url: string) => void;
}

function formatDuration(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return '';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function SongSearchInput({ onSelect }: SongSearchInputProps) {
  const { query, setQuery, results, isLoading, error } = useSongSearch();
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Open the dropdown whenever the user types something
  useEffect(() => {
    if (query.trim().length > 0) setIsOpen(true);
  }, [query]);

  // Click-outside to close
  useEffect(() => {
    const onMouseDown = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', onMouseDown);
    return () => document.removeEventListener('mousedown', onMouseDown);
  }, []);

  const handlePick = (result: SongSearchResult) => {
    setIsOpen(false);
    onSelect(result.url);
  };

  const showDropdown = isOpen && query.trim().length > 0;
  const showEmpty =
    showDropdown && !isLoading && !error && results.length === 0;

  return (
    <div className="relative" ref={containerRef}>
      <div className="flex items-center gap-2 mb-3">
        <Search className="w-5 h-5 text-game-primary" />
        <span className="text-sm font-medium text-gray-300">Search a song</span>
      </div>

      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => query.trim().length > 0 && setIsOpen(true)}
          placeholder="Type a song or artist..."
          className="w-full pl-4 pr-10 py-3 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-game-primary text-white text-sm"
        />
        {isLoading ? (
          <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 animate-spin" />
        ) : query ? (
          <button
            type="button"
            onClick={() => {
              setQuery('');
              setIsOpen(false);
            }}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
            aria-label="Clear search"
          >
            <X className="w-4 h-4" />
          </button>
        ) : null}
      </div>

      {showDropdown && (
        <div className="absolute z-20 mt-2 left-0 right-0 bg-game-bg/95 backdrop-blur-md border border-white/10 rounded-lg shadow-2xl overflow-hidden">
          {error && (
            <div className="px-4 py-3 text-sm text-red-400">
              {error} — paste a link instead below.
            </div>
          )}

          {showEmpty && (
            <div className="px-4 py-3 text-sm text-gray-400">No matches.</div>
          )}

          {!error && results.length > 0 && (
            <ul className="max-h-80 overflow-y-auto divide-y divide-white/5">
              {results.map((r) => (
                <li key={`${r.source}:${r.id}`}>
                  <button
                    type="button"
                    onClick={() => handlePick(r)}
                    className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-white/5 transition-colors"
                  >
                    {r.thumbnail ? (
                      <img
                        src={r.thumbnail}
                        alt=""
                        className="w-12 h-12 rounded object-cover flex-shrink-0"
                      />
                    ) : (
                      <div className="w-12 h-12 rounded bg-white/10 flex items-center justify-center flex-shrink-0">
                        <Music className="w-5 h-5 text-gray-400" />
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-white truncate">
                        {r.title}
                      </div>
                      <div className="text-xs text-gray-400 truncate flex items-center gap-1.5">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide flex-shrink-0 ${
                            r.source === 'audius'
                              ? 'bg-purple-500/20 text-purple-300'
                              : 'bg-orange-500/20 text-orange-300'
                          }`}
                        >
                          {r.source}
                        </span>
                        <span className="truncate">
                          {r.artist}
                          {r.album ? ` • ${r.album}` : ''}
                        </span>
                      </div>
                    </div>
                    <div className="text-xs font-mono text-gray-500 flex-shrink-0">
                      {formatDuration(r.duration_seconds)}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

export default SongSearchInput;

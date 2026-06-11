/**
 * Discover tab — browse Audius + Jamendo with no search query.
 *
 * Feed toggle (Trending / Hidden gems) + genre chips drive a grid of clickable
 * track cards. Clicking a card (or "Surprise me") calls `onSelect(url)`, which
 * runs the same generation pipeline as search/URL paste.
 */

import { Flame, Gem, Loader2, Music, Sparkles } from 'lucide-react';
import { useDiscover } from '../hooks';
import type { SongSearchResult } from '@/types/common.types';

interface DiscoverTabProps {
  onSelect: (url: string) => void;
}

function formatDuration(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return '';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function DiscoverTab({ onSelect }: DiscoverTabProps) {
  const {
    feed,
    setFeed,
    genre,
    setGenre,
    genres,
    results,
    isLoading,
    error,
  } = useDiscover();

  const isUnderground = feed === 'underground';

  const handleSurprise = () => {
    if (results.length === 0) return;
    const pick = results[Math.floor(Math.random() * results.length)];
    onSelect(pick.url);
  };

  return (
    <div className="space-y-4">
      {/* Feed toggle + Surprise me */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex gap-1 bg-white/5 p-1 rounded-xl border border-white/10 w-fit">
          <FeedButton
            active={feed === 'trending'}
            onClick={() => setFeed('trending')}
            icon={<Flame className="w-4 h-4" />}
            label="Trending"
          />
          <FeedButton
            active={isUnderground}
            onClick={() => setFeed('underground')}
            icon={<Gem className="w-4 h-4" />}
            label="Hidden gems"
          />
        </div>
        <button
          type="button"
          onClick={handleSurprise}
          disabled={results.length === 0}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-game-primary text-game-bg hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition"
        >
          <Sparkles className="w-4 h-4" />
          Surprise me
        </button>
      </div>

      {/* Genre chips (disabled on the underground feed — Audius takes no genre) */}
      <div className="flex flex-wrap gap-2">
        <GenreChip
          label="All"
          active={genre === null}
          disabled={isUnderground}
          onClick={() => setGenre(null)}
        />
        {genres.map((g) => (
          <GenreChip
            key={g.key}
            label={g.label}
            active={genre === g.key}
            disabled={isUnderground}
            onClick={() => setGenre(g.key)}
          />
        ))}
      </div>
      {isUnderground && (
        <p className="text-xs text-gray-400 -mt-2">
          Hidden gems are surfaced across all genres.
        </p>
      )}

      {/* Results */}
      {isLoading ? (
        <div className="flex items-center justify-center py-16 text-gray-400">
          <Loader2 className="w-6 h-6 animate-spin" />
        </div>
      ) : error ? (
        <div className="py-10 text-center text-sm text-red-400">
          {error} — try another genre, or paste a link in the Create tab.
        </div>
      ) : results.length === 0 ? (
        <div className="py-10 text-center text-sm text-gray-400">
          Nothing here right now. Try a different genre.
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3 max-h-[65vh] overflow-y-auto pr-1">
          {results.map((r) => (
            <TrackCard key={`${r.source}:${r.id}`} track={r} onSelect={onSelect} />
          ))}
        </div>
      )}
    </div>
  );
}

interface TrackCardProps {
  track: SongSearchResult;
  onSelect: (url: string) => void;
}

function TrackCard({ track, onSelect }: TrackCardProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(track.url)}
      className="group text-left bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg overflow-hidden transition-colors"
    >
      <div className="aspect-square w-full bg-white/10 overflow-hidden">
        {track.thumbnail ? (
          <img
            src={track.thumbnail}
            alt=""
            className="w-full h-full object-cover group-hover:scale-105 transition-transform"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <Music className="w-8 h-8 text-gray-500" />
          </div>
        )}
      </div>
      <div className="p-2">
        <div className="text-sm font-medium text-white truncate">
          {track.title}
        </div>
        <div className="text-xs text-gray-400 truncate flex items-center gap-1.5 mt-0.5">
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide flex-shrink-0 ${
              track.source === 'audius'
                ? 'bg-purple-500/20 text-purple-300'
                : 'bg-orange-500/20 text-orange-300'
            }`}
          >
            {track.source}
          </span>
          <span className="truncate">{track.artist}</span>
        </div>
        {track.duration_seconds != null && (
          <div className="text-[10px] font-mono text-gray-500 mt-1">
            {formatDuration(track.duration_seconds)}
          </div>
        )}
      </div>
    </button>
  );
}

interface FeedButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}

function FeedButton({ active, onClick, icon, label }: FeedButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
        active
          ? 'bg-game-primary text-game-bg'
          : 'text-white/70 hover:text-white hover:bg-white/5'
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

interface GenreChipProps {
  label: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
}

function GenreChip({ label, active, disabled, onClick }: GenreChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
        active
          ? 'bg-game-primary text-game-bg border-game-primary'
          : 'bg-white/5 text-white/70 border-white/10 hover:text-white hover:bg-white/10'
      } disabled:opacity-30 disabled:cursor-not-allowed`}
    >
      {label}
    </button>
  );
}

export default DiscoverTab;

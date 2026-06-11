/**
 * Chart (difficulty) select panel for library songs.
 *
 * Mirrors the Create/Discover DifficultySelectScreen, but the charts already
 * exist in the catalog — there's no generation step. Picking a difficulty
 * hands the chart straight to the play pipeline.
 */

import { ArrowLeft, Loader2, Music } from 'lucide-react';
import { ChartSummaryDTO, SongSummaryDTO } from '../types';

const DIFFICULTY_COLORS: Record<string, string> = {
  beginner: 'text-emerald-400',
  easy: 'text-sky-400',
  medium: 'text-amber-400',
  hard: 'text-orange-400',
  challenge: 'text-rose-400',
  edit: 'text-fuchsia-400',
};

interface Props {
  song: SongSummaryDTO;
  charts: ChartSummaryDTO[];
  onSelectChart: (song: SongSummaryDTO, chart: ChartSummaryDTO) => void;
  onBack: () => void;
  disabled?: boolean;
  isLoading?: boolean;
  error?: string | null;
}

function ChartSelect({
  song,
  charts,
  onSelectChart,
  onBack,
  disabled,
  isLoading,
  error,
}: Props) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="p-2 rounded-lg hover:bg-white/10 text-white/80 transition-colors"
          aria-label="Back"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold text-white truncate">
            {song.title}
          </h2>
          {song.artist && (
            <p className="text-sm text-white/60 truncate">{song.artist}</p>
          )}
        </div>
      </div>

      <p className="text-gray-400 text-sm">Pick a difficulty</p>

      {isLoading ? (
        <div className="flex items-center justify-center py-12 text-white/60">
          <Loader2 className="w-5 h-5 animate-spin mr-2" />
          Loading charts…
        </div>
      ) : error ? (
        <div className="py-8 text-center text-rose-300">{error}</div>
      ) : charts.length === 0 ? (
        <div className="py-12 text-center text-white/60">
          No charts for this song.
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {charts.map((chart) => {
            const color =
              DIFFICULTY_COLORS[chart.difficulty_name.toLowerCase()] ??
              'text-white/90';
            return (
              <button
                key={chart.id}
                onClick={() => onSelectChart(song, chart)}
                disabled={disabled}
                className="p-4 rounded-lg border-2 border-white/20 hover:border-game-accent hover:bg-game-accent/10 transition-all text-left disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:border-white/20 disabled:hover:bg-transparent"
              >
                <div
                  className={`flex items-center gap-2 font-bold mb-1 ${color}`}
                >
                  <Music className="w-4 h-4" />
                  {chart.difficulty_name}
                  <span className="text-white/40 font-semibold">
                    Lv {chart.difficulty_level}
                  </span>
                </div>
                <div className="text-sm text-gray-400">
                  {chart.step_count.toLocaleString()} steps
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default ChartSelect;

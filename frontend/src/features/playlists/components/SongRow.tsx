import { Music } from 'lucide-react';
import { getAudioUrl } from '@/lib/axios';
import { ChartSummaryDTO, SongSummaryDTO } from '../types';
import DifficultyChip from './DifficultyChip';

interface Props {
  song: SongSummaryDTO;
  charts: ChartSummaryDTO[];
  onSelectChart: (song: SongSummaryDTO, chart: ChartSummaryDTO) => void;
  disabled?: boolean;
}

function SongRow({ song, charts, onSelectChart, disabled }: Props) {
  const thumb = song.jacket_url || song.banner_url;
  return (
    <div className="flex items-center gap-4 p-3 bg-white/5 hover:bg-white/[0.07] rounded-xl border border-white/5 transition-colors">
      <div className="w-16 h-16 rounded-lg overflow-hidden bg-white/10 flex items-center justify-center shrink-0">
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

      <div className="flex flex-wrap gap-1.5 justify-end">
        {charts.length === 0 ? (
          <span className="text-xs text-white/40 italic">No charts</span>
        ) : (
          charts.map((c) => (
            <DifficultyChip
              key={c.id}
              chart={c}
              onClick={() => onSelectChart(song, c)}
              disabled={disabled}
            />
          ))
        )}
      </div>
    </div>
  );
}

export default SongRow;

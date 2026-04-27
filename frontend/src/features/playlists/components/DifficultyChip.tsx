import { ChartSummaryDTO } from '../types';

const DIFFICULTY_STYLES: Record<string, string> = {
  beginner: 'bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-200 border-emerald-400/30',
  easy: 'bg-sky-500/20 hover:bg-sky-500/30 text-sky-200 border-sky-400/30',
  medium: 'bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 border-amber-400/30',
  hard: 'bg-orange-500/20 hover:bg-orange-500/30 text-orange-200 border-orange-400/30',
  challenge: 'bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border-rose-400/30',
  edit: 'bg-fuchsia-500/20 hover:bg-fuchsia-500/30 text-fuchsia-200 border-fuchsia-400/30',
};

const DEFAULT_STYLE =
  'bg-white/10 hover:bg-white/20 text-white/90 border-white/20';

interface Props {
  chart: ChartSummaryDTO;
  onClick: () => void;
  disabled?: boolean;
}

function DifficultyChip({ chart, onClick, disabled }: Props) {
  const key = chart.difficulty_name.toLowerCase();
  const style = DIFFICULTY_STYLES[key] ?? DEFAULT_STYLE;
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`flex flex-col items-center justify-center min-w-[72px] px-3 py-2 rounded-lg border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${style}`}
      title={`${chart.difficulty_name} · ${chart.step_count} steps`}
    >
      <span className="text-xs uppercase tracking-wider opacity-80">
        {chart.difficulty_name}
      </span>
      <span className="text-lg font-bold leading-tight">
        {chart.difficulty_level}
      </span>
      <span className="text-[10px] opacity-60">{chart.step_count} steps</span>
    </button>
  );
}

export default DifficultyChip;

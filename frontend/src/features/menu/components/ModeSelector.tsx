/**
 * Single / Dual mode toggle. Used both on the main menu (above the tabs)
 * and inside the in-game pause overlay.
 */

import { GameMode } from '@/types/common.types';

interface ModeSelectorProps {
  mode: GameMode;
  onModeChange: (mode: GameMode) => void;
  // Compact variant tightens spacing for the in-game pause panel.
  compact?: boolean;
}

const MODES: { value: GameMode; label: string }[] = [
  { value: 'single', label: 'Single' },
  { value: 'dual', label: 'Dual' },
];

function ModeSelector({ mode, onModeChange, compact = false }: ModeSelectorProps) {
  const padding = compact ? 'px-3 py-1.5 text-sm' : 'px-4 py-2 text-sm';
  return (
    <div className="flex gap-1 bg-white/5 p-1 rounded-xl border border-white/10 w-fit">
      {MODES.map(({ value, label }) => {
        const active = mode === value;
        return (
          <button
            key={value}
            onClick={() => onModeChange(value)}
            className={`${padding} rounded-lg font-medium transition-colors ${
              active
                ? 'bg-game-primary text-game-bg'
                : 'text-white/70 hover:text-white hover:bg-white/5'
            }`}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

export default ModeSelector;

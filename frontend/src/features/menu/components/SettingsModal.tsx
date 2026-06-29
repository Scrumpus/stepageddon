/**
 * SettingsModal — gear icon top-right that opens a centered overlay with
 * game mode and arrow-speed controls. Click outside or press Escape to close.
 */

import { useState, useEffect, useCallback } from 'react';
import { Settings, X } from 'lucide-react';
import { useGameStore } from '@/app/store/useGameStore';
import { ArrowSpeedMultiplier } from '@/app/store/slices/preferencesSlice';
import ModeSelector from './ModeSelector';

const SPEED_OPTIONS: ArrowSpeedMultiplier[] = [
  0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,
];

function formatSpeed(m: ArrowSpeedMultiplier): string {
  return m === 1 ? '1×' : `${m}×`;
}

function SettingsModal() {
  const gameMode = useGameStore((s) => s.gameMode);
  const setGameMode = useGameStore((s) => s.setGameMode);
  const speedMult = useGameStore((s) => s.arrowSpeedMultiplier);
  const setSpeedMult = useGameStore((s) => s.setArrowSpeedMultiplier);
  const [open, setOpen] = useState(false);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    },
    [],
  );

  useEffect(() => {
    if (open) document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [open, handleKeyDown]);

  return (
    <>
      {/* Gear button — fixed top-right */}
      <button
        onClick={() => setOpen(true)}
        className="fixed top-4 right-4 z-50 p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-white/70 hover:text-white transition-colors"
        aria-label="Settings"
      >
        <Settings className="w-5 h-5" />
      </button>

      {/* Modal overlay */}
      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          onClick={(e) => { if (e.target === e.currentTarget) setOpen(false); }}
        >
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
          <div className="relative bg-white/5 backdrop-blur-lg rounded-2xl p-6 shadow-2xl border border-white/10 w-full max-w-lg">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold">Settings</h2>
              <button
                onClick={() => setOpen(false)}
                className="p-1 rounded-lg hover:bg-white/10 text-white/60 hover:text-white transition-colors"
                aria-label="Close"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Game Mode */}
            <div className="flex flex-col items-center gap-2 mb-6">
              <p className="text-sm text-white/60">Game Mode</p>
              <ModeSelector mode={gameMode} onModeChange={setGameMode} />
              <p className="text-xs text-gray-500 mt-1">
                Changing mode during a game resets combos.
              </p>
            </div>

            {/* Arrow Speed */}
            <div className="flex flex-col items-center gap-2">
              <p className="text-sm text-white/60">Arrow Speed</p>
              <div className="flex flex-wrap justify-center gap-1">
                {SPEED_OPTIONS.map((m) => (
                  <button
                    key={m}
                    onClick={() => setSpeedMult(m)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      speedMult === m
                        ? 'bg-game-primary text-game-bg'
                        : 'text-white/70 hover:text-white hover:bg-white/5'
                    }`}
                  >
                    {formatSpeed(m)}
                  </button>
                ))}
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Applies on the next play.
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default SettingsModal;

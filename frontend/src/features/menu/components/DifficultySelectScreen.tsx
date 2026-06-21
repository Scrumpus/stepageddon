/**
 * Difficulty select screen — shown after generation finishes, before READY.
 * The user picks which of the generated charts to play.
 */

import { useState } from 'react';
import { ArrowLeft, Download, Loader2, Music } from 'lucide-react';
import { DifficultyLevel } from '@/types/common.types';
import { useGameStore } from '@/app/store/useGameStore';
import { exportGeneratedCharts } from '@/lib/exportSimfile';
import { DIFFICULTY_INFO } from '../types/menu.types';

const DIFFICULTY_ORDER: DifficultyLevel[] = [
  'beginner',
  'easy',
  'medium',
  'hard',
  'challenge',
];

function DifficultySelectScreen() {
  const songData = useGameStore((s) => s.songData);
  const stepsByDifficulty = useGameStore((s) => s.stepsByDifficulty);
  const audioUrl = useGameStore((s) => s.audioUrl);
  const difficultyPicked = useGameStore((s) => s.difficultyPicked);
  const resetGame = useGameStore((s) => s.resetGame);

  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  if (!stepsByDifficulty) {
    return null;
  }

  const handlePick = (level: DifficultyLevel) => {
    difficultyPicked(level);
  };

  const handleExport = async () => {
    if (!songData || !audioUrl || !stepsByDifficulty) return;
    setExporting(true);
    setExportError(null);
    try {
      await exportGeneratedCharts({ songInfo: songData, audioUrl, stepsByDifficulty });
    } catch {
      setExportError('Export failed. Please try again.');
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 sm:p-8 shadow-2xl border border-white/10">
          <div className="mb-6 text-center">
            <h1 className="text-3xl font-bold mb-2">Pick a difficulty</h1>
            {songData && (
              <p className="text-gray-400">
                {songData.title}
                {songData.artist ? ` — ${songData.artist}` : ''}
              </p>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
            {DIFFICULTY_ORDER.map((level) => {
              const steps = stepsByDifficulty[level];
              const info = DIFFICULTY_INFO[level];
              const disabled = !steps || steps.length === 0;
              const stepCount = steps?.length ?? 0;
              return (
                <button
                  key={level}
                  onClick={() => handlePick(level)}
                  disabled={disabled}
                  className={`p-4 rounded-lg border-2 transition-all text-left ${
                    disabled
                      ? 'border-white/10 opacity-40 cursor-not-allowed'
                      : 'border-white/20 hover:border-game-accent hover:bg-game-accent/10'
                  }`}
                >
                  <div className={`flex items-center gap-2 font-bold mb-1 ${info.color}`}>
                    <Music className="w-4 h-4" />
                    {info.name}
                  </div>
                  <div className="text-sm text-gray-400">
                    {stepCount.toLocaleString()} steps
                  </div>
                </button>
              );
            })}
          </div>

          <button
            onClick={handleExport}
            disabled={exporting || !audioUrl}
            className="w-full py-3 mb-3 bg-white/10 rounded-lg font-semibold hover:bg-white/20 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Download as a StepMania song folder (.zip)"
          >
            {exporting ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Download className="w-5 h-5" />
            )}
            Export to StepMania
          </button>
          {exportError && (
            <p className="text-rose-300 text-sm text-center mb-3">{exportError}</p>
          )}

          <button
            onClick={resetGame}
            className="w-full py-3 bg-white/10 rounded-lg font-semibold hover:bg-white/20 transition-all flex items-center justify-center gap-2"
          >
            <ArrowLeft className="w-5 h-5" />
            Back
          </button>
        </div>
      </div>
    </div>
  );
}

export default DifficultySelectScreen;

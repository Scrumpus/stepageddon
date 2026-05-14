/**
 * Difficulty select screen — shown after generation finishes, before READY.
 * The user picks which of the generated charts to play.
 */

import { ArrowLeft, Music } from 'lucide-react';
import { DifficultyLevel, GameState } from '@/types/common.types';
import { useApp } from '@/app/providers/AppProvider';
import { DIFFICULTY_INFO } from '../types/menu.types';

const DIFFICULTY_ORDER: DifficultyLevel[] = [
  'beginner',
  'easy',
  'medium',
  'hard',
  'challenge',
];

function DifficultySelectScreen() {
  const {
    songData,
    stepsByDifficulty,
    setSteps,
    setDifficulty,
    setGameState,
    resetGame,
  } = useApp();

  if (!stepsByDifficulty) {
    return null;
  }

  const handlePick = (level: DifficultyLevel) => {
    const steps = stepsByDifficulty[level];
    if (!steps) return;
    setDifficulty(level);
    setSteps(steps);
    setGameState(GameState.READY);
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

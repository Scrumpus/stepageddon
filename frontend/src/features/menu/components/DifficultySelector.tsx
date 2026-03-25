/**
 * Difficulty selection component
 */

import { Music } from 'lucide-react';
import { DifficultyLevel } from '@/types/common.types';
import { DIFFICULTY_INFO } from '@/features/menu/types/menu.types';

interface DifficultySelectorProps {
  difficulty: DifficultyLevel;
  onDifficultyChange: (difficulty: DifficultyLevel) => void;
}

function DifficultySelector({ difficulty, onDifficultyChange }: DifficultySelectorProps) {
  return (
    <div className="mb-8">
      <div className="grid grid-cols-3 gap-3">
        {Object.entries(DIFFICULTY_INFO).map(([key, info]) => (
          <button
            key={key}
            onClick={() => onDifficultyChange(key as DifficultyLevel)}
            className={`p-4 rounded-lg border-2 transition-all ${
              difficulty === key
                ? 'border-game-accent bg-game-accent/20 scale-105'
                : 'border-white/20 hover:border-white/40'
            }`}
          >
            <div className={`font-bold mb-1 ${info.color}`}>
              {info.name}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

export default DifficultySelector;

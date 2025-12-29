/**
 * Judgment display - shows hit feedback (PERFECT, GOOD, OK, MISS)
 */

import { Judgment, JudgmentDisplay as JudgmentDisplayType } from '../types/game.types';

interface JudgmentDisplayProps {
  judgment: JudgmentDisplayType | null;
}

const JUDGMENT_COLORS = {
  [Judgment.PERFECT]: 'text-yellow-400',
  [Judgment.GOOD]: 'text-green-400',
  [Judgment.OK]: 'text-blue-400',
  [Judgment.MISS]: 'text-red-400',
};

function JudgmentDisplay({ judgment }: JudgmentDisplayProps) {
  if (!judgment) return null;

  return (
    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 pointer-events-none">
      <div className="text-center animate-pulse-hit">
        <div className={`text-4xl font-bold ${JUDGMENT_COLORS[judgment.judgment]}`}>
          {judgment.judgment}
        </div>
        {judgment.points > 0 && (
          <div className="text-2xl text-white">+{judgment.points}</div>
        )}
      </div>
    </div>
  );
}

export default JudgmentDisplay;

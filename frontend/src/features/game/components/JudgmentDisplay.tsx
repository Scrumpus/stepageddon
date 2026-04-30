/**
 * Judgment display - shows hit feedback (PERFECT, GOOD, OK, MISS) plus combo.
 * Combo is always rendered; the judgment block only renders when a hit landed
 * recently (cleared on a 500ms timer in GameScreen).
 */

import { Judgment, JudgmentDisplay as JudgmentDisplayType } from '../types/game.types';

interface JudgmentDisplayProps {
  judgment: JudgmentDisplayType | null;
  combo: number;
  // Horizontal center as a percent of the playfield width. Match the
  // corresponding ArrowLane's centerPercent so judgment+combo sit over the
  // player's lane.
  centerPercent?: number;
}

const JUDGMENT_COLORS = {
  [Judgment.PERFECT]: 'text-yellow-400',
  [Judgment.GOOD]: 'text-green-400',
  [Judgment.OK]: 'text-blue-400',
  [Judgment.MISS]: 'text-red-400',
};

function JudgmentDisplay({ judgment, combo, centerPercent = 50 }: JudgmentDisplayProps) {
  return (
    <div
      className="absolute pointer-events-none text-center"
      style={{
        top: '40%',
        left: `${centerPercent}%`,
        transform: 'translate(-50%, -50%)',
      }}
    >
      {judgment && (
        <div className="animate-pulse-hit">
          <div className={`text-4xl font-bold ${JUDGMENT_COLORS[judgment.judgment]}`}>
            {judgment.judgment}
          </div>
          {judgment.points > 0 && (
            <div className="text-2xl text-white">+{judgment.points}</div>
          )}
        </div>
      )}
      <div className="mt-3 text-3xl font-bold text-game-accent">
        {combo}x
      </div>
    </div>
  );
}

export default JudgmentDisplay;

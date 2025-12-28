/**
 * Arrow lane - renders target zone and active arrows
 */

import { ArrowLeft, ArrowDown, ArrowUp, ArrowRight } from 'lucide-react';
import { Direction } from '@/types/common.types';
import { ActiveArrow } from '../../types/step.types';
import { VISUAL_CONFIG, DIRECTIONS } from '../../types/game.types';

interface ArrowLaneProps {
  activeArrows: ActiveArrow[];
  activeKeys: Record<Direction, boolean>;
}

const ARROW_ICONS = {
  [Direction.LEFT]: ArrowLeft,
  [Direction.DOWN]: ArrowDown,
  [Direction.UP]: ArrowUp,
  [Direction.RIGHT]: ArrowRight,
};

export function ArrowLane({ activeArrows, activeKeys }: ArrowLaneProps) {
  return (
    <div className="flex-1 relative overflow-hidden">
      {/* Target Zone */}
      <div
        className="absolute left-1/2 transform -translate-x-1/2 flex gap-4 z-10"
        style={{ top: `${VISUAL_CONFIG.HIT_ZONE_Y}px` }}
      >
        {DIRECTIONS.map((direction) => {
          const Icon = ARROW_ICONS[direction];
          return (
            <div
              key={direction}
              className={`arrow-target transition-all ${
                activeKeys[direction] ? 'arrow-active scale-110' : ''
              }`}
            >
              <Icon className="w-10 h-10" />
            </div>
          );
        })}
      </div>

      {/* Falling Arrows */}
      {activeArrows.map((arrow, idx) => {
        const Icon = ARROW_ICONS[arrow.direction];
        const directionIndex = DIRECTIONS.indexOf(arrow.direction);
        const x = -160 + directionIndex * 96;

        return (
          <div
            key={`${arrow.index}-${idx}`}
            className="absolute left-1/2 transform -translate-x-1/2 transition-opacity"
            style={{
              top: `${arrow.y}px`,
              left: `calc(50% + ${x}px)`,
              opacity: arrow.index !== undefined && arrow.hit ? 0 : 1,
            }}
          >
            <div className="w-20 h-20 bg-game-primary/80 rounded-lg flex items-center justify-center border-2 border-game-primary">
              <Icon className="w-10 h-10" />
            </div>
          </div>
        );
      })}
    </div>
  );
}

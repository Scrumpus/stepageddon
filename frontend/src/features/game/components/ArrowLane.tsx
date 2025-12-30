/**
 * Arrow lane - renders target zone and active arrows (DDR style)
 * Supports hold notes with trails
 */

import { Direction } from '@/types/common.types';
import { ActiveArrow, ActiveHold } from '../types/step.types';
import { VISUAL_CONFIG, DIRECTIONS } from '../types/game.types';

interface ArrowLaneProps {
  activeArrows: ActiveArrow[];
  activeKeys: Record<Direction, boolean>;
  activeHolds: ActiveHold[];
  arrowSpeed: number;  // Pixels per second for calculating hold trail length
}

// DDR-style color scheme
const ARROW_COLORS: Record<Direction, { main: string; glow: string; bg: string }> = {
  [Direction.LEFT]: { main: '#ff6bf3', glow: '#ff6bf3', bg: 'rgba(255, 107, 243, 0.15)' },
  [Direction.DOWN]: { main: '#00d4ff', glow: '#00d4ff', bg: 'rgba(0, 212, 255, 0.15)' },
  [Direction.UP]: { main: '#00ff87', glow: '#00ff87', bg: 'rgba(0, 255, 135, 0.15)' },
  [Direction.RIGHT]: { main: '#ff4757', glow: '#ff4757', bg: 'rgba(255, 71, 87, 0.15)' },
};

// Arrow rotation for each direction
const ARROW_ROTATION: Record<Direction, number> = {
  [Direction.LEFT]: -90,
  [Direction.DOWN]: 180,
  [Direction.UP]: 0,
  [Direction.RIGHT]: 90,
};

// DDR-style arrow SVG
function ArrowSVG({ color, size = 64 }: { color: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      {/* Arrow shape */}
      <path
        d="M32 8L52 28H40V56H24V28H12L32 8Z"
        fill={color}
        stroke={color}
        strokeWidth="2"
        strokeLinejoin="round"
      />
      {/* Inner highlight */}
      <path
        d="M32 14L46 28H40V50H24V28H18L32 14Z"
        fill="rgba(255,255,255,0.3)"
      />
    </svg>
  );
}

// Target receptor (outline only)
function ReceptorSVG({ color, active, size = 64 }: { color: string; active: boolean; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      <path
        d="M32 8L52 28H40V56H24V28H12L32 8Z"
        fill={active ? `${color}33` : 'transparent'}
        stroke={color}
        strokeWidth="3"
        strokeLinejoin="round"
        opacity={active ? 1 : 0.5}
      />
    </svg>
  );
}

// Hold trail component
function HoldTrail({
  color,
  length,
  width,
  glowColor,
  progress = 0,
  isActive = false,
}: {
  color: string;
  length: number;
  width: number;
  glowColor: string;
  progress?: number;  // 0-1 indicating how much of the hold has been completed
  isActive?: boolean; // Whether the hold is currently being held
}) {
  const trailWidth = width * 0.35;  // Trail is narrower than arrow
  // Shrink trail from bottom as hold progresses
  const remainingLength = length * (1 - progress);
  // Active holds glow brighter
  const activeGlow = isActive ? 1.5 : 1;

  return (
    <div
      className="absolute transition-all duration-75"
      style={{
        width: trailWidth,
        height: Math.max(remainingLength, 0),
        left: (width - trailWidth) / 2,
        top: width * 0.6,  // Start from bottom of arrow head
        background: `linear-gradient(to bottom, ${color}, ${color}88 20%, ${color}66 80%, ${color}33)`,
        borderRadius: trailWidth / 2,
        boxShadow: `0 0 ${10 * activeGlow}px ${glowColor}66, inset 0 0 8px rgba(255,255,255,0.3)`,
        opacity: isActive ? 1 : 0.7,
      }}
    >
      {/* Inner glow line */}
      <div
        className="absolute left-1/2 -translate-x-1/2 h-full"
        style={{
          width: trailWidth * 0.4,
          background: `linear-gradient(to bottom, rgba(255,255,255,0.6), rgba(255,255,255,0.2) 50%, transparent)`,
          borderRadius: trailWidth / 2,
        }}
      />
      {/* End cap - only show if trail has length */}
      {remainingLength > trailWidth && (
        <div
          className="absolute bottom-0 left-1/2 -translate-x-1/2"
          style={{
            width: trailWidth * 1.5,
            height: trailWidth * 1.5,
            background: color,
            borderRadius: '50%',
            boxShadow: `0 0 ${15 * activeGlow}px ${glowColor}`,
          }}
        />
      )}
    </div>
  );
}

function ArrowLane({ activeArrows, activeKeys, activeHolds, arrowSpeed }: ArrowLaneProps) {
  const arrowSize = VISUAL_CONFIG.ARROW_SIZE;
  const gap = 16;
  const totalWidth = (arrowSize * 4) + (gap * 3);
  const startX = -totalWidth / 2;

  return (
    <div className="flex-1 relative overflow-hidden bg-gradient-to-b from-black/50 to-transparent">
      {/* Lane guides */}
      <div
        className="absolute left-1/2 h-full pointer-events-none"
        style={{ transform: 'translateX(-50%)' }}
      >
        {DIRECTIONS.map((direction, index) => {
          const x = startX + index * (arrowSize + gap);
          const colors = ARROW_COLORS[direction];
          return (
            <div
              key={`lane-${direction}`}
              className="absolute h-full"
              style={{
                left: `calc(50% + ${x}px)`,
                width: arrowSize,
                background: `linear-gradient(to top, ${colors.bg}, transparent 30%)`,
              }}
            />
          );
        })}
      </div>

      {/* Target Zone (Receptors) */}
      <div
        className="absolute left-1/2 transform -translate-x-1/2 flex z-20"
        style={{
          top: `${VISUAL_CONFIG.HIT_ZONE_Y}px`,
          gap: `${gap}px`,
        }}
      >
        {DIRECTIONS.map((direction) => {
          const colors = ARROW_COLORS[direction];
          const rotation = ARROW_ROTATION[direction];
          const isActive = activeKeys[direction];

          return (
            <div
              key={direction}
              className="transition-transform duration-75"
              style={{
                width: arrowSize,
                height: arrowSize,
                transform: `rotate(${rotation}deg) ${isActive ? 'scale(1.1)' : 'scale(1)'}`,
                filter: isActive ? `drop-shadow(0 0 20px ${colors.glow})` : 'none',
              }}
            >
              <ReceptorSVG color={colors.main} active={isActive} size={arrowSize} />
            </div>
          );
        })}
      </div>

      {/* Rising Arrows */}
      {activeArrows.map((arrow) => {
        const directionIndex = DIRECTIONS.indexOf(arrow.direction);
        const x = startX + directionIndex * (arrowSize + gap);
        const arrowKey = `${arrow.stepIndex}-${arrow.arrowIndex}`;
        const colors = ARROW_COLORS[arrow.direction];
        const rotation = ARROW_ROTATION[arrow.direction];

        // Glow intensity based on proximity to hit zone
        const distanceToHit = Math.abs(arrow.timeUntilHit);
        const glowIntensity = distanceToHit < 0.3 ? (0.3 - distanceToHit) / 0.3 : 0;

        // Calculate hold trail length if this is a hold note
        const isHold = arrow.type === 'hold' && arrow.hold_duration;
        const trailLength = isHold ? arrow.hold_duration! * arrowSpeed : 0;

        // Find if this hold is being actively held
        const activeHold = activeHolds.find((h) => h.arrowKey === arrowKey);
        const holdProgress = activeHold?.holdProgress ?? 0;
        const isActivelyHeld = !!activeHold;

        return (
          <div
            key={arrowKey}
            className="absolute"
            style={{
              top: `${arrow.y}px`,
              left: `calc(50% + ${x}px)`,
              width: arrowSize,
            }}
          >
            {/* Hold trail (not rotated - stays vertical) */}
            {isHold && (
              <HoldTrail
                color={colors.main}
                length={trailLength}
                width={arrowSize}
                glowColor={colors.glow}
                progress={holdProgress}
                isActive={isActivelyHeld}
              />
            )}
            {/* Arrow head (rotated) */}
            <div
              style={{
                width: arrowSize,
                height: arrowSize,
                transform: `rotate(${rotation}deg)`,
                filter: glowIntensity > 0
                  ? `drop-shadow(0 0 ${10 + glowIntensity * 20}px ${colors.glow})`
                  : 'none',
              }}
            >
              <ArrowSVG color={colors.main} size={arrowSize} />
            </div>
          </div>
        );
      })}

      {/* Hit zone line */}
      <div
        className="absolute left-0 right-0 h-0.5 bg-white/20 z-10"
        style={{ top: `${VISUAL_CONFIG.HIT_ZONE_Y + arrowSize / 2}px` }}
      />
    </div>
  );
}

export default ArrowLane;

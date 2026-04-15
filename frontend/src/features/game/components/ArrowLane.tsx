/**
 * Arrow lane - renders target zone and active arrows (DDR style)
 * Supports hold notes with trails
 */

import { Direction } from '@/types/common.types';
import { ActiveArrow, ActiveHold, BeatSubdivision } from '../types/step.types';
import { VISUAL_CONFIG, DIRECTIONS } from '../types/game.types';

interface ArrowLaneProps {
  activeArrows: ActiveArrow[];
  activeKeys: Record<Direction, boolean>;
  activeHolds: ActiveHold[];
  arrowSpeed: number;  // Pixels per second for calculating hold trail length
  tempo: number;       // BPM for beat-synced spritesheet animation
}

// DDR-style color scheme
const ARROW_COLORS: Record<Direction, { main: string; glow: string; bg: string }> = {
  [Direction.LEFT]: { main: '#ff6bf3', glow: '#ff6bf3', bg: 'rgba(255, 107, 243, 0.15)' },
  [Direction.DOWN]: { main: '#00d4ff', glow: '#00d4ff', bg: 'rgba(0, 212, 255, 0.15)' },
  [Direction.UP]: { main: '#00ff87', glow: '#00ff87', bg: 'rgba(0, 255, 135, 0.15)' },
  [Direction.RIGHT]: { main: '#ff4757', glow: '#ff4757', bg: 'rgba(255, 71, 87, 0.15)' },
};

// Arrow rotation for each direction (set to 0 when using pre-rotated PNGs)
const ARROW_ROTATION: Record<Direction, number> = {
  [Direction.LEFT]: 0,
  [Direction.DOWN]: 0,
  [Direction.UP]: 0,
  [Direction.RIGHT]: 0,
};

// Spritesheet config
// Each file is a horizontal strip of 4 frames (e.g., 512x128 for 128px frames)
// Frames cycle through inner section lighting: dim → 1 lit → 2 lit → all lit
//
// Drop into public/arrows/:
//   arrow-left-4th.png, arrow-left-8th.png, arrow-left-16th.png  (different colors)
//   arrow-down-4th.png, arrow-down-8th.png, arrow-down-16th.png
//   arrow-up-4th.png, arrow-up-8th.png, arrow-up-16th.png
//   arrow-right-4th.png, arrow-right-8th.png, arrow-right-16th.png
//   receptor-left.png, receptor-down.png, receptor-up.png, receptor-right.png
// CSS filters to recolor arrows by beat subdivision
// Base arrows are blue (~210° hue). hue-rotate shifts from there.
const SUBDIVISION_FILTERS: Record<BeatSubdivision, string> = {
  '4th': 'none',                                    // default blue
  '8th': 'hue-rotate(150deg) saturate(1.3)',         // red
  '16th': 'hue-rotate(70deg) saturate(1.3)',         // purple
};

const SPRITE_FRAMES = 4;

const ARROW_SHEETS: Record<BeatSubdivision, Record<Direction, string>> = {
  '4th': {
    [Direction.LEFT]: '/arrows/arrow-left-4th.png',
    [Direction.DOWN]: '/arrows/arrow-down-4th.png',
    [Direction.UP]: '/arrows/arrow-up-4th.png',
    [Direction.RIGHT]: '/arrows/arrow-right-4th.png',
  },
  '8th': {
    [Direction.LEFT]: '/arrows/arrow-left-8th.png',
    [Direction.DOWN]: '/arrows/arrow-down-8th.png',
    [Direction.UP]: '/arrows/arrow-up-8th.png',
    [Direction.RIGHT]: '/arrows/arrow-right-8th.png',
  },
  '16th': {
    [Direction.LEFT]: '/arrows/arrow-left-16th.png',
    [Direction.DOWN]: '/arrows/arrow-down-16th.png',
    [Direction.UP]: '/arrows/arrow-up-16th.png',
    [Direction.RIGHT]: '/arrows/arrow-right-16th.png',
  },
};

const RECEPTOR_SHEETS: Record<Direction, string> = {
  [Direction.LEFT]: '/arrows/receptor-left.png',
  [Direction.DOWN]: '/arrows/receptor-down.png',
  [Direction.UP]: '/arrows/receptor-up.png',
  [Direction.RIGHT]: '/arrows/receptor-right.png',
};

function ArrowImage({ direction, subdivision = '4th', size = 64, tempo = 120 }: {
  direction: Direction;
  subdivision?: BeatSubdivision;
  size?: number;
  tempo?: number;
}) {
  const beatDuration = 60 / tempo;
  const sheet = ARROW_SHEETS[subdivision][direction];
  const sheetWidth = size * SPRITE_FRAMES;

  return (
    <div
      style={{
        width: size,
        height: size,
        backgroundImage: `url(${sheet})`,
        backgroundSize: `${sheetWidth}px ${size}px`,
        '--sheet-width': `${-sheetWidth}px`,
        animation: `sprite-step ${beatDuration}s steps(${SPRITE_FRAMES}) infinite`,
        filter: SUBDIVISION_FILTERS[subdivision],
      } as React.CSSProperties}
    />
  );
}

function ReceptorImage({ direction, size = 64, tempo = 120 }: {
  direction: Direction;
  size?: number;
  tempo?: number;
}) {
  const beatDuration = 60 / tempo;

  return (
    <img
      src={RECEPTOR_SHEETS[direction]}
      width={size}
      height={size}
      alt=""
      draggable={false}
      style={{
        opacity: 0.5,
        animation: `receptor-pulse ${beatDuration}s steps(1) infinite`,
      }}
    />
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

function ArrowLane({ activeArrows, activeKeys, activeHolds, arrowSpeed, tempo }: ArrowLaneProps) {
  const arrowSize = VISUAL_CONFIG.ARROW_SIZE;
  const gap = 16;
  const totalWidth = (arrowSize * 4) + (gap * 3);
  const startX = -totalWidth / 2;

  return (
    <div className="flex-1 relative overflow-hidden bg-gradient-to-b from-black/50 to-transparent">

      {/* Target Zone (Receptors) */}
      <div
        className="absolute left-1/2 transform -translate-x-1/2 flex z-20"
        style={{
          top: `${VISUAL_CONFIG.HIT_ZONE_Y}px`,
          gap: `${gap}px`,
        }}
      >
        {DIRECTIONS.map((direction) => {
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
              }}
            >
              <ReceptorImage direction={direction} size={arrowSize} tempo={tempo} />
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
              }}
            >
              <ArrowImage direction={arrow.direction} subdivision={arrow.beat_subdivision} size={arrowSize} tempo={tempo} />
            </div>
          </div>
        );
      })}

    </div>
  );
}

export default ArrowLane;

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
// CSS filters to recolor arrows by beat subdivision
// Base arrows are blue (~210° hue). hue-rotate shifts from there.
const SUBDIVISION_FILTERS: Record<BeatSubdivision, string> = {
  '4th': 'none',                                    // default blue
  '8th': 'hue-rotate(150deg) saturate(1.3)',         // red
  '16th': 'hue-rotate(70deg) saturate(1.3)',         // purple
};

// Hold notes override subdivision color so the mechanic reads at a glance.
const HOLD_FILTER = 'hue-rotate(120deg) saturate(1.5) brightness(1.05)';   // pink

const SPRITE_FRAMES = 4;

const ARROW_SHEETS: Record<Direction, string> = {
  [Direction.LEFT]: '/arrows/arrow-left.png',
  [Direction.DOWN]: '/arrows/arrow-down.png',
  [Direction.UP]: '/arrows/arrow-up.png',
  [Direction.RIGHT]: '/arrows/arrow-right.png',
};

const RECEPTOR_SHEETS: Record<Direction, string> = {
  [Direction.LEFT]: '/arrows/receptor-left.png',
  [Direction.DOWN]: '/arrows/receptor-down.png',
  [Direction.UP]: '/arrows/receptor-up.png',
  [Direction.RIGHT]: '/arrows/receptor-right.png',
};

function ArrowImage({ direction, subdivision = '4th', isHold = false, size = 64, tempo = 120 }: {
  direction: Direction;
  subdivision?: BeatSubdivision;
  isHold?: boolean;
  size?: number;
  tempo?: number;
}) {
  const beatDuration = 60 / tempo;
  const sheet = ARROW_SHEETS[direction];
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
        filter: isHold ? HOLD_FILTER : SUBDIVISION_FILTERS[subdivision],
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

// Hold trail - classic DDR style: a vertical column of repeated direction arrows.
// Trail length is fixed; tiles fade individually as the hold's elapsed portion
// passes the receptor.
function HoldTrail({
  direction,
  subdivision = '4th',
  length,
  width,
  arrowSpeed,
  tempo,
  progress = 0,
  isActive = false,
}: {
  direction: Direction;
  subdivision?: BeatSubdivision;
  length: number;
  width: number;
  arrowSpeed: number;
  tempo: number;
  progress?: number;  // 0-1 indicating how much of the hold has been completed
  isActive?: boolean; // Whether the hold is currently being held
}) {
  if (length <= 0) return null;

  // Space trail tiles every 16th note so the column reads as a denser ribbon
  // rather than discrete arrow-sized blocks.
  const sixteenthSeconds = 60 / tempo / 4;
  const tileSpacing = Math.max(1, sixteenthSeconds * arrowSpeed);
  const tileCount = Math.max(1, Math.ceil(length / tileSpacing));
  const sheet = ARROW_SHEETS[direction];
  const sheetWidth = width * SPRITE_FRAMES;
  // Hold trails always render in pink so the mechanic stays distinct from the
  // subdivision-based color of one-shot taps.
  const filter = HOLD_FILTER;
  // Subdivision is currently unused for trails; kept on the prop for parity
  // with the head and future tweaks.
  void subdivision;
  const baseOpacity = isActive ? 1 : 0.7;

  return (
    <div
      className="absolute"
      style={{
        width,
        // Anchor the trail to the head's top so the first tile sits beneath the
        // arrow head rather than after a head-sized gap (which read as a missing
        // first beat with dense 16th-note tile spacing).
        top: 0,
        left: 0,
        height: length + width,
      }}
    >
      {Array.from({ length: tileCount }, (_, i) => {
        // Each tile represents 1 / tileCount of progress; clamp to [0.2, 1] so
        // played-through tiles remain dimly visible.
        const consumption = progress * tileCount - i;
        const opacity = Math.max(0.2, Math.min(1, 1 - consumption)) * baseOpacity;

        return (
          <div
            key={i}
            className="absolute"
            style={{
              top: i * tileSpacing,
              left: 0,
              width,
              height: width,
              backgroundImage: `url(${sheet})`,
              // Static first frame: pin sheet to its left edge, no animation.
              backgroundPosition: '0 0',
              backgroundSize: `${sheetWidth}px ${width}px`,
              filter,
              opacity,
            }}
          />
        );
      })}
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
        const rotation = ARROW_ROTATION[arrow.direction];

        // Calculate hold trail length if this is a hold note
        const isHold = arrow.type === 'hold' && arrow.hold_duration;
        const trailLength = isHold ? arrow.hold_duration! * arrowSpeed : 0;
        const subdivision: BeatSubdivision = arrow.beat_subdivision ?? '4th';

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
                direction={arrow.direction}
                subdivision={subdivision}
                length={trailLength}
                width={arrowSize}
                arrowSpeed={arrowSpeed}
                tempo={tempo}
                progress={holdProgress}
                isActive={isActivelyHeld}
              />
            )}
            {/* Arrow head (rotated) - keep above the trail so it covers the
                tiles drawn behind it (trail starts at the head's top). */}
            <div
              className="relative"
              style={{
                width: arrowSize,
                height: arrowSize,
                transform: `rotate(${rotation}deg)`,
                zIndex: 1,
              }}
            >
              <ArrowImage direction={arrow.direction} subdivision={subdivision} isHold={!!isHold} size={arrowSize} tempo={tempo} />
            </div>
          </div>
        );
      })}

    </div>
  );
}

export default ArrowLane;

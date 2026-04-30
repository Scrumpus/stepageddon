/**
 * Game Screen - orchestrates game play
 * Composes hooks and subcomponents for clean separation of concerns
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { GameState } from '@/types/common.types';
import { useApp } from '@/app/providers/AppProvider';
import {
  Judgment,
  HitAccuracy,
  getArrowSpeed,
  IS_DUAL_MODE,
  KEY_MAP,
  KEY_MAP_P2,
} from '../types/game.types';
import { calculateAccuracy } from '../utils/scoring';
import {
  useGameLoop,
  useHitDetection,
  useKeyboardInput,
  useAudioPlayer,
  useHoldTracking,
  useMissTracking,
} from '../hooks';
import GameHUD from './GameHUD';
import ArrowLane from './ArrowLane';
import JudgmentDisplay from './JudgmentDisplay';
import PauseOverlay from './PauseOverlay';

const EMPTY_HIT_ACCURACY: HitAccuracy = { perfect: 0, good: 0, ok: 0, miss: 0 };

function GameScreen() {
  const { steps, audioRef, songData, setGameResults, resetGame } = useApp();

  // Local game state
  const [gameState, setGameState] = useState<GameState>(GameState.PLAYING);

  // P1 state
  const [score, setScore] = useState(0);
  const [combo, setCombo] = useState(0);
  const [_maxCombo, setMaxCombo] = useState(0);
  const [_hitAccuracy, setHitAccuracy] = useState<HitAccuracy>(EMPTY_HIT_ACCURACY);
  const [judgmentDisplay, setJudgmentDisplay] = useState<{
    judgment: Judgment;
    points: number;
  } | null>(null);
  const processedStepsRefP1 = useRef<Set<string>>(new Set());

  // P2 state — always created so the hook order stays stable; only wired
  // visually and to the keyboard when IS_DUAL_MODE.
  const [score2, setScore2] = useState(0);
  const [combo2, setCombo2] = useState(0);
  const [_maxCombo2, setMaxCombo2] = useState(0);
  const [_hitAccuracy2, setHitAccuracy2] = useState<HitAccuracy>(EMPTY_HIT_ACCURACY);
  const [judgmentDisplay2, setJudgmentDisplay2] = useState<{
    judgment: Judgment;
    points: number;
  } | null>(null);
  const processedStepsRefP2 = useRef<Set<string>>(new Set());

  // Show judgment feedback
  const showJudgment = useCallback((judgment: Judgment, points: number) => {
    setJudgmentDisplay({ judgment, points });
    setTimeout(() => setJudgmentDisplay(null), 500);
  }, []);
  const showJudgment2 = useCallback((judgment: Judgment, points: number) => {
    setJudgmentDisplay2({ judgment, points });
    setTimeout(() => setJudgmentDisplay2(null), 500);
  }, []);

  // Handle miss
  const handleMiss = useCallback(() => {
    setCombo(0);
    setHitAccuracy((prev) => ({ ...prev, miss: prev.miss + 1 }));
    showJudgment(Judgment.MISS, 0);
  }, [showJudgment]);
  const handleMiss2 = useCallback(() => {
    setCombo2(0);
    setHitAccuracy2((prev) => ({ ...prev, miss: prev.miss + 1 }));
    showJudgment2(Judgment.MISS, 0);
  }, [showJudgment2]);

  // Finish game — P1 results only for now (results screen is single-player).
  const finishGame = useCallback(() => {
    setHitAccuracy((currentAccuracy) => {
      setScore((currentScore) => {
        setMaxCombo((currentMaxCombo) => {
          const totalNotes = Object.values(currentAccuracy).reduce((sum, val) => sum + val, 0);
          const accuracy = calculateAccuracy(currentAccuracy);
          setGameResults({
            score: currentScore,
            maxCombo: currentMaxCombo,
            hitAccuracy: currentAccuracy,
            accuracy,
            totalNotes,
          });
          return currentMaxCombo;
        });
        return currentScore;
      });
      return currentAccuracy;
    });
  }, [setGameResults]);

  // Toggle pause
  const togglePause = useCallback(() => {
    setGameState((prev) => {
      if (prev === GameState.PLAYING) return GameState.PAUSED;
      if (prev === GameState.PAUSED) return GameState.PLAYING;
      return prev;
    });
  }, []);

  if (!songData) return null;

  const addScore = useCallback((points: number) => setScore((prev) => prev + points), []);
  const addScore2 = useCallback((points: number) => setScore2((prev) => prev + points), []);

  // Hold tracking — one per player. activeHolds for both feeds the loop's
  // visibility extension so a hold trail rendered in either lane stays on
  // screen through to release/end.
  const {
    activeHolds: activeHolds1,
    startHold: startHold1,
    releaseHold: releaseHold1,
    updateHolds: updateHolds1,
  } = useHoldTracking({ combo, onScoreUpdate: addScore });
  const {
    activeHolds: activeHolds2,
    startHold: startHold2,
    releaseHold: releaseHold2,
    updateHolds: updateHolds2,
  } = useHoldTracking({ combo: combo2, onScoreUpdate: addScore2 });

  const mergedHolds = useMemo(
    () => (IS_DUAL_MODE ? [...activeHolds1, ...activeHolds2] : activeHolds1),
    [activeHolds1, activeHolds2]
  );

  // Game loop hook — pure: emits time + visible arrows for both players.
  const { currentTime, activeArrows } = useGameLoop({
    audioRef,
    steps,
    gameState,
    songDuration: songData.duration,
    tempo: songData.tempo || 120,
    activeHolds: mergedHolds,
    onFinish: finishGame,
  });

  // Update holds each frame when playing
  useEffect(() => {
    if (gameState === GameState.PLAYING) {
      updateHolds1(currentTime);
      if (IS_DUAL_MODE) updateHolds2(currentTime);
    }
  }, [currentTime, gameState, updateHolds1, updateHolds2]);

  // Hit detection — P1
  const { checkHit: checkHit1 } = useHitDetection({
    activeArrows,
    processedStepsRef: processedStepsRefP1,
    combo,
    currentTime,
    onScoreUpdate: addScore,
    onComboUpdate: useCallback((newCombo: number) => {
      setCombo(newCombo);
      setMaxCombo((prev) => Math.max(prev, newCombo));
    }, []),
    onHitAccuracyUpdate: useCallback((judgment: Judgment) => {
      setHitAccuracy((prev) => ({
        ...prev,
        [judgment.toLowerCase()]: prev[judgment.toLowerCase() as keyof HitAccuracy] + 1,
      }));
    }, []),
    onJudgmentDisplay: useCallback(
      ({ judgment, points }: { judgment: Judgment; points: number }) =>
        showJudgment(judgment, points),
      [showJudgment]
    ),
    onMiss: handleMiss,
    onHoldStart: startHold1,
  });

  // Hit detection — P2
  const { checkHit: checkHit2 } = useHitDetection({
    activeArrows,
    processedStepsRef: processedStepsRefP2,
    combo: combo2,
    currentTime,
    onScoreUpdate: addScore2,
    onComboUpdate: useCallback((newCombo: number) => {
      setCombo2(newCombo);
      setMaxCombo2((prev) => Math.max(prev, newCombo));
    }, []),
    onHitAccuracyUpdate: useCallback((judgment: Judgment) => {
      setHitAccuracy2((prev) => ({
        ...prev,
        [judgment.toLowerCase()]: prev[judgment.toLowerCase() as keyof HitAccuracy] + 1,
      }));
    }, []),
    onJudgmentDisplay: useCallback(
      ({ judgment, points }: { judgment: Judgment; points: number }) =>
        showJudgment2(judgment, points),
      [showJudgment2]
    ),
    onMiss: handleMiss2,
    onHoldStart: startHold2,
  });

  // Miss tracking — once per player against their own ref.
  useMissTracking({
    activeArrows,
    processedStepsRef: processedStepsRefP1,
    activeHolds: activeHolds1,
    gameState,
    onMiss: handleMiss,
  });
  useMissTracking({
    activeArrows,
    processedStepsRef: processedStepsRefP2,
    activeHolds: activeHolds2,
    gameState,
    // Suppress P2 miss bookkeeping in solo mode — otherwise a P1-hit arrow
    // would fire P2 misses behind the scenes and clutter the (unused) P2 state.
    onMiss: IS_DUAL_MODE ? handleMiss2 : NOOP,
  });

  // Keyboard input — P1 always; P2 only listens to keys when dual-mode is on.
  const { activeKeys: activeKeys1 } = useKeyboardInput({
    gameState,
    keyMap: KEY_MAP,
    onArrowPress: checkHit1,
    onArrowRelease: releaseHold1,
    onPause: togglePause,
    enablePause: true,
  });
  const { activeKeys: activeKeys2 } = useKeyboardInput({
    gameState,
    keyMap: IS_DUAL_MODE ? KEY_MAP_P2 : EMPTY_KEYMAP,
    onArrowPress: checkHit2,
    onArrowRelease: releaseHold2,
    onPause: togglePause,
    enablePause: false,
  });

  // Audio player hook
  useAudioPlayer({
    audioRef,
    gameState,
    shouldAutoPlay: true,
  });

  const arrowSpeed = getArrowSpeed(songData.tempo || 120);
  const tempo = songData.tempo || 120;

  return (
    <div className="min-h-screen flex flex-col">
      <GameHUD
        score={score}
        score2={IS_DUAL_MODE ? score2 : undefined}
        currentTime={currentTime}
        songInfo={songData}
        gameState={gameState}
        onPause={togglePause}
        onReturnToMenu={resetGame}
      />

      <div className="flex-1 relative overflow-hidden bg-gradient-to-b from-black/50 to-transparent">
        <ArrowLane
          activeArrows={activeArrows}
          activeKeys={activeKeys1}
          activeHolds={activeHolds1}
          arrowSpeed={arrowSpeed}
          tempo={tempo}
          centerPercent={25}
          processedStepsRef={processedStepsRefP1}
        />

        {IS_DUAL_MODE && (
          <ArrowLane
            activeArrows={activeArrows}
            activeKeys={activeKeys2}
            activeHolds={activeHolds2}
            arrowSpeed={arrowSpeed}
            tempo={tempo}
            centerPercent={75}
            processedStepsRef={processedStepsRefP2}
          />
        )}

        <JudgmentDisplay
          judgment={judgmentDisplay}
          combo={combo}
          centerPercent={25}
        />

        {IS_DUAL_MODE && (
          <JudgmentDisplay
            judgment={judgmentDisplay2}
            combo={combo2}
            centerPercent={75}
          />
        )}
      </div>

      {gameState === GameState.PAUSED && <PauseOverlay onResume={togglePause} />}
    </div>
  );
}

const NOOP = () => {};
const EMPTY_KEYMAP: Record<string, never> = {};

export default GameScreen;

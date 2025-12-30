/**
 * Game Screen - orchestrates game play
 * Composes hooks and subcomponents for clean separation of concerns
 */

import { useState, useEffect } from 'react';
import { GameState } from '@/types/common.types';
import { useApp } from '@/app/providers/AppProvider';
import { Judgment, HitAccuracy, getArrowSpeed } from '../types/game.types';
import { calculateAccuracy } from '../utils/scoring';
import {
  useGameLoop,
  useHitDetection,
  useKeyboardInput,
  useAudioPlayer,
  useHoldTracking,
} from '../hooks';
import GameHUD from './GameHUD';
import ArrowLane from './ArrowLane';
import JudgmentDisplay from './JudgmentDisplay';
import PauseOverlay from './PauseOverlay';

function GameScreen() {
  const { steps, audioRef, songData, setGameResults, resetGame } = useApp();

  // Local game state
  const [gameState, setGameState] = useState<GameState>(GameState.PLAYING);
  const [score, setScore] = useState(0);
  const [combo, setCombo] = useState(0);
  const [maxCombo, setMaxCombo] = useState(0);
  const [hitAccuracy, setHitAccuracy] = useState<HitAccuracy>({
    perfect: 0,
    good: 0,
    ok: 0,
    miss: 0,
  });
  const [judgmentDisplay, setJudgmentDisplay] = useState<{
    judgment: Judgment;
    points: number;
  } | null>(null);

  if (!songData) return null;

  // Handle miss
  const handleMiss = () => {
    setCombo(0);
    setHitAccuracy((prev) => ({ ...prev, miss: prev.miss + 1 }));
    showJudgment(Judgment.MISS, 0);
  };

  // Show judgment feedback
  const showJudgment = (judgment: Judgment, points: number) => {
    setJudgmentDisplay({ judgment, points });
    setTimeout(() => setJudgmentDisplay(null), 500);
  };

  // Finish game
  const finishGame = () => {
    const totalNotes = Object.values(hitAccuracy).reduce((sum, val) => sum + val, 0);
    const accuracy = calculateAccuracy(hitAccuracy);

    setGameResults({
      score,
      maxCombo,
      hitAccuracy,
      accuracy,
      totalNotes,
    });
  };

  // Toggle pause
  const togglePause = () => {
    if (gameState === GameState.PLAYING) {
      setGameState(GameState.PAUSED);
    } else if (gameState === GameState.PAUSED) {
      setGameState(GameState.PLAYING);
    }
  };

  // Game loop hook
  const { currentTime, activeArrows, processedStepsRef } = useGameLoop({
    audioRef,
    steps,
    gameState,
    songDuration: songData.duration,
    tempo: songData.tempo || 120,
    onFinish: finishGame,
    onMiss: handleMiss,
  });

  // Hold tracking hook
  const { activeHolds, startHold, releaseHold, updateHolds } = useHoldTracking({
    combo,
    onScoreUpdate: (points) => setScore((prev) => prev + points),
  });

  // Update holds each frame when playing
  useEffect(() => {
    if (gameState === GameState.PLAYING) {
      updateHolds(currentTime);
    }
  }, [currentTime, gameState, updateHolds]);

  // Hit detection hook
  const { checkHit } = useHitDetection({
    activeArrows,
    processedStepsRef,
    combo,
    currentTime,
    onScoreUpdate: (points) => setScore((prev) => prev + points),
    onComboUpdate: (newCombo) => {
      setCombo(newCombo);
      setMaxCombo((prev) => Math.max(prev, newCombo));
    },
    onHitAccuracyUpdate: (judgment) => {
      setHitAccuracy((prev) => ({
        ...prev,
        [judgment.toLowerCase()]: prev[judgment.toLowerCase() as keyof HitAccuracy] + 1,
      }));
    },
    onJudgmentDisplay: ({ judgment, points }) => showJudgment(judgment, points),
    onMiss: handleMiss,
    onHoldStart: startHold,
  });

  // Keyboard input hook
  const { activeKeys } = useKeyboardInput({
    gameState,
    onArrowPress: checkHit,
    onArrowRelease: releaseHold,
    onPause: togglePause,
  });

  // Audio player hook
  useAudioPlayer({
    audioRef,
    gameState,
    shouldAutoPlay: true,
  });

  return (
    <div className="min-h-screen flex flex-col">
      <GameHUD
        score={score}
        combo={combo}
        currentTime={currentTime}
        songInfo={songData}
        gameState={gameState}
        onPause={togglePause}
        onReturnToMenu={resetGame}
      />

      <ArrowLane
        activeArrows={activeArrows}
        activeKeys={activeKeys}
        activeHolds={activeHolds}
        arrowSpeed={getArrowSpeed(songData.tempo || 120)}
      />

      <JudgmentDisplay judgment={judgmentDisplay} />

      {gameState === GameState.PAUSED && <PauseOverlay onResume={togglePause} />}
    </div>
  );
}

export default GameScreen;

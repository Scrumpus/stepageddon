/**
 * Game Screen - orchestrates game play
 * Composes hooks and subcomponents for clean separation of concerns
 */

import { useState, useEffect, useCallback } from 'react';
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
  const [_maxCombo, setMaxCombo] = useState(0);
  const [_hitAccuracy, setHitAccuracy] = useState<HitAccuracy>({
    perfect: 0,
    good: 0,
    ok: 0,
    miss: 0,
  });
  const [judgmentDisplay, setJudgmentDisplay] = useState<{
    judgment: Judgment;
    points: number;
  } | null>(null);

  // Show judgment feedback
  const showJudgment = useCallback((judgment: Judgment, points: number) => {
    setJudgmentDisplay({ judgment, points });
    setTimeout(() => setJudgmentDisplay(null), 500);
  }, []);

  // Handle miss
  const handleMiss = useCallback(() => {
    setCombo(0);
    setHitAccuracy((prev) => ({ ...prev, miss: prev.miss + 1 }));
    showJudgment(Judgment.MISS, 0);
  }, [showJudgment]);

  // Finish game
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

  const addScore = useCallback((points: number) => setScore((prev) => prev + points), []);

  // Hold tracking hook
  const { activeHolds, startHold, releaseHold, updateHolds } = useHoldTracking({
    combo,
    onScoreUpdate: addScore,
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
    onJudgmentDisplay: useCallback(({ judgment, points }: { judgment: Judgment; points: number }) => showJudgment(judgment, points), [showJudgment]),
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
        tempo={songData.tempo || 120}
      />

      <JudgmentDisplay judgment={judgmentDisplay} />

      {gameState === GameState.PAUSED && <PauseOverlay onResume={togglePause} />}
    </div>
  );
}

export default GameScreen;

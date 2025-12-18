import { useState, useEffect, useRef, useCallback } from 'react';
import { Pause, Play, Home, ArrowLeft, ArrowDown, ArrowUp, ArrowRight } from 'lucide-react';
import { 
  DIRECTIONS, 
  KEY_MAP, 
  TIMING, 
  HIT_ZONE_Y, 
  ARROW_SPEED, 
  GAME_STATES 
} from '../utils/gameConstants';
import { evaluateHit, calculatePoints } from '../utils/scoring';

// Arrow component icons
const ARROW_ICONS = {
  left: ArrowLeft,
  down: ArrowDown,
  up: ArrowUp,
  right: ArrowRight,
};

function GameScreen({ steps, audioRef, songInfo, difficulty, onFinish, onReturnToMenu }) {
  const [gameState, setGameState] = useState(GAME_STATES.PLAYING);
  const [currentTime, setCurrentTime] = useState(0);
  const [score, setScore] = useState(0);
  const [combo, setCombo] = useState(0);
  const [maxCombo, setMaxCombo] = useState(0);
  const [hitAccuracy, setHitAccuracy] = useState({
    perfect: 0,
    good: 0,
    ok: 0,
    miss: 0,
  });
  
  const [activeArrows, setActiveArrows] = useState([]);
  const [judgmentDisplay, setJudgmentDisplay] = useState(null);
  const [activeKeys, setActiveKeys] = useState({});
  
  const animationRef = useRef(null);
  const startTimeRef = useRef(null);
  const processedStepsRef = useRef(new Set());
  
  // Initialize game
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play();
      startTimeRef.current = Date.now();
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);
  
  // Game loop
  useEffect(() => {
    if (gameState !== GAME_STATES.PLAYING) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      return;
    }
    
    const gameLoop = () => {
      if (!audioRef.current) return;
      
      const currentTime = audioRef.current.currentTime;
      setCurrentTime(currentTime);
      
      // Update active arrows
      const visibleWindow = 2; // Show arrows 2 seconds ahead
      const newActiveArrows = [];
      
      steps.forEach((step, index) => {
        const timeUntilHit = step.time - currentTime;
        
        // Skip if already processed as miss
        if (processedStepsRef.current.has(index)) return;
        
        // Check for missed arrows
        if (timeUntilHit < -0.2 && !step.hit) {
          processedStepsRef.current.add(index);
          handleMiss();
          return;
        }
        
        // Show arrows in visible window
        if (timeUntilHit >= -0.2 && timeUntilHit <= visibleWindow) {
          const y = HIT_ZONE_Y - (timeUntilHit * ARROW_SPEED);
          newActiveArrows.push({
            ...step,
            index,
            y,
            timeUntilHit,
          });
        }
      });
      
      setActiveArrows(newActiveArrows);
      
      // Check if song is finished
      if (currentTime >= songInfo.duration - 0.5) {
        finishGame();
        return;
      }
      
      animationRef.current = requestAnimationFrame(gameLoop);
    };
    
    animationRef.current = requestAnimationFrame(gameLoop);
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [gameState, steps, songInfo.duration]);
  
  // Handle key press
  const handleKeyDown = useCallback((e) => {
    const direction = KEY_MAP[e.key];
    
    if (direction && gameState === GAME_STATES.PLAYING) {
      e.preventDefault();
      setActiveKeys(prev => ({ ...prev, [direction]: true }));
      checkHit(direction);
    } else if (e.key === 'Escape') {
      togglePause();
    }
  }, [gameState, activeArrows, currentTime]);
  
  const handleKeyUp = useCallback((e) => {
    const direction = KEY_MAP[e.key];
    if (direction) {
      setActiveKeys(prev => ({ ...prev, [direction]: false }));
    }
  }, []);
  
  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [handleKeyDown, handleKeyUp]);
  
  // Check for hit
  const checkHit = (direction) => {
    const hittableArrows = activeArrows.filter(
      arrow => 
        arrow.direction === direction && 
        !arrow.hit && 
        Math.abs(arrow.timeUntilHit) <= TIMING.MISS / 1000
    );
    
    if (hittableArrows.length === 0) return;
    
    // Find closest arrow
    const closest = hittableArrows.reduce((prev, curr) => 
      Math.abs(curr.timeUntilHit) < Math.abs(prev.timeUntilHit) ? curr : prev
    );
    
    const timingMs = closest.timeUntilHit * 1000;
    const judgment = evaluateHit(timingMs);
    
    // Mark as hit
    closest.hit = true;
    processedStepsRef.current.add(closest.index);
    
    // Update score and combo
    if (judgment !== 'MISS') {
      const newCombo = combo + 1;
      const points = calculatePoints(judgment, newCombo);
      
      setScore(prev => prev + points);
      setCombo(newCombo);
      setMaxCombo(prev => Math.max(prev, newCombo));
      setHitAccuracy(prev => ({
        ...prev,
        [judgment.toLowerCase()]: prev[judgment.toLowerCase()] + 1,
      }));
      
      // Show judgment
      showJudgment(judgment, points);
    } else {
      handleMiss();
    }
  };
  
  const handleMiss = () => {
    setCombo(0);
    setHitAccuracy(prev => ({ ...prev, miss: prev.miss + 1 }));
    showJudgment('MISS', 0);
  };
  
  const showJudgment = (judgment, points) => {
    setJudgmentDisplay({ judgment, points });
    setTimeout(() => setJudgmentDisplay(null), 500);
  };
  
  const togglePause = () => {
    if (gameState === GAME_STATES.PLAYING) {
      audioRef.current?.pause();
      setGameState(GAME_STATES.PAUSED);
    } else if (gameState === GAME_STATES.PAUSED) {
      audioRef.current?.play();
      setGameState(GAME_STATES.PLAYING);
    }
  };
  
  const finishGame = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    audioRef.current?.pause();
    
    const totalNotes = Object.values(hitAccuracy).reduce((sum, val) => sum + val, 0);
    const accuracy = totalNotes > 0
      ? ((hitAccuracy.perfect * 100 + hitAccuracy.good * 66 + hitAccuracy.ok * 33) / (totalNotes * 100)) * 100
      : 0;
    
    onFinish({
      score,
      maxCombo,
      hitAccuracy,
      accuracy,
      totalNotes,
    });
  };
  
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <div className="bg-black/50 backdrop-blur-sm p-4 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <div className="text-sm text-gray-400">Score</div>
            <div className="text-2xl font-bold">{score.toLocaleString()}</div>
          </div>
          
          <div>
            <div className="text-sm text-gray-400">Combo</div>
            <div className="text-2xl font-bold text-game-accent">{combo}x</div>
          </div>
        </div>
        
        <div className="text-center">
          <div className="text-lg font-semibold">{songInfo.title}</div>
          <div className="text-sm text-gray-400">
            {Math.floor(currentTime)}s / {Math.floor(songInfo.duration)}s
          </div>
        </div>
        
        <div className="flex gap-2">
          <button
            onClick={togglePause}
            className="p-3 bg-white/10 hover:bg-white/20 rounded-lg transition-all"
          >
            {gameState === GAME_STATES.PAUSED ? (
              <Play className="w-5 h-5" />
            ) : (
              <Pause className="w-5 h-5" />
            )}
          </button>
          
          <button
            onClick={onReturnToMenu}
            className="p-3 bg-white/10 hover:bg-white/20 rounded-lg transition-all"
          >
            <Home className="w-5 h-5" />
          </button>
        </div>
      </div>
      
      {/* Progress Bar */}
      <div className="h-1 bg-white/10">
        <div
          className="h-full bg-gradient-to-r from-game-primary to-game-accent transition-all"
          style={{ width: `${(currentTime / songInfo.duration) * 100}%` }}
        />
      </div>
      
      {/* Game Area */}
      <div className="flex-1 relative overflow-hidden">
        {/* Target Zone */}
        <div 
          className="absolute left-1/2 transform -translate-x-1/2 flex gap-4 z-10"
          style={{ top: `${HIT_ZONE_Y}px` }}
        >
          {Object.values(DIRECTIONS).map(direction => {
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
        
        {/* Arrows */}
        {activeArrows.map((arrow, idx) => {
          const Icon = ARROW_ICONS[arrow.direction];
          const directionIndex = Object.values(DIRECTIONS).indexOf(arrow.direction);
          const x = -160 + (directionIndex * 96);
          
          return (
            <div
              key={`${arrow.index}-${idx}`}
              className="absolute left-1/2 transform -translate-x-1/2 transition-opacity"
              style={{
                top: `${arrow.y}px`,
                left: `calc(50% + ${x}px)`,
                opacity: arrow.hit ? 0 : 1,
              }}
            >
              <div className="w-20 h-20 bg-game-primary/80 rounded-lg flex items-center justify-center border-2 border-game-primary">
                <Icon className="w-10 h-10" />
              </div>
            </div>
          );
        })}
        
        {/* Judgment Display */}
        {judgmentDisplay && (
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 pointer-events-none">
            <div className="text-center animate-pulse-hit">
              <div className={`text-4xl font-bold ${
                judgmentDisplay.judgment === 'PERFECT' ? 'text-yellow-400' :
                judgmentDisplay.judgment === 'GOOD' ? 'text-green-400' :
                judgmentDisplay.judgment === 'OK' ? 'text-blue-400' :
                'text-red-400'
              }`}>
                {judgmentDisplay.judgment}
              </div>
              {judgmentDisplay.points > 0 && (
                <div className="text-2xl text-white">
                  +{judgmentDisplay.points}
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Pause Overlay */}
        {gameState === GAME_STATES.PAUSED && (
          <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="text-center">
              <h2 className="text-4xl font-bold mb-4">Paused</h2>
              <p className="text-gray-400 mb-8">Press ESC or click Play to resume</p>
              <button
                onClick={togglePause}
                className="px-8 py-4 bg-gradient-to-r from-game-primary to-game-accent rounded-lg font-semibold hover:shadow-lg transition-all"
              >
                <Play className="w-6 h-6 inline mr-2" />
                Resume
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default GameScreen;

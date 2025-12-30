/**
 * Application-level context provider
 * Manages global state and eliminates prop drilling
 */

import { createContext, useContext, useState, useRef, useEffect, ReactNode } from 'react';
import { GameState, DifficultyLevel, SongInfo } from '@/types/common.types';
import { Step } from '@/features/game/types/step.types';
import { GameResults } from '@/features/results/types/results.types';
import { useToast } from '@/hooks/useToast';
import ToastContainer from '@/components/Toast/ToastContainer';

interface AppContextValue {
  // Game flow state
  gameState: GameState;
  setGameState: (state: GameState) => void;

  // User inputs
  difficulty: DifficultyLevel;
  setDifficulty: (difficulty: DifficultyLevel) => void;

  // Song data
  songData: SongInfo | null;
  setSongData: (data: SongInfo | null) => void;
  steps: Step[];
  setSteps: (steps: Step[]) => void;
  audioUrl: string | null;
  setAudioUrl: (url: string | null) => void;

  // Loading state
  loadingMessage: string;
  setLoadingMessage: (message: string) => void;
  loadingProgress: number;
  setLoadingProgress: (progress: number) => void;

  // Results
  gameResults: GameResults | null;
  setGameResults: (results: GameResults | null) => void;

  // Audio ref
  audioRef: React.RefObject<HTMLAudioElement>;

  // Toast notifications
  showToast: (message: string, type?: 'success' | 'error' | 'info' | 'warning', duration?: number) => void;

  // Actions
  resetGame: () => void;
}

const AppContext = createContext<AppContextValue | undefined>(undefined);

interface AppProviderProps {
  children: ReactNode;
}

function AppProvider({ children }: AppProviderProps) {
  // Game flow state
  const [gameState, setGameState] = useState<GameState>(GameState.MENU);
  const [difficulty, setDifficulty] = useState<DifficultyLevel>('intermediate');

  // Song data
  const [songData, setSongData] = useState<SongInfo | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // Loading state
  const [loadingMessage, setLoadingMessage] = useState('');
  const [loadingProgress, setLoadingProgress] = useState(0);

  // Results
  const [gameResults, setGameResults] = useState<GameResults | null>(null);

  // Audio ref
  const audioRef = useRef<HTMLAudioElement>(null);

  // Toast notifications
  const { toasts, showToast, hideToast } = useToast();

  // Reset game state
  const resetGame = () => {
    setGameState(GameState.MENU);
    setSongData(null);
    setSteps([]);
    setAudioUrl(null);
    setGameResults(null);

    // Stop audio if playing
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);

  const value: AppContextValue = {
    gameState,
    setGameState,
    difficulty,
    setDifficulty,
    songData,
    setSongData,
    steps,
    setSteps,
    audioUrl,
    setAudioUrl,
    loadingMessage,
    setLoadingMessage,
    loadingProgress,
    setLoadingProgress,
    gameResults,
    setGameResults,
    audioRef,
    showToast,
    resetGame,
  };

  return (
    <AppContext.Provider value={value}>
      {children}
      {/* Toast notifications */}
      <ToastContainer toasts={toasts} onDismiss={hideToast} />
      {/* Audio element */}
      {audioUrl && <audio ref={audioRef} src={audioUrl} preload="auto" />}
    </AppContext.Provider>
  );
}

/**
 * Hook to access app context
 * Must be used within AppProvider
 */
function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
}

export default AppProvider;
export { useApp };

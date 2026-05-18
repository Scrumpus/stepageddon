import { createContext, useContext, useEffect, useRef, ReactNode } from 'react';
import { GameState } from '@/types/common.types';
import { useGameStore } from '@/app/store/useGameStore';

interface AudioApi {
  audioRef: React.RefObject<HTMLAudioElement>;
}

const AudioContext = createContext<AudioApi | undefined>(undefined);

interface AudioProviderProps {
  children: ReactNode;
}

function AudioProvider({ children }: AudioProviderProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const audioUrl = useGameStore((s) => s.audioUrl);
  const gameState = useGameStore((s) => s.gameState);

  // Pause/rewind whenever the FSM returns to MENU.
  useEffect(() => {
    if (gameState === GameState.MENU && audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  }, [gameState]);

  useEffect(() => {
    return () => {
      if (audioRef.current) audioRef.current.pause();
    };
  }, []);

  return (
    <AudioContext.Provider value={{ audioRef }}>
      {children}
      {audioUrl && <audio ref={audioRef} src={audioUrl} preload="auto" />}
    </AudioContext.Provider>
  );
}

export function useAudio(): AudioApi {
  const ctx = useContext(AudioContext);
  if (!ctx) throw new Error('useAudio must be used within AudioProvider');
  return ctx;
}

export default AudioProvider;

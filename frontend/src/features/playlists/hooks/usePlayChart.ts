import { useCallback, useState } from 'react';
import { useApp } from '@/app/providers/AppProvider';
import { GameState } from '@/types/common.types';
import { getAudioUrl } from '@/lib/axios';
import { getChart } from '../api';
import { ChartSummaryDTO, SongSummaryDTO } from '../types';

/**
 * Loads a chart from the database and hands it off to the existing
 * LOADING → READY → PLAYING pipeline by populating AppContext the same
 * way `useStepGeneration` does.
 */
export function usePlayChart() {
  const {
    setSongData,
    setSteps,
    setAudioUrl,
    setGameState,
    setLoadingMessage,
    setLoadingProgress,
    showToast,
  } = useApp();

  const [isLoading, setIsLoading] = useState(false);

  const playChart = useCallback(
    async (song: SongSummaryDTO, chart: ChartSummaryDTO) => {
      try {
        setIsLoading(true);
        setGameState(GameState.LOADING);
        setLoadingMessage(`Loading ${song.title}…`);
        setLoadingProgress(40);

        const fullChart = await getChart(song.id, chart.difficulty_name);

        setLoadingProgress(100);
        setLoadingMessage('Ready!');

        setSongData({
          title: song.title,
          artist: song.artist ?? undefined,
          duration: song.duration,
          tempo: song.tempo,
          thumbnail: song.jacket_url ?? song.banner_url ?? undefined,
        });
        setSteps(fullChart.steps);
        setAudioUrl(getAudioUrl(song.audio_url));

        setTimeout(() => {
          setGameState(GameState.READY);
          setIsLoading(false);
        }, 400);
      } catch (e: any) {
        const msg =
          e.response?.data?.detail || e.message || 'Failed to load chart';
        showToast(msg, 'error');
        setGameState(GameState.MENU);
        setIsLoading(false);
      }
    },
    [
      setGameState,
      setLoadingMessage,
      setLoadingProgress,
      setSongData,
      setSteps,
      setAudioUrl,
      showToast,
    ]
  );

  return { playChart, isLoading };
}

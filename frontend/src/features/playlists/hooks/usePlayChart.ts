import { useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGameStore } from '@/app/store/useGameStore';
import { useToastApi } from '@/app/providers/ToastProvider';
import { GameState } from '@/types/common.types';
import { getAudioUrl } from '@/lib/axios';
import { getChart } from '../api';
import { ChartSummaryDTO, SongSummaryDTO } from '../types';

/**
 * Loads a chart from the database and hands it off to the existing
 * LOADING → READY → PLAYING pipeline.
 */
export function usePlayChart() {
  const chartLoadStarted = useGameStore((s) => s.chartLoadStarted);
  const chartLoaded = useGameStore((s) => s.chartLoaded);
  const setGameState = useGameStore((s) => s.setGameState);
  const { showToast } = useToastApi();

  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);

  const playChart = useCallback(
    async (song: SongSummaryDTO, chart: ChartSummaryDTO) => {
      try {
        setIsLoading(true);
        navigate('/game');
        chartLoadStarted({ message: `Loading ${song.title}…`, progress: 40 });

        const fullChart = await getChart(song.id, chart.difficulty_name);

        chartLoaded({
          songData: {
            title: song.title,
            artist: song.artist ?? undefined,
            duration: song.duration,
            tempo: song.tempo,
            thumbnail: song.jacket_url ?? song.banner_url ?? undefined,
            // Carries chart origin so the receptor pulse only applies the
            // generation offset to ML charts, not imported DDR .sm charts.
            source: song.source,
          },
          steps: fullChart.steps,
          audioUrl: getAudioUrl(song.audio_url),
        });

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
    [chartLoadStarted, chartLoaded, setGameState, showToast],
  );

  return { playChart, isLoading };
}

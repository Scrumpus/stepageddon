import { useEffect, useState } from 'react';
import { getSong } from '../api';
import { ChartSummaryDTO, SongDetailDTO } from '@/features/playlists/types';

interface State {
  song: SongDetailDTO | null;
  charts: ChartSummaryDTO[];
  isLoading: boolean;
  error: string | null;
}

const initial: State = {
  song: null,
  charts: [],
  isLoading: false,
  error: null,
};

export function useSongDetail(songId: string | null) {
  const [state, setState] = useState<State>(initial);

  useEffect(() => {
    if (!songId) {
      setState(initial);
      return;
    }

    let cancelled = false;
    setState({ song: null, charts: [], isLoading: true, error: null });

    (async () => {
      try {
        const song = await getSong(songId);
        if (cancelled) return;
        const charts = [...song.charts].sort(
          (a, b) => a.difficulty_level - b.difficulty_level
        );
        setState({ song, charts, isLoading: false, error: null });
      } catch (e: any) {
        if (cancelled) return;
        setState({
          song: null,
          charts: [],
          isLoading: false,
          error: e.response?.data?.detail || e.message || 'Failed to load song',
        });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [songId]);

  return state;
}

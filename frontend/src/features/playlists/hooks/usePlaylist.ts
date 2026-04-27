import { useEffect, useState } from 'react';
import { getPlaylist, getSong } from '../api';
import { PlaylistDetailDTO, SongWithCharts } from '../types';

interface State {
  playlist: PlaylistDetailDTO | null;
  songsWithCharts: SongWithCharts[];
  isLoading: boolean;
  error: string | null;
}

export function usePlaylist(playlistId: string | null) {
  const [state, setState] = useState<State>({
    playlist: null,
    songsWithCharts: [],
    isLoading: false,
    error: null,
  });

  useEffect(() => {
    if (!playlistId) {
      setState({ playlist: null, songsWithCharts: [], isLoading: false, error: null });
      return;
    }

    let cancelled = false;
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    (async () => {
      try {
        const playlist = await getPlaylist(playlistId);
        if (cancelled) return;

        const songs = playlist.songs;
        const details = await Promise.all(songs.map((s) => getSong(s.id)));
        if (cancelled) return;

        const songsWithCharts: SongWithCharts[] = songs.map((song, i) => ({
          song,
          charts: [...details[i].charts].sort(
            (a, b) => a.difficulty_level - b.difficulty_level
          ),
        }));

        setState({
          playlist,
          songsWithCharts,
          isLoading: false,
          error: null,
        });
      } catch (e: any) {
        if (cancelled) return;
        setState({
          playlist: null,
          songsWithCharts: [],
          isLoading: false,
          error: e.response?.data?.detail || e.message || 'Failed to load playlist',
        });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [playlistId]);

  return state;
}

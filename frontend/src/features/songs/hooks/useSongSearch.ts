import { useCallback, useEffect, useRef, useState } from 'react';
import { listSongs } from '../api';
import { SongSummaryDTO } from '@/features/playlists/types';

const PAGE_SIZE = 50;
const DEBOUNCE_MS = 300;

interface State {
  songs: SongSummaryDTO[];
  isLoading: boolean;
  isLoadingMore: boolean;
  error: string | null;
  hasMore: boolean;
  offset: number;
}

const initial: State = {
  songs: [],
  isLoading: true,
  isLoadingMore: false,
  error: null,
  hasMore: false,
  offset: 0,
};

export function useSongSearch() {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [state, setState] = useState<State>(initial);
  const requestIdRef = useRef(0);

  // Debounce the query
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query), DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [query]);

  // Fetch first page whenever the debounced query changes
  useEffect(() => {
    const reqId = ++requestIdRef.current;
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    (async () => {
      try {
        const data = await listSongs({
          q: debouncedQuery || undefined,
          limit: PAGE_SIZE,
          offset: 0,
        });
        if (reqId !== requestIdRef.current) return;
        setState({
          songs: data,
          isLoading: false,
          isLoadingMore: false,
          error: null,
          hasMore: data.length === PAGE_SIZE,
          offset: data.length,
        });
      } catch (e: any) {
        if (reqId !== requestIdRef.current) return;
        setState({
          songs: [],
          isLoading: false,
          isLoadingMore: false,
          error:
            e.response?.data?.detail || e.message || 'Failed to load songs',
          hasMore: false,
          offset: 0,
        });
      }
    })();
  }, [debouncedQuery]);

  const loadMore = useCallback(async () => {
    if (state.isLoading || state.isLoadingMore || !state.hasMore) return;
    const reqId = requestIdRef.current;
    setState((prev) => ({ ...prev, isLoadingMore: true }));
    try {
      const data = await listSongs({
        q: debouncedQuery || undefined,
        limit: PAGE_SIZE,
        offset: state.offset,
      });
      if (reqId !== requestIdRef.current) return;
      setState((prev) => ({
        ...prev,
        songs: [...prev.songs, ...data],
        isLoadingMore: false,
        hasMore: data.length === PAGE_SIZE,
        offset: prev.offset + data.length,
      }));
    } catch (e: any) {
      if (reqId !== requestIdRef.current) return;
      setState((prev) => ({
        ...prev,
        isLoadingMore: false,
        error:
          e.response?.data?.detail || e.message || 'Failed to load more songs',
      }));
    }
  }, [debouncedQuery, state.isLoading, state.isLoadingMore, state.hasMore, state.offset]);

  return {
    query,
    setQuery,
    songs: state.songs,
    isLoading: state.isLoading,
    isLoadingMore: state.isLoadingMore,
    error: state.error,
    hasMore: state.hasMore,
    loadMore,
  };
}

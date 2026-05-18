/**
 * Debounced YouTube Music search hook.
 *
 * Modeled on features/songs/hooks/useSongSearch.ts. Skips network calls when
 * the trimmed query is empty.
 */

import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { searchSongs } from '../api';
import type { SongSearchResult } from '../types/menu.types';

const DEBOUNCE_MS = 300;
const LIMIT = 10;

interface State {
  results: SongSearchResult[];
  isLoading: boolean;
  error: string | null;
}

const initial: State = {
  results: [],
  isLoading: false,
  error: null,
};

export function useYouTubeMusicSearch() {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [state, setState] = useState<State>(initial);
  const requestIdRef = useRef(0);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query.trim()), DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [query]);

  useEffect(() => {
    if (!debouncedQuery) {
      setState(initial);
      return;
    }

    const reqId = ++requestIdRef.current;
    const controller = new AbortController();
    setState({ results: [], isLoading: true, error: null });

    (async () => {
      try {
        const data = await searchSongs(debouncedQuery, LIMIT, controller.signal);
        if (reqId !== requestIdRef.current) return;
        setState({ results: data, isLoading: false, error: null });
      } catch (e: any) {
        if (axios.isCancel(e) || e.name === 'CanceledError') return;
        if (reqId !== requestIdRef.current) return;
        setState({
          results: [],
          isLoading: false,
          error:
            e.response?.data?.detail || e.message || 'Search failed',
        });
      }
    })();

    return () => controller.abort();
  }, [debouncedQuery]);

  return {
    query,
    setQuery,
    results: state.results,
    isLoading: state.isLoading,
    error: state.error,
  };
}

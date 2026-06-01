/**
 * Discovery feed hook. Refetches whenever the feed or genre changes, cancelling
 * any in-flight request (same pattern as features/menu/hooks/useSongSearch).
 * Genres load once on mount.
 */

import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { fetchDiscover, fetchGenres } from '../api';
import type { DiscoverFeed, Genre } from '../api';
import type { SongSearchResult } from '@/types/common.types';

const LIMIT = 24;

interface State {
  results: SongSearchResult[];
  isLoading: boolean;
  error: string | null;
}

const initial: State = { results: [], isLoading: true, error: null };

export function useDiscover() {
  const [feed, setFeed] = useState<DiscoverFeed>('trending');
  // null genre = no filter ("All"). Underground ignores the genre entirely.
  const [genre, setGenre] = useState<string | null>(null);
  const [genres, setGenres] = useState<Genre[]>([]);
  const [state, setState] = useState<State>(initial);
  const requestIdRef = useRef(0);

  // Load the genre chips once.
  useEffect(() => {
    const controller = new AbortController();
    fetchGenres(controller.signal)
      .then(setGenres)
      .catch((e) => {
        if (!axios.isCancel(e)) setGenres([]);
      });
    return () => controller.abort();
  }, []);

  // Underground takes no genre filter — only pass genre on the trending feed.
  const effectiveGenre = feed === 'underground' ? null : genre;

  useEffect(() => {
    const reqId = ++requestIdRef.current;
    const controller = new AbortController();
    setState({ results: [], isLoading: true, error: null });

    (async () => {
      try {
        const data = await fetchDiscover(
          feed,
          effectiveGenre,
          LIMIT,
          controller.signal,
        );
        if (reqId !== requestIdRef.current) return;
        setState({ results: data, isLoading: false, error: null });
      } catch (e: any) {
        if (axios.isCancel(e) || e.name === 'CanceledError') return;
        if (reqId !== requestIdRef.current) return;
        setState({
          results: [],
          isLoading: false,
          error: e.response?.data?.detail || e.message || 'Discovery failed',
        });
      }
    })();

    return () => controller.abort();
  }, [feed, effectiveGenre]);

  return {
    feed,
    setFeed,
    genre,
    setGenre,
    genres,
    results: state.results,
    isLoading: state.isLoading,
    error: state.error,
  };
}

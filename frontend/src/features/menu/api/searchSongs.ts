/**
 * YouTube Music song search.
 */

import { api } from '@/lib/axios';
import type { SongSearchResult, SongSearchResponse } from '../types/menu.types';

export async function searchSongs(
  query: string,
  limit = 10,
  signal?: AbortSignal,
): Promise<SongSearchResult[]> {
  const response = await api.get<SongSearchResponse>('/api/search-songs', {
    params: { q: query, limit },
    signal,
  });
  return response.data.results ?? [];
}

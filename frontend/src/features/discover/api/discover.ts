/**
 * Song discovery (Audius + Jamendo browse feeds, merged) via /api/discover.
 */

import { api } from '@/lib/axios';
import type {
  SongSearchResult,
  SongSearchResponse,
} from '@/types/common.types';

export type DiscoverFeed = 'trending' | 'underground';

export interface Genre {
  key: string;
  label: string;
}

export async function fetchDiscover(
  feed: DiscoverFeed,
  genre: string | null,
  limit = 20,
  signal?: AbortSignal,
): Promise<SongSearchResult[]> {
  const response = await api.get<SongSearchResponse>('/api/discover', {
    params: { feed, ...(genre ? { genre } : {}), limit },
    signal,
  });
  return response.data.results ?? [];
}

export async function fetchGenres(signal?: AbortSignal): Promise<Genre[]> {
  const response = await api.get<{ genres: Genre[] }>('/api/discover/genres', {
    signal,
  });
  return response.data.genres ?? [];
}

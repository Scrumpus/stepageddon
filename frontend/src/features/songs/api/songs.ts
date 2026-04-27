/**
 * Songs API client.
 */

import { api } from '@/lib/axios';
import { SongSummaryDTO, SongSource } from '@/features/playlists/types';

export interface ListSongsParams {
  q?: string;
  limit?: number;
  offset?: number;
  source?: SongSource;
  game?: string;
}

export const listSongs = (
  params: ListSongsParams = {}
): Promise<SongSummaryDTO[]> =>
  api
    .get<SongSummaryDTO[]>('/api/songs', { params })
    .then((r) => r.data);

export { getSong } from '@/features/playlists/api';

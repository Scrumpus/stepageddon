/**
 * Playlists / songs / charts API client.
 */

import { api } from '@/lib/axios';
import {
  ChartDTO,
  PlaylistDTO,
  PlaylistDetailDTO,
  SongDetailDTO,
} from '../types';

export const listPlaylists = (): Promise<PlaylistDTO[]> =>
  api.get<PlaylistDTO[]>('/api/playlists').then((r) => r.data);

export const getPlaylist = (id: string): Promise<PlaylistDetailDTO> =>
  api.get<PlaylistDetailDTO>(`/api/playlists/${id}`).then((r) => r.data);

export const getSong = (id: string): Promise<SongDetailDTO> =>
  api.get<SongDetailDTO>(`/api/songs/${id}`).then((r) => r.data);

export const getChart = (
  songId: string,
  difficultyName: string
): Promise<ChartDTO> =>
  api
    .get<ChartDTO>(
      `/api/songs/${songId}/charts/${encodeURIComponent(difficultyName)}`
    )
    .then((r) => r.data);

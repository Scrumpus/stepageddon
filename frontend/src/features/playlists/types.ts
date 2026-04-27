/**
 * Backend DTO mirrors for playlists, songs, and charts.
 */

import { Step } from '@/features/game/types/step.types';

export type SongSource = 'USER' | 'DDR';

export interface SongSummaryDTO {
  id: string;
  title: string;
  artist: string | null;
  source: SongSource;
  slug: string;
  audio_url: string;
  banner_url: string | null;
  jacket_url: string | null;
  bg_url: string | null;
  tempo: number;
  duration: number;
  ddr_game: string | null;
  ddr_release_index: number | null;
}

export interface ChartSummaryDTO {
  id: string;
  difficulty_name: string;
  difficulty_level: number;
  chart_type: string;
  step_count: number;
  hold_count: number;
  generator: string;
}

export interface SongDetailDTO extends SongSummaryDTO {
  offset: number;
  bpms: Array<[number, number]> | null;
  charts: ChartSummaryDTO[];
}

export interface ChartDTO {
  id: string;
  song_id: string;
  difficulty_name: string;
  difficulty_level: number;
  chart_type: string;
  steps: Step[];
  step_count: number;
  hold_count: number;
  radar_values: number[] | null;
  generator: string;
}

export interface PlaylistDTO {
  id: string;
  name: string;
  description: string | null;
  song_count: number;
  created_at: string;
  updated_at: string;
}

export interface PlaylistDetailDTO extends PlaylistDTO {
  songs: SongSummaryDTO[];
}

export interface SongWithCharts {
  song: SongSummaryDTO;
  charts: ChartSummaryDTO[];
}

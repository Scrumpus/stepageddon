/**
 * StepMania (.sm) export helpers.
 *
 * Two sources, mirroring the backend:
 *  - library songs  → GET  /api/songs/{id}/export   (read from DB)
 *  - generated charts → POST /api/charts/export      (posted in-memory payload)
 *
 * Both return a zipped song folder (.sm + audio + art) as a blob, which we hand
 * to the browser as a download.
 */

import { api } from './axios';
import { Step } from '@/features/game/types/step.types';
import { SongInfo } from '@/types/common.types';

/** Trigger a browser download for a blob, honoring the server filename. */
function downloadBlob(blob: Blob, disposition: string | undefined, fallback: string) {
  const match = disposition?.match(/filename="?([^"]+)"?/i);
  const filename = match?.[1] ?? fallback;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** Export a library song (all difficulties) as a StepMania zip. */
export async function exportLibrarySong(songId: string, title: string): Promise<void> {
  const res = await api.get(`/api/songs/${songId}/export`, { responseType: 'blob' });
  downloadBlob(res.data, res.headers['content-disposition'], `${title || 'song'}.zip`);
}

/** Export freshly generated charts (not yet persisted) as a StepMania zip. */
export async function exportGeneratedCharts(params: {
  songInfo: SongInfo;
  audioUrl: string;
  stepsByDifficulty: Partial<Record<string, Step[]>>;
}): Promise<void> {
  const charts: Record<string, { steps: Step[] }> = {};
  for (const [difficulty, steps] of Object.entries(params.stepsByDifficulty)) {
    if (steps && steps.length) charts[difficulty] = { steps };
  }
  const body = {
    song_info: {
      title: params.songInfo.title,
      artist: params.songInfo.artist ?? '',
      tempo: params.songInfo.tempo,
      timing_offset: params.songInfo.timing_offset ?? 0,
    },
    audio_url: params.audioUrl,
    charts,
  };
  const res = await api.post('/api/charts/export', body, { responseType: 'blob' });
  downloadBlob(
    res.data,
    res.headers['content-disposition'],
    `${params.songInfo.title || 'song'}.zip`,
  );
}

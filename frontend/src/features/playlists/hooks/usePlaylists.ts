import { useCallback, useEffect, useState } from 'react';
import { listPlaylists } from '../api';
import { PlaylistDTO } from '../types';

export function usePlaylists() {
  const [playlists, setPlaylists] = useState<PlaylistDTO[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await listPlaylists();
      setPlaylists(data);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || 'Failed to load playlists');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { playlists, isLoading, error, refetch };
}

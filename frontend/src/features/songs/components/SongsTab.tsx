/**
 * SongsTab — list or detail view, driven by URL params.
 * /songs       → SongList
 * /songs/:id   → SongDetail
 */

import { useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { SongSummaryDTO } from '@/features/playlists/types';
import SongList from './SongList';
import SongDetail from './SongDetail';

function SongsTab() {
  const { songId } = useParams();
  const navigate = useNavigate();
  const [fallbackSong, setFallbackSong] = useState<SongSummaryDTO | null>(null);

  if (songId && fallbackSong) {
    return (
      <SongDetail
        songId={songId}
        fallbackSong={fallbackSong}
        onBack={() => navigate('/songs')}
      />
    );
  }

  return (
    <SongList
      onSelect={(song) => {
        setFallbackSong(song);
        navigate(`/songs/${song.id}`);
      }}
    />
  );
}

export default SongsTab;

/**
 * PlaylistsTab — list or detail view, driven by URL params.
 * /playlists      → PlaylistList
 * /playlists/:id  → PlaylistDetail
 */

import { useNavigate, useParams } from 'react-router-dom';
import PlaylistList from './PlaylistList';
import PlaylistDetail from './PlaylistDetail';

function PlaylistsTab() {
  const { playlistId } = useParams();
  const navigate = useNavigate();

  if (playlistId) {
    return (
      <PlaylistDetail
        playlistId={playlistId}
        onBack={() => navigate('/playlists')}
      />
    );
  }

  return (
    <PlaylistList
      onSelect={(id) => navigate(`/playlists/${id}`)}
    />
  );
}

export default PlaylistsTab;

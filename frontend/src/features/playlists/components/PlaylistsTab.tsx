import { useState } from 'react';
import PlaylistList from './PlaylistList';
import PlaylistDetail from './PlaylistDetail';

type View = { type: 'list' } | { type: 'detail'; playlistId: string };

function PlaylistsTab() {
  const [view, setView] = useState<View>({ type: 'list' });

  if (view.type === 'detail') {
    return (
      <PlaylistDetail
        playlistId={view.playlistId}
        onBack={() => setView({ type: 'list' })}
      />
    );
  }

  return (
    <PlaylistList
      onSelect={(playlistId) => setView({ type: 'detail', playlistId })}
    />
  );
}

export default PlaylistsTab;

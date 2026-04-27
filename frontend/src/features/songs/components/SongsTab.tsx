import { useState } from 'react';
import SongList from './SongList';
import SongDetail from './SongDetail';

type View = { type: 'list' } | { type: 'detail'; songId: string };

function SongsTab() {
  const [view, setView] = useState<View>({ type: 'list' });

  if (view.type === 'detail') {
    return (
      <SongDetail
        songId={view.songId}
        onBack={() => setView({ type: 'list' })}
      />
    );
  }

  return (
    <SongList
      onSelect={(songId) => setView({ type: 'detail', songId })}
    />
  );
}

export default SongsTab;

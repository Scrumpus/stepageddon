import { useState } from 'react';
import { SongSummaryDTO } from '@/features/playlists/types';
import SongList from './SongList';
import SongDetail from './SongDetail';

function SongsTab() {
  const [selected, setSelected] = useState<SongSummaryDTO | null>(null);

  if (selected) {
    return (
      <SongDetail
        songId={selected.id}
        fallbackSong={selected}
        onBack={() => setSelected(null)}
      />
    );
  }

  return <SongList onSelect={setSelected} />;
}

export default SongsTab;

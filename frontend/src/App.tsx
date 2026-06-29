/**
 * App — top-level route definitions.
 *
 * /playlists | /songs | /discover | /create → MenuScreen layout + tab content
 * /game                                      → GameFlow (Zustand-driven screens)
 */

import { Navigate, Route, Routes } from 'react-router-dom';
import MenuScreen from '@/features/menu/components/MenuScreen';
import CreateTab from '@/features/menu/components/CreateTab';
import GameFlow from '@/features/menu/components/GameFlow';
import { PlaylistsTab } from '@/features/playlists/components';
import { SongsTab } from '@/features/songs/components';
import { DiscoverTab } from '@/features/discover/components';

function App() {
  return (
    <div className="min-h-screen bg-game-bg">
      <Routes>
        {/* Menu shell — renders tab bar + Outlet */}
        <Route element={<MenuScreen />}>
          <Route index element={<Navigate to="/playlists" replace />} />
          <Route path="playlists" element={<PlaylistsTab />} />
          <Route path="playlists/:playlistId" element={<PlaylistsTab />} />
          <Route path="songs" element={<SongsTab />} />
          <Route path="songs/:songId" element={<SongsTab />} />
          <Route path="discover" element={<DiscoverTab />} />
          <Route path="create" element={<CreateTab />} />
        </Route>

        {/* Game flow — no menu chrome */}
        <Route path="/game" element={<GameFlow />} />
      </Routes>
    </div>
  );
}

export default App;

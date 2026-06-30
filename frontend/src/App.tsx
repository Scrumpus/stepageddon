/**
 * App — top-level route definitions.
 *
 * /playlists | /songs | /discover | /create → MenuScreen layout + tab content
 * /game                                      → GameFlow (Zustand-driven screens)
 */

import { Navigate, Route, Routes, useNavigate } from 'react-router-dom';
import { Plus } from 'lucide-react';
import MenuScreen from '@/features/menu/components/MenuScreen';
import CreateTab from '@/features/menu/components/CreateTab';
import GameFlow from '@/features/menu/components/GameFlow';
import SettingsModal from '@/features/menu/components/SettingsModal';
import { PlaylistsTab } from '@/features/playlists/components';
import { SongsTab } from '@/features/songs/components';
import { DiscoverTab } from '@/features/discover/components';

function App() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-game-bg">
      {/* Top-right actions — visible on every page */}
      <div className="fixed top-4 right-4 z-50 flex items-center gap-2">
        <button
          onClick={() => navigate('/create')}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-game-primary text-game-bg hover:brightness-110 transition-all font-semibold shadow-lg shadow-game-primary/25"
          aria-label="Create chart"
        >
          <Plus className="w-5 h-5" />
          <span>Create</span>
        </button>
        <SettingsModal />
      </div>

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

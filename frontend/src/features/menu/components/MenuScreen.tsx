/**
 * MenuScreen — layout shell with tab navigation.
 * Renders <Outlet /> for the active route content.
 */

import { Compass, Disc3, ListMusic, Sparkles } from 'lucide-react';
import { NavLink, Outlet } from 'react-router-dom';
import SettingsModal from './SettingsModal';

function MenuScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      {/* Settings gear — fixed top-right */}
      <SettingsModal />

      <div className="max-w-6xl w-full">
        {/* Title */}
        <div className="text-center mb-4">
          <h1 className="text-6xl font-bold mb-4 text-game-primary">
            
          </h1>
        </div>

        {/* Tabs — NavLink replaces useState + onClick */}
        <div className="flex gap-1 mb-4 bg-white/5 p-1 rounded-xl border border-white/10 w-fit mx-auto">
          <TabLink to="/playlists" icon={<ListMusic className="w-4 h-4" />} label="Playlists" />
          <TabLink to="/songs" icon={<Disc3 className="w-4 h-4" />} label="Songs" />
          <TabLink to="/discover" icon={<Compass className="w-4 h-4" />} label="Discover" />
          <TabLink to="/create" icon={<Sparkles className="w-4 h-4" />} label="Create" />
        </div>

        {/* Main Card — renders the active route */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 sm:p-8 shadow-2xl border border-white/10">
          <Outlet />
        </div>

        {/* Credit */}
        <p className="text-center text-xs text-white/40 mt-6">
          Created by{' '}
          <a
            href="mailto:scottdschwalbe@gmail.com"
            className="hover:text-white/70 transition-colors"
          >
            scottdschwalbe@gmail.com
          </a>
        </p>
      </div>
    </div>
  );
}

interface TabLinkProps {
  to: string;
  icon: React.ReactNode;
  label: string;
}

function TabLink({ to, icon, label }: TabLinkProps) {
  return (
    <NavLink
      to={to}
      end={to === '/playlists'}
      className={({ isActive }) =>
        `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
          isActive
            ? 'bg-game-primary text-game-bg'
            : 'text-white/70 hover:text-white hover:bg-white/5'
        }`
      }
    >
      {icon}
      {label}
    </NavLink>
  );
}

export default MenuScreen;

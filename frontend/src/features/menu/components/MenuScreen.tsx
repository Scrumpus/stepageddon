/**
 * Main menu screen
 */

import { useState } from 'react';
import { Disc3, ListMusic, Sparkles } from 'lucide-react';
import { useApp } from '@/app/providers/AppProvider';
import { useStepGeneration } from '../hooks';
import FileUploader from './FileUploader';
import UrlInput from './UrlInput';
import ModeSelector from './ModeSelector';
import { PlaylistsTab } from '@/features/playlists/components';
import { SongsTab } from '@/features/songs/components';

type Tab = 'playlists' | 'songs' | 'create';

function MenuScreen() {
  const { gameMode, setGameMode } = useApp();
  const { handleFileUpload, handleUrlSubmit } = useStepGeneration();
  const [tab, setTab] = useState<Tab>('playlists');

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Title */}
        <div className="text-center mb-4">
          <h1 className="text-6xl font-bold mb-4 text-game-primary">
            
          </h1>
        </div>

        {/* Mode toggle (applies to all tabs) */}
        <div className="flex justify-center mb-6">
          <ModeSelector mode={gameMode} onModeChange={setGameMode} />
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-4 bg-white/5 p-1 rounded-xl border border-white/10 w-fit mx-auto">
          <TabButton
            active={tab === 'playlists'}
            onClick={() => setTab('playlists')}
            icon={<ListMusic className="w-4 h-4" />}
            label="Playlists"
          />
          <TabButton
            active={tab === 'songs'}
            onClick={() => setTab('songs')}
            icon={<Disc3 className="w-4 h-4" />}
            label="Songs"
          />
          <TabButton
            active={tab === 'create'}
            onClick={() => setTab('create')}
            icon={<Sparkles className="w-4 h-4" />}
            label="Create"
          />
        </div>

        {/* Main Card */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-6 sm:p-8 shadow-2xl border border-white/10">
          {tab === 'playlists' && <PlaylistsTab />}
          {tab === 'songs' && <SongsTab />}
          {tab === 'create' && (
            <div className="space-y-6">
              <p className="text-sm text-gray-400 text-center">
                We'll generate charts for every difficulty — you pick which to play next.
              </p>

              <div className="space-y-4">
                <UrlInput source="youtube" onUrlSubmit={handleUrlSubmit} />
                <UrlInput source="spotify" onUrlSubmit={handleUrlSubmit} />
              </div>

              <div className="flex items-center gap-4">
                <div className="flex-1 border-t border-white/10" />
                <span className="text-xs text-gray-500 uppercase tracking-wider">
                  or upload a file
                </span>
                <div className="flex-1 border-t border-white/10" />
              </div>

              <FileUploader onFileSelect={handleFileUpload} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}

function TabButton({ active, onClick, icon, label }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
        active
          ? 'bg-game-primary text-game-bg'
          : 'text-white/70 hover:text-white hover:bg-white/5'
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

export default MenuScreen;

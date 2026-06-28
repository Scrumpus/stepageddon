import { useState, useCallback } from 'react';
import { Compass, Disc3, Link2, ListMusic, Search, Sparkles, Upload } from 'lucide-react';
import { useGameStore } from '@/app/store/useGameStore';
import { useStepGeneration } from '../hooks';
import FileUploader from './FileUploader';
import UrlInput from './UrlInput';
import SongSearchInput from './SongSearchInput';
import ModeSelector from './ModeSelector';
import { PlaylistsTab } from '@/features/playlists/components';
import { SongsTab } from '@/features/songs/components';
import { DiscoverTab } from '@/features/discover/components';

type Tab = 'playlists' | 'songs' | 'discover' | 'create';
type CreateTab = 'search' | 'upload' | 'url';

function MenuScreen() {
  const gameMode = useGameStore((s) => s.gameMode);
  const setGameMode = useGameStore((s) => s.setGameMode);
  const { handleFileUpload, handleUrlSubmit } = useStepGeneration();
  const [tab, setTab] = useState<Tab>('playlists');
  const [createTab, setCreateTab] = useState<CreateTab>('search');
  const [chartStyle, setChartStyle] = useState<string>('auto');

  const handleFileUploadStyled = useCallback(
    (file: File) => handleFileUpload(file, chartStyle),
    [handleFileUpload, chartStyle],
  );
  const handleUrlSubmitStyled = useCallback(
    (url: string) => handleUrlSubmit(url, chartStyle),
    [handleUrlSubmit, chartStyle],
  );

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="max-w-6xl w-full">
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
            active={tab === 'discover'}
            onClick={() => setTab('discover')}
            icon={<Compass className="w-4 h-4" />}
            label="Discover"
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
          <div className={tab === 'playlists' ? '' : 'hidden'}>
            <PlaylistsTab />
          </div>
          <div className={tab === 'songs' ? '' : 'hidden'}>
            <SongsTab />
          </div>
          <div className={tab === 'discover' ? '' : 'hidden'}>
            <DiscoverTab onSelect={handleUrlSubmit} />
          </div>
          <div className={tab === 'create' ? 'space-y-6' : 'hidden'}>
            <div className="flex gap-1 bg-white/5 p-1 rounded-xl border border-white/10 w-fit mx-auto">
              <TabButton
                active={createTab === 'search'}
                onClick={() => setCreateTab('search')}
                icon={<Search className="w-4 h-4" />}
                label="Search"
              />
              <TabButton
                active={createTab === 'upload'}
                onClick={() => setCreateTab('upload')}
                icon={<Upload className="w-4 h-4" />}
                label="Upload"
              />
              <TabButton
                active={createTab === 'url'}
                onClick={() => setCreateTab('url')}
                icon={<Link2 className="w-4 h-4" />}
                label="URL"
              />
            </div>

            {/* Style selector */}
            <div className="flex items-center gap-3 justify-center">
              <label className="text-sm text-gray-400">Chart style:</label>
              <select
                value={chartStyle}
                onChange={(e) => setChartStyle(e.target.value)}
                className="px-3 py-1.5 bg-white/5 border border-white/20 rounded-lg text-sm text-white focus:outline-none focus:border-game-accent"
              >
                <option value="auto">Auto (match song)</option>
                <option value="Flowing">Flowing</option>
                <option value="Stream-Heavy">Stream-Heavy</option>
                <option value="Jump-Heavy">Jump-Heavy</option>
                <option value="Hold-Heavy">Hold-Heavy</option>
                <option value="Technical">Technical</option>
                <option value="Dense">Dense</option>
              </select>
            </div>

            <div className={createTab === 'search' ? '' : 'hidden'}>
              <SongSearchInput onSelect={handleUrlSubmitStyled} />
            </div>

            <div className={createTab === 'upload' ? '' : 'hidden'}>
              <FileUploader onFileSelect={handleFileUploadStyled} />
            </div>

            <div className={createTab === 'url' ? 'space-y-4' : 'hidden'}>
              <UrlInput source="audius" onUrlSubmit={handleUrlSubmitStyled} />
              <UrlInput source="jamendo" onUrlSubmit={handleUrlSubmitStyled} />
            </div>
          </div>
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

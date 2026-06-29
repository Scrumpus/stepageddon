/**
 * Create tab — sub-tabs for search / upload / URL-based step generation.
 * Extracted from MenuScreen so each tab owns its own component.
 */

import { useState } from 'react';
import { Link2, Search, Upload } from 'lucide-react';
import { useStepGeneration } from '../hooks';
import FileUploader from './FileUploader';
import UrlInput from './UrlInput';
import SongSearchInput from './SongSearchInput';

type CreateSubTab = 'search' | 'upload' | 'url';

function CreateTab() {
  const { handleFileUpload, handleUrlSubmit } = useStepGeneration();
  const [createTab, setCreateTab] = useState<CreateSubTab>('search');

  return (
    <div className="space-y-6">
      {/* Sub-tabs */}
      <div className="flex gap-1 bg-white/5 p-1 rounded-xl border border-white/10 w-fit mx-auto">
        <SubTabButton
          active={createTab === 'search'}
          onClick={() => setCreateTab('search')}
          icon={<Search className="w-4 h-4" />}
          label="Search"
        />
        <SubTabButton
          active={createTab === 'upload'}
          onClick={() => setCreateTab('upload')}
          icon={<Upload className="w-4 h-4" />}
          label="Upload"
        />
        <SubTabButton
          active={createTab === 'url'}
          onClick={() => setCreateTab('url')}
          icon={<Link2 className="w-4 h-4" />}
          label="URL"
        />
      </div>

      <div className={createTab === 'search' ? '' : 'hidden'}>
        <SongSearchInput onSelect={handleUrlSubmit} />
      </div>

      <div className={createTab === 'upload' ? '' : 'hidden'}>
        <FileUploader onFileSelect={handleFileUpload} />
      </div>

      <div className={createTab === 'url' ? 'space-y-4' : 'hidden'}>
        <UrlInput source="audius" onUrlSubmit={handleUrlSubmit} />
        <UrlInput source="jamendo" onUrlSubmit={handleUrlSubmit} />
      </div>
    </div>
  );
}

interface SubTabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}

function SubTabButton({ active, onClick, icon, label }: SubTabButtonProps) {
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

export default CreateTab;

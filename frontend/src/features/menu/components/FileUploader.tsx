/**
 * File upload component
 */

import { useState } from 'react';
import { Upload } from 'lucide-react';

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
}

const ACCEPTED_EXTENSIONS = ['.mp3', '.wav', '.ogg', '.flac'];

function isAcceptedAudioFile(file: File): boolean {
  if (file.type.startsWith('audio/')) return true;
  const name = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext));
}

function FileUploader({ onFileSelect }: FileUploaderProps) {
  const [isDragActive, setIsDragActive] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'copy';
    if (!isDragActive) setIsDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file && isAcceptedAudioFile(file)) {
      onFileSelect(file);
    }
  };

  return (
    <div>
      <label
        className="block cursor-pointer"
        onDragEnter={handleDragOver}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div
          className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
            isDragActive
              ? 'border-game-accent bg-game-accent/10'
              : 'border-white/30 hover:border-game-accent hover:bg-game-accent/5'
          }`}
        >
          <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p className="text-lg mb-2">Drop your audio file here</p>
          <p className="text-sm text-gray-400">
            or click to browse
          </p>
          <p className="text-xs text-gray-500 mt-2">
            Supported: MP3, WAV, OGG, FLAC (Max 50MB, 10 min)
          </p>
        </div>
        <input
          type="file"
          accept=".mp3,.wav,.ogg,.flac"
          onChange={handleFileChange}
          className="hidden"
        />
      </label>
    </div>
  );
}

export default FileUploader;

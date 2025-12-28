/**
 * File upload component
 */

import { Upload } from 'lucide-react';

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
}

export function FileUploader({ onFileSelect }: FileUploaderProps) {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <div>
      <label className="block cursor-pointer">
        <div className="border-2 border-dashed border-white/30 rounded-lg p-12 text-center hover:border-game-accent hover:bg-game-accent/5 transition-all">
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

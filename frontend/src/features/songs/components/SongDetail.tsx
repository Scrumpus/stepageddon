import { ChartSelect } from '@/features/playlists/components';
import { usePlayChart } from '@/features/playlists/hooks';
import { SongSummaryDTO } from '@/features/playlists/types';
import { useSongDetail } from '../hooks';

interface Props {
  songId: string;
  /** Summary already in hand from the list — header renders instantly. */
  fallbackSong: SongSummaryDTO;
  onBack: () => void;
}

function SongDetail({ songId, fallbackSong, onBack }: Props) {
  const { song, charts, isLoading, error } = useSongDetail(songId);
  const { playChart, isLoading: isPlaying } = usePlayChart();

  return (
    <ChartSelect
      song={song ?? fallbackSong}
      charts={charts}
      onSelectChart={playChart}
      onBack={onBack}
      disabled={isPlaying}
      isLoading={isLoading}
      error={error}
    />
  );
}

export default SongDetail;

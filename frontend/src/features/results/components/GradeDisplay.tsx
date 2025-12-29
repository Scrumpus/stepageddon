/**
 * Grade display - shows final letter grade
 */

import { Grade, GRADE_COLORS } from '../types/results.types';

interface GradeDisplayProps {
  grade: Grade;
  songTitle: string;
}

function GradeDisplay({ grade, songTitle }: GradeDisplayProps) {
  return (
    <div className="text-center mb-8">
      <div className={`text-8xl font-bold mb-4 ${GRADE_COLORS[grade]}`}>
        {grade}
      </div>
      <h2 className="text-2xl font-semibold mb-2">Song Complete!</h2>
      <p className="text-gray-400">{songTitle}</p>
    </div>
  );
}

export default GradeDisplay;

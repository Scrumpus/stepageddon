import { useState, useRef, useEffect } from 'react';
import MenuScreen from './screens/MenuScreen';
import LoadingScreen from './screens/LoadingScreen';
import ReadyScreen from './screens/ReadyScreen';
import GameScreen from './screens/GameScreen';
import ResultsScreen from './screens/ResultsScreen';
import { GAME_STATES } from './utils/gameConstants';
import { generateStepsFromFile, generateStepsFromUrl, getAudioUrl } from './config/api';

function App() {
  // Game state
  const [gameState, setGameState] = useState(GAME_STATES.MENU);
  const [difficulty, setDifficulty] = useState('intermediate');
  
  // Audio and steps data
  const [songData, setSongData] = useState(null);
  const [steps, setSteps] = useState([]);
  const [audioUrl, setAudioUrl] = useState(null);
  
  // Loading state
  const [loadingMessage, setLoadingMessage] = useState('');
  const [loadingProgress, setLoadingProgress] = useState(0);
  
  // Game results
  const [gameResults, setGameResults] = useState(null);
  
  // Audio element reference
  const audioRef = useRef(null);
  
  // Handle file upload
  const handleFileUpload = async (file) => {
    try {
      setGameState(GAME_STATES.LOADING);
      setLoadingMessage('Uploading audio...');
      setLoadingProgress(25);
      
      const result = await generateStepsFromFile(file, difficulty);
      
      setLoadingProgress(100);
      setLoadingMessage('Generation complete!');
      
      // Set data
      setSongData(result.song_info);
      setSteps(result.steps);
      setAudioUrl(getAudioUrl(result.audio_url));
      
      // Move to ready screen
      setTimeout(() => {
        setGameState(GAME_STATES.READY);
      }, 500);
      
    } catch (error) {
      console.error('Upload failed:', error);
      alert(`Failed to generate steps: ${error.response?.data?.detail || error.message}`);
      setGameState(GAME_STATES.MENU);
    }
  };
  
  // Handle URL submission
  const handleUrlSubmit = async (url) => {
    try {
      setGameState(GAME_STATES.LOADING);
      setLoadingMessage('Downloading audio...');
      setLoadingProgress(20);
      
      setTimeout(() => {
        setLoadingMessage('Analyzing music...');
        setLoadingProgress(50);
      }, 1000);
      
      const result = await generateStepsFromUrl(url, difficulty);
      
      setLoadingProgress(100);
      setLoadingMessage('Generation complete!');
      
      // Set data
      setSongData(result.song_info);
      setSteps(result.steps);
      setAudioUrl(getAudioUrl(result.audio_url));
      
      // Move to ready screen
      setTimeout(() => {
        setGameState(GAME_STATES.READY);
      }, 500);
      
    } catch (error) {
      console.error('URL processing failed:', error);
      alert(`Failed to generate steps: ${error.response?.data?.detail || error.message}`);
      setGameState(GAME_STATES.MENU);
    }
  };
  
  // Start game
  const handleStartGame = () => {
    setGameState(GAME_STATES.PLAYING);
  };
  
  // Game finished
  const handleGameFinish = (results) => {
    setGameResults(results);
    setGameState(GAME_STATES.FINISHED);
  };
  
  // Play again
  const handlePlayAgain = () => {
    setGameState(GAME_STATES.READY);
    setGameResults(null);
  };
  
  // Return to menu
  const handleReturnToMenu = () => {
    setGameState(GAME_STATES.MENU);
    setSongData(null);
    setSteps([]);
    setAudioUrl(null);
    setGameResults(null);
    
    // Stop audio if playing
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-game-bg via-purple-900/20 to-game-bg">
      {/* Audio element */}
      {audioUrl && (
        <audio ref={audioRef} src={audioUrl} preload="auto" />
      )}
      
      {/* Render current screen */}
      {gameState === GAME_STATES.MENU && (
        <MenuScreen
          difficulty={difficulty}
          onDifficultyChange={setDifficulty}
          onFileUpload={handleFileUpload}
          onUrlSubmit={handleUrlSubmit}
        />
      )}
      
      {gameState === GAME_STATES.LOADING && (
        <LoadingScreen
          message={loadingMessage}
          progress={loadingProgress}
        />
      )}
      
      {gameState === GAME_STATES.READY && (
        <ReadyScreen
          songInfo={songData}
          difficulty={difficulty}
          onStart={handleStartGame}
          onBack={handleReturnToMenu}
        />
      )}
      
      {(gameState === GAME_STATES.PLAYING || gameState === GAME_STATES.PAUSED) && (
        <GameScreen
          steps={steps}
          audioRef={audioRef}
          songInfo={songData}
          difficulty={difficulty}
          onFinish={handleGameFinish}
          onReturnToMenu={handleReturnToMenu}
        />
      )}
      
      {gameState === GAME_STATES.FINISHED && (
        <ResultsScreen
          results={gameResults}
          songInfo={songData}
          onPlayAgain={handlePlayAgain}
          onReturnToMenu={handleReturnToMenu}
        />
      )}
    </div>
  );
}

export default App;

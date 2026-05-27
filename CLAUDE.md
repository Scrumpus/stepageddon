# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Beat Sync is a DDR-style rhythm game that generates custom step charts from audio. It uses:
- **Frontend**: React 18 + Vite + TailwindCSS
- **Backend**: FastAPI (Python) with librosa for audio analysis
- **Audio Processing**: librosa 22050 Hz sample rate, comprehensive feature extraction

## Development Commands

### Backend (FastAPI)

```bash
cd backend

# Setup (first time)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
cp .env.example .env

# Run development server (with auto-reload)
python main.py
# Backend runs at http://localhost:8000

# API docs available at:
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### Frontend (React + Vite)

```bash
cd frontend

# Setup (first time)
npm install

# Run development server
npm run dev
# Frontend runs at http://localhost:3000
# Vite proxies /api/* to backend at localhost:8000

# Build for production
npm run build

# Lint
npm run lint
```

### Docker

```bash
# Start both services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Architecture

### Backend Architecture

The backend has a three-layer step generation architecture:

**Layer 1: Audio Analysis** (`backend/services/audio_processor.py`)
- Loads audio at 22050 Hz using librosa
- **Beat Detection**: Detects tempo and beat frames via `librosa.beat.beat_track()`
- **Onset Detection**: Finds percussive transients with backtracking for precision
- **Energy Analysis**: RMS energy profile, normalized and downsampled
- **Spectral Features**: Extracts brightness (spectral centroid), rolloff, zero-crossing rate, MFCC
- **Genre Inference**: Basic heuristic based on brightness and ZCR (electronic/rock vs chill/ambient)
- **Structure Detection**: Uses chroma features and self-similarity matrix to find song sections

Output: Comprehensive analysis dict with beat_times, onset_times, energy_profile, spectral_features, structure boundaries

**Layer 2: Step Generation** — ML model

Chart generation is handled by a trained PyTorch model loaded at app startup.

- **Entry point**: `ml.inference.MLChartGenerator.generate_from_audio(audio_path, difficulty)` → `Chart`
- **Module**: `backend/ml/` (model in `model.py`, inference in `inference.py`, training in `train.py`)
- **Checkpoint path**: `settings.ML_MODEL_PATH` (defaults to `./ml/checkpoints/best_model.pt`)
- **Schemas**: `backend/modules/step_generator/` now contains only the shared Pydantic schemas (`Chart`, `Step`, `Direction`, `StepType`, `BeatSubdivision`, `DifficultyConfig`) and the difficulty presets (`DIFFICULTY_PRESETS`, `get_difficulty_config`) consumed by the ML inference path, persistence, and the `.sm` parser.
- **JSON serialization**: `Chart.to_json_dict()` returns the API response shape.

If the checkpoint can't be loaded, app startup fails fast — there is no algorithmic fallback.

**Layer 3: API Routers** (`backend/routers/`):
- `generation.py` - POST endpoints for chart generation (file upload or URL)
- `audio.py` - GET endpoint for streaming stored audio files

**Complete Flow**:
```
Upload/URL → Router validates → download_audio (Audius/Jamendo, if URL) → AudioProcessor analyzes
→ MLChartGenerator predicts chart → Chart.to_json_dict() → Response with steps + audio URL
```

### Frontend Architecture

**State Management** (`frontend/src/App.jsx`):
- Centralized game state machine with states: MENU → LOADING → READY → PLAYING → RESULTS
- State flows through props to screen components
- Audio element managed via `audioRef` and passed to GameScreen

**Screens** (`frontend/src/screens/`):
- `MenuScreen.jsx` - Difficulty selection, file upload, URL input
- `LoadingScreen.jsx` - Progress indicator during generation
- `ReadyScreen.jsx` - Countdown before gameplay starts
- `GameScreen.jsx` - Main gameplay loop (60 FPS, hit detection, scoring)
- `ResultsScreen.jsx` - Score breakdown and accuracy stats

**Game Loop Architecture** (`GameScreen.jsx`):

The game runs at 60 FPS using `requestAnimationFrame`:

```javascript
const gameLoop = () => {
  const currentTime = audioRef.current.currentTime;

  // 2-second visible window for arrows
  const visibleWindow = 2;

  steps.forEach((step) => {
    const timeUntilHit = step.time - currentTime;

    // Check for misses (passed by >200ms without hit)
    if (timeUntilHit < -0.2 && !step.hit) {
      handleMiss();
    }

    // Calculate arrow Y position based on time
    const y = HIT_ZONE_Y - (timeUntilHit * ARROW_SPEED);

    // Show if within visible window
    if (timeUntilHit >= -0.2 && timeUntilHit <= visibleWindow) {
      // Render arrow at calculated position
    }
  });

  animationRef.current = requestAnimationFrame(gameLoop);
};
```

**Gameplay Constants** (`frontend/src/utils/gameConstants.js`):

```javascript
// Timing windows (milliseconds)
TIMING = {
  PERFECT: 50,   // ±50ms
  GOOD: 100,     // ±100ms
  OK: 150,       // ±150ms
  MISS: 200      // Beyond 150ms
}

// Scoring
POINTS = {
  PERFECT: 100,
  GOOD: 50,
  OK: 25,
  MISS: 0
}

// Combo multipliers (at specific combo thresholds)
COMBO_MULTIPLIER = {
  10: 1.1,    // 10% bonus at 10 combo
  25: 1.25,   // 25% bonus at 25 combo
  50: 1.5,    // 50% bonus at 50 combo
  100: 2.0    // 2x bonus at 100 combo
}

// Visual settings
ARROW_SPEED = 400  // pixels per second
HIT_ZONE_Y = 600   // Hit zone distance from top
ARROW_SIZE = 80    // Arrow dimensions in pixels
```

**Hit Detection** (`frontend/src/utils/scoring.js`):
- `evaluateHit(timeDiff)` - Returns judgment string based on timing window
- `calculatePoints(judgment, combo)` - Applies base points + combo multiplier
- Timing differences calculated as `Math.abs(step.time - currentAudioTime)`

### API Contract

**Generate from File**:
```http
POST /api/generate-steps
Content-Type: multipart/form-data

file: <audio file>
difficulty: beginner|intermediate|expert
```

**Generate from URL**:
```http
POST /api/generate-steps-url
Content-Type: application/json

{
  "url": "https://audius.co/artist/track-name",
  "difficulty": "intermediate"
}
```

**Response Format**:
```json
{
  "song_id": "uuid",
  "steps": [
    {"time": 1.875, "arrows": ["left"]},
    {"time": 2.250, "arrows": ["up"]},
    {"time": 2.625, "arrows": ["down", "right"]}
  ],
  "song_info": {
    "title": "Song Title",
    "duration": 180.5,
    "tempo": 128
  },
  "audio_url": "/api/audio/{song_id}.mp3"
}
```

## Configuration

### Environment Variables

**Backend** (`.env` in `backend/`):
```env
# Path to the trained step-chart model checkpoint
ML_MODEL_PATH=./ml/checkpoints/best_model.pt

# Audio sources
# Audius needs no key — just an app identifier sent on every request.
AUDIUS_APP_NAME=stepageddon
# Jamendo needs a free client_id (devportal.jamendo.com). Empty disables
# Jamendo search/links gracefully.
JAMENDO_CLIENT_ID=...

# Server
HOST=0.0.0.0
PORT=8000

# Limits
MAX_FILE_SIZE_MB=50
MAX_DURATION_SECONDS=600
```

**Frontend** (`.env` in `frontend/`):
```env
VITE_API_URL=http://localhost:8000
```

## Common Development Tasks

### Modifying Step Generation Logic

Generation lives in `backend/ml/`. The model is trained offline; the API only runs inference.

- **Inference entry point**: `ml.inference.MLChartGenerator.generate_from_audio(audio_path, difficulty)` returns a `Chart`.
- **Schemas / difficulty presets**: `backend/modules/step_generator/{schemas,difficulty}.py` — shared by the ML inference path, `services/chart_persistence.py`, and `services/sm_parser.py`.
- **Training**: `backend/ml/train.py`, `prepare_data.py`, `dataset.py`, `model.py`. Checkpoints land in `backend/ml/checkpoints/`.
- **Behavior tuning at inference time**: see `MLChartGenerator.__init__` knobs (`chunk_frames`, `overlap_frames`, `confidence_threshold`, `min_note_gap`, `snap_to_beats`).

```python
from ml import MLChartGenerator
gen = MLChartGenerator(model_path="./ml/checkpoints/best_model.pt")
chart = gen.generate_from_audio("song.mp3", "medium")
response = chart.to_json_dict()
```

**Difficulty Presets**: `backend/modules/step_generator/difficulty.py` (`DIFFICULTY_PRESETS`, `get_difficulty_config`). Used by ML inference for difficulty conditioning; the algorithmic density/grid fields remain on `DifficultyConfig` for compatibility but are no longer interpreted by a placement algorithm.

### Adjusting Timing Windows

To modify hit detection timing in the frontend:

1. Edit `frontend/src/utils/gameConstants.js`
2. Modify `TIMING` object values (in milliseconds)
3. Timing is ± the value (e.g., PERFECT: 50 means ±50ms = 100ms window)
4. Test with different BPM songs (faster songs need tighter windows)

### Debugging Step Generation

- **No notes / sparse output**: Check `confidence_threshold` and `min_note_gap` on `MLChartGenerator`.
- **Notes feel off-beat**: Verify `snap_to_beats=True` and inspect the librosa beat track on the input audio.
- **Server fails to start**: `ML_MODEL_PATH` likely points to a missing/incompatible checkpoint. There is no algorithmic fallback — fix the path or restore the file.

**Logging**: Run backend with `LOG_LEVEL=DEBUG` for verbose ML inference logs.

### Frontend State Flow

When adding new screens or modifying state transitions:

```javascript
// State flow
MENU (select difficulty, upload)
  ↓ call generateSteps API
LOADING (show progress)
  ↓ on completion
READY (3-2-1 countdown)
  ↓ on countdown finish
PLAYING (game loop active)
  ↓ on song end
FINISHED/RESULTS (show score breakdown)
```

State changes via `setGameState()` in `App.jsx`. All screens receive state via props.

## Development Notes

- **Vite proxy**: Frontend dev server proxies `/api/*` to backend at localhost:8000 (see `frontend/vite.config.js`)
- **CORS**: Backend allows `localhost:3000` and `localhost:5173` in development (see `core/config.py`)
- **Audio storage**: Files saved to `backend/audio_storage/` with UUID filenames (never deleted automatically)
- **File limits**: 50MB max file size, 600s (10 min) max duration
- **Supported formats**:
  - Upload: MP3, WAV, OGG, FLAC
  - URLs: Audius (full tracks, no key), Jamendo (full tracks, free `client_id`)
- **Generation**: ML-only via `MLChartGenerator`; checkpoint loaded at startup from `ML_MODEL_PATH`
- **Sample rate**: All audio loaded at 22050 Hz for consistency

## Troubleshooting

### Backend Issues

**"Tempo detection failed"**
- Fallback to 120 BPM is applied
- Check if audio file is corrupted or has very low volume
- Songs with variable tempo may need manual tempo tracking

**"Step generation timeout"**
- Default timeout is 30s (see `settings.MAX_GENERATION_TIME`)
- Happens with very long songs or complex analysis
- Consider increasing timeout or implementing progress callbacks

**Audius/Jamendo download fails**
- Jamendo: confirm `JAMENDO_CLIENT_ID` is set (devportal.jamendo.com); empty disables Jamendo
- Audius: a track may be unstreamable/removed — try another result
- Check backend logs for the specific source error (resolve vs stream/download)

### Frontend Issues

**Arrows not syncing with audio**
- Audio element latency varies by browser/OS
- Check `audioRef.current.currentTime` vs expected time in console
- Possible fix: Add offset calibration in settings

**Performance drops during gameplay**
- Reduce visible window from 2s to 1.5s
- Limit number of arrows rendered (cull off-screen arrows)
- Check for memory leaks in game loop

**Game loop not stopping on pause**
- Ensure `cancelAnimationFrame(animationRef.current)` is called
- Check `gameState` is properly set to PAUSED

## Testing the Game

1. Start backend: `cd backend && python main.py`
2. Start frontend: `cd frontend && npm run dev`
3. Upload a song (try different genres to test pattern variety)
4. Select each difficulty to verify different patterns
5. During gameplay:
   - Arrow keys: ←↓↑→
   - ESC: Pause/Resume
   - Check browser console for timing info
6. Verify results screen shows accurate stats

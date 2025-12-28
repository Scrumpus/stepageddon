# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Beat Sync is a DDR-style rhythm game that generates custom step charts from audio. It uses:
- **Frontend**: React 18 + Vite + TailwindCSS
- **Backend**: FastAPI (Python) with librosa for audio analysis
- **Audio Processing**: librosa 22050 Hz sample rate, comprehensive feature extraction
- **Two Generation Modes**: Pure algorithmic (default) or AI-enhanced (optional, requires Anthropic API key)

## Development Commands

### Backend (FastAPI)

```bash
cd backend

# Setup (first time)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
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

**Layer 2: Step Generation** (THREE generators coexist):

1. **`step_generator_new.py`** - **RECOMMENDED** - New deterministic engine
   - Implements the STEP_ENGINE.md contract
   - Grid-locked timing (quarter/eighth/sixteenth notes by difficulty)
   - Deterministic seeding from audio hash + difficulty
   - Supports TAP and HOLD note types
   - Uses dataclasses: `Beat`, `Step`, `Chart`, `Direction`, `StepType`
   - Difficulty presets with precise configs (see Difficulty System section)
   - Called via `ChartGenerationPipeline.generate_from_audio()`

2. **`algorithmic_generator.py`** - Enhanced algorithmic generator
   - Pure librosa-based, no AI dependency
   - Energy-aware pattern generation
   - Supports doubles (two-arrow steps) on higher difficulties
   - Foot logic to avoid awkward patterns (opposites, adjacents)
   - Used by `step_generator.py` wrapper

3. **`step_generator.py`** - Legacy wrapper (being phased out)
   - Simple wrapper around `algorithmic_generator.py`
   - Has `use_ai` parameter (currently unused)
   - Basic validation and refinement (deduplication, min gap enforcement)

**Current Status**: `routers/generation.py` calls BOTH `step_generator` (legacy) and `step_generator_new` to allow comparison during transition. The response includes both `steps` (legacy) and `new_steps_json` (new format).

**Layer 3: API Routers** (`backend/routers/`):
- `generation.py` - POST endpoints for chart generation (file upload or URL)
- `audio.py` - GET endpoint for streaming stored audio files

**Complete Flow**:
```
Upload/URL → Router validates → AudioDownloader (if URL) → AudioProcessor analyzes
→ StepGenerator creates chart → Response with steps + audio URL
```

### Step Generation Engine

The step generator follows strict design principles documented in `backend/STEP_ENGINE.md`:

1. **Grid-Locked Timing**: All steps align to musical grid (quarter/eighth/sixteenth notes based on difficulty)
2. **Determinism**: Same audio + difficulty = identical chart every time
3. **Playability First**: Readable charts preferred over dense ones
4. **Foot Logic**: Arrow selection uses cost function to avoid awkward patterns (consecutive same arrows, excessive same-foot usage, crossovers)

**Difficulty System** (from `step_generator_new.py`):

The new generator uses detailed `DifficultyConfig` dataclasses with precise parameters:

- **Beginner**:
  - Density: 0.6-1.2 steps/second
  - Grid: Downbeats only (quarter notes)
  - Singles only, no jumps
  - 15% holds (0.8-2.0s duration)
  - No crossovers, no brackets
  - Energy scale: 0.3x

- **Intermediate**:
  - Density: 1.3-2.3 steps/second
  - Grid: Downbeats + upbeats + offbeats (eighth notes)
  - Singles + doubles
  - 20% holds (0.6-3.0s duration)
  - Max 2 consecutive jumps, max 6-note streams
  - Crossovers allowed
  - Energy scale: 0.6x

- **Expert**:
  - Density: 2.2-4.0 steps/second
  - Grid: All subdivisions (sixteenth notes)
  - Singles + doubles + brackets
  - 25% holds (0.5-4.0s duration)
  - Max 4 consecutive jumps, max 16-note streams
  - Crossovers + brackets allowed
  - Energy scale: 1.0x

**Data Structures** (`step_generator_new.py`):
```python
@dataclass
class Beat:
    time: float
    strength: float
    beat_type: str  # 'downbeat', 'upbeat', 'offbeat'
    measure_position: int  # 0-3 for 4/4 time
    is_strong: bool

@dataclass
class Step:
    time: float
    arrows: List[Direction]  # Direction.LEFT/DOWN/UP/RIGHT
    step_type: StepType  # StepType.TAP or StepType.HOLD
    hold_duration: Optional[float]  # Required for HOLD, None for TAP

@dataclass
class Chart:
    steps: List[Step]
    difficulty: str
    tempo: float
    duration: float
```

**Algorithm Overview**:
1. `analyze_beats()` - Classify all beats as downbeat/upbeat/offbeat in 4/4 time
2. `detect_subdivisions()` - Find eighth/sixteenth note positions based on onsets
3. `analyze_energy_sections()` - Segment song into low/medium/high/climax intensity
4. `detect_sustained_notes()` - Find melodic holds using pitch tracking
5. Place steps on grid with energy-aware density
6. Apply foot logic and cost function for arrow selection
7. Insert holds on sustained notes (15-25% of steps)
8. Export to JSON via `ChartExporter.to_json()`

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
  "url": "https://youtube.com/watch?v=...",
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
# Required for AI mode only (optional)
ANTHROPIC_API_KEY=sk-ant-...
USE_AI_GENERATION=false  # Set to true for AI mode

# Optional - for Spotify support
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...

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

When working with the step generator:

1. **Read STEP_ENGINE.md first** - Contains the complete design contract and requirements
2. **Primary file**: `backend/services/step_generator_new.py` (800+ lines)
3. **Test across difficulties** - Each has vastly different grid subdivision and density limits
4. **Verify determinism** - Use same audio file + difficulty multiple times, compare outputs
5. **Check playability** - Patterns must be physically feasible (avoid same arrow spam, impossible crossovers)

**Key Entry Points**:
```python
# Main pipeline
ChartGenerationPipeline.generate_from_audio(audio_path, difficulty) -> Chart

# Core analysis functions
analyze_beats(y, sr) -> Tuple[List[Beat], float]
detect_subdivisions(y, sr, beat_times) -> List[float]
analyze_energy_sections(y, sr) -> List[EnergySection]
detect_sustained_notes(y, sr) -> List[SustainedNote]

# Export
ChartExporter.to_json(chart) -> dict
```

**Difficulty Presets**: Located in `DIFFICULTY_PRESETS` dict at top of `step_generator_new.py`
- Modify density ranges, hold percentages, grid subdivisions
- Add new difficulty: Create new `DifficultyConfig` instance

### Adjusting Timing Windows

To modify hit detection timing in the frontend:

1. Edit `frontend/src/utils/gameConstants.js`
2. Modify `TIMING` object values (in milliseconds)
3. Timing is ± the value (e.g., PERFECT: 50 means ±50ms = 100ms window)
4. Test with different BPM songs (faster songs need tighter windows)

### Adding New Audio Features

To extract additional features from audio:

1. Add extraction logic to `AudioProcessor._extract_spectral_features()` or create new method
2. Include in return dict from `analyze_audio()`
3. Access in step generator via `audio_analysis['your_feature']`
4. Use in placement decisions or pattern generation

### Debugging Step Generation

Common issues and solutions:

- **Steps feel off-beat**: Check `analyze_beats()` tempo detection. Songs with tempo changes or complex rhythms may need manual BPM hints.
- **Too many/few steps**: Adjust `min_density` / `max_density` in difficulty preset
- **Awkward patterns**: Review cost function in arrow selection logic
- **No holds appearing**: Check `hold_percentage` in preset, verify `detect_sustained_notes()` is finding melodic content

**Logging**: Step generator has extensive logging at INFO level. Run backend with `LOG_LEVEL=DEBUG` for verbose output.

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
  - URLs: YouTube (via yt-dlp), Spotify (30-second previews via spotipy)
- **Generation mode**: Set `USE_AI_GENERATION=true` in backend `.env` for AI mode (requires Anthropic API key)
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

**YouTube download fails**
- Update yt-dlp: `pip install --upgrade yt-dlp`
- Some videos may be geo-restricted or age-restricted
- Check backend logs for specific yt-dlp error

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

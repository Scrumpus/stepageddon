# Beat Sync - System Architecture

## Overview

Beat Sync is a full-stack web application that combines AI-powered music analysis with classic rhythm game mechanics. The system follows a three-tier architecture: Client (React SPA), API (FastAPI), and Processing (AI + Audio Analysis).

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         CLIENT TIER                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              React SPA (Port 3000)                   │  │
│  │  - Menu/Upload UI                                    │  │
│  │  - Game Engine (Arrow Animation)                     │  │
│  │  - Audio Player (Web Audio API)                      │  │
│  │  - Score Tracking & Display                          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                         API TIER                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           FastAPI Backend (Port 8000)                │  │
│  │  - /generate-steps endpoint                          │  │
│  │  - /audio/{id} streaming endpoint                    │  │
│  │  - Rate limiting & validation                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      PROCESSING TIER                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Audio Analysis Module                      │  │
│  │  - librosa beat detection                            │  │
│  │  - Spectral analysis (genre detection)               │  │
│  │  - Energy & structure analysis                       │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           AI Step Generation                         │  │
│  │  - Anthropic Claude API integration                  │  │
│  │  - Enhanced prompt engineering                       │  │
│  │  - Algorithmic fallback                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                         │
│  - Anthropic Claude API (step generation)                   │
│  - YouTube API / yt-dlp (audio extraction)                   │
│  - Spotify API (track metadata & preview URLs)              │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Frontend Components

#### Screens
- **MenuScreen**: Audio upload and difficulty selection
- **LoadingScreen**: Progress indicator during generation
- **ReadyScreen**: Song info and game preparation
- **GameScreen**: Main gameplay with arrow rendering and hit detection
- **ResultsScreen**: Final scores and statistics

#### Core Systems
- **Game Loop**: 60 FPS animation loop using `requestAnimationFrame`
- **Hit Detection**: Timing-based scoring with ±50-150ms windows
- **Audio Sync**: Synchronized audio playback with visual arrows
- **State Management**: React hooks for game state

### Backend Services

#### Audio Processing (`audio_processor.py`)
- Beat detection using librosa
- Tempo estimation with fallback
- Energy profile analysis
- Spectral feature extraction
- Song structure detection

#### Step Generation (`step_generator.py`)
- AI-powered generation using Claude Sonnet 4
- Enhanced prompt with musical context
- Difficulty-appropriate parameters
- Algorithmic fallback for reliability
- Step validation and refinement

#### Audio Downloader (`audio_downloader.py`)
- YouTube audio extraction with yt-dlp
- Spotify metadata and preview handling
- Format conversion to MP3
- Error handling and validation

### API Endpoints

#### POST `/api/generate-steps`
Upload audio file and generate steps.

**Request:**
```
Content-Type: multipart/form-data
file: <audio file>
difficulty: beginner|intermediate|expert
```

**Response:**
```json
{
  "song_id": "uuid",
  "steps": [
    {"time": 1.23, "direction": "left", "type": "single"},
    ...
  ],
  "song_info": {
    "title": "Song Name",
    "duration": 180.5,
    "tempo": 120
  },
  "audio_url": "/api/audio/uuid.mp3"
}
```

#### POST `/api/generate-steps-url`
Generate steps from YouTube/Spotify URL.

**Request:**
```json
{
  "url": "https://youtube.com/watch?v=...",
  "difficulty": "intermediate"
}
```

#### GET `/api/audio/{filename}`
Stream audio file with byte-range support.

## Data Flow

### File Upload Flow
1. User uploads audio file
2. Backend validates file (size, format, duration)
3. Audio saved to storage with unique ID
4. Audio analyzed with librosa
5. Analysis sent to Claude API with prompt
6. Steps generated and validated
7. Steps returned to frontend
8. Game initialized with steps and audio

### URL Flow
1. User pastes YouTube/Spotify URL
2. Backend validates URL
3. Audio downloaded/extracted
4. Same as steps 3-8 in file upload

### Gameplay Flow
1. Audio starts playing
2. Game loop runs at 60 FPS
3. Arrows rendered based on time
4. User presses arrow keys
5. Hit detection checks timing
6. Score calculated with combo multiplier
7. Visual feedback shown
8. Game ends when audio finishes
9. Results calculated and displayed

## Performance Optimizations

### Frontend
- `requestAnimationFrame` for smooth 60 FPS
- Efficient state updates
- Arrow culling (only render visible arrows)
- Audio element reuse
- CSS animations for visual effects

### Backend
- Async/await for I/O operations
- Audio analysis caching (future)
- Efficient librosa parameters
- Response streaming for large files
- Connection pooling

### AI Generation
- Concise prompts to reduce tokens
- Structured JSON output
- Validation and filtering
- Algorithmic fallback
- 30-second timeout

## Security Considerations

### Input Validation
- File type whitelist
- File size limits (50MB)
- Duration limits (10 minutes)
- URL format validation
- XSS prevention

### API Security
- CORS configuration
- Rate limiting (future)
- API key protection
- File cleanup
- Error message sanitization

## Scalability

### Current Architecture
- Single server deployment
- Local file storage
- Synchronous processing

### Future Improvements
- Redis for caching
- S3/cloud storage
- Queue-based processing (Celery)
- Horizontal scaling with load balancer
- CDN for static assets
- Database for user accounts

## Technology Stack

### Frontend
- React 18 (UI framework)
- Vite (build tool)
- TailwindCSS (styling)
- Axios (HTTP client)
- Lucide React (icons)

### Backend
- FastAPI (API framework)
- librosa (audio analysis)
- Anthropic SDK (AI integration)
- yt-dlp (YouTube support)
- spotipy (Spotify support)
- uvicorn (ASGI server)

### Infrastructure
- Docker (containerization)
- Nginx (reverse proxy)
- Docker Compose (orchestration)

## Monitoring & Logging

### Metrics to Track
- Generation success rate
- Average generation time
- API response times
- Error rates by type
- User engagement metrics

### Logging Strategy
- Structured logging (JSON)
- Log levels: DEBUG, INFO, WARNING, ERROR
- Rotating file handlers
- Centralized logging (future)

## Disaster Recovery

### Backup Strategy
- Configuration files in git
- Audio storage cleanup policy
- Database backups (future)
- Container registry

### Failure Handling
- Algorithmic fallback for AI
- Graceful error messages
- Audio cleanup on failure
- Health check endpoints

## Development Workflow

1. Local development with hot reload
2. Docker for integration testing
3. Git for version control
4. Docker Compose for deployment
5. CI/CD pipeline (future)

## Future Architecture Changes

### Phase 2: User Accounts
- PostgreSQL database
- JWT authentication
- User data models
- Chart storage

### Phase 3: Real-time Features
- WebSocket support
- Live multiplayer
- Real-time leaderboards

### Phase 4: Microservices
- Separate generation service
- Separate auth service
- API gateway
- Message queue

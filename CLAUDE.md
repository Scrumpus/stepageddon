# CLAUDE.md

Stepageddon — DDR-style rhythm game that generates custom step charts from audio.
**Stack**: FastAPI (Python) + React 18 (TypeScript, Vite, TailwindCSS, zustand).
**ML**: PyTorch model for step-chart generation.

## Project Layout

```
backend/
├── main.py                 # Entry point (runs uvicorn on src.main:app)
├── src/                    # FastAPI app — flat feature-module layout
│   ├── main.py             # App factory
│   ├── config.py           # Pydantic-settings config (env-based)
│   ├── database.py         # SQLAlchemy async engine
│   ├── audio/router.py     # Audio streaming endpoint
│   ├── audius/client.py    # Audius API client
│   ├── charts/             # Chart models, router, service, SM export
│   ├── discover/           # Genre-based discovery
│   ├── generation/         # Step generation router, schemas, service
│   │   └── utils.py        # AudioProcessor (librosa wrapper)
│   ├── jamendo/client.py   # Jamendo API client
│   ├── playlists/          # Playlist CRUD
│   ├── songs/              # Song models, router, SM parsing (utils.py)
│   ├── search/             # Search endpoints
│   ├── storage/client.py   # Local or S3 storage abstraction
│   └── rate_limit.py
├── ml/                     # ML training & inference
│   ├── inference.py        # MLChartGenerator (main inference entry point)
│   ├── model.py            # PyTorch network definition
│   ├── dataset.py          # Training dataset
│   ├── prepare_data.py     # Converts .sm + audio → npz training data
│   ├── train.py            # Training loop
│   └── checkpoints/        # best_model.pt, style_profiles.json
├── alembic/                # DB migrations
├── tests/
├── scripts/
└── data/

frontend/
├── src/
│   ├── main.tsx            # React entry
│   ├── App.tsx             # State-driven screen router
│   ├── app/store/          # zustand store (useGameStore + slices)
│   ├── components/         # Shared UI: Button, Toast, ProgressBar
│   ├── config/api.ts       # Axios instance
│   ├── features/
│   │   ├── game/           # Gameplay loop: components, hooks, types, utils
│   │   ├── menu/           # Menu/Loading/DifficultySelect/Ready screens
│   │   ├── results/        # Results screen + utils
│   │   ├── discover/       # Song discovery
│   │   ├── playlists/      # Playlist browsing
│   │   └── songs/          # Song detail
│   ├── hooks/              # Shared hooks (useLocalStorage, useToast)
│   ├── lib/                # Axios config, simfile export
│   └── types/              # Shared types (api.types, common.types)
```
**Note**: There are no top-level `services/`, `routers/`, or `modules/` directories.
All backend code lives under `backend/src/`.

## Commands

### Backend
```bash
cd backend
source .venv/bin/activate  # uv sync first if missing
uv sync                    # install deps
cp .env.example .env       # edit as needed

python main.py             # dev server at localhost:8000 (auto-reload)
PYTHONPATH=. python main.py  # if run from outside backend/
```

### ML Data Prep
```bash
cd backend
source .venv/bin/activate
python -m ml.prepare_data --charts-dir ../../charts --output-dir data/training_data [--limit N]
```

### Frontend
```bash
cd frontend
npm install
npm run dev      # dev server at localhost:3000 (proxies /api → backend:8000)
npm run build    # production build
npm run lint     # ESLint on .ts/.tsx
npm run preview  # preview production build
```

### Docker
```bash
docker compose up -d     # start both services
docker compose logs -f   # follow logs
docker compose down      # stop
```

There are **no test files or test commands** configured.

## Architecture

### Backend — Feature Modules

Each domain is a self-contained subpackage under `src/` with its own **router** (FastAPI APIRouter), **schemas** (Pydantic), **service** (business logic), and optionally **models** (SQLAlchemy).

| Module | Responsibility |
|---|---|
| `src/generation/` | Step generation: `AudioProcessor` (librosa beat/onset/energy/spectral analysis), ML call via `MLChartGenerator`, schemas |
| `src/songs/` | Song CRUD, `.sm`/`.ssc` parsing (`parse_sm_file` → flat `Note` events; `parse_sm` → grouped `Step` records) |
| `src/charts/` | Chart data models, SM export |
| `src/audio/` | Audio file streaming |
| `src/playlists/` | Playlist CRUD |
| `src/discover/` | Genre-based song discovery |
| `src/search/` | Song search |
| `src/storage/` | File storage abstraction (local or S3) |
| `src/audius/` | Audius API integration (no key needed) |
| `src/jamendo/` | Jamendo API integration (needs `JAMENDO_CLIENT_ID`) |

### ML Module (`ml/`)

- **`MLChartGenerator`** (`ml/inference.py:360`) — the sole inference entry point. Loads checkpoint at startup via `ML_MODEL_PATH` env var.
- **`prepare_data.py`** — converts `.sm` charts + audio into npz training data (mel spectrogram + onset strength + spectral contrast features).
- **Training pipeline**: `prepare_data.py` → `dataset.py` → `train.py` → checkpoint saved to `ml/checkpoints/best_model.pt`.
- No algorithmic fallback if checkpoint can't be loaded — app fails fast.

### Frontend — State via zustand

State is managed by a **zustand** store (`useGameStore`) composed of slices:

| Slice | Holds |
|---|---|
| `flowSlice` | `gameState: GameState` enum |
| `chartSlice` | Current chart/steps data |
| `loadingSlice` | Generation/loading progress |
| `resultsSlice` | Score + accuracy for results screen |
| `preferencesSlice` | User settings (not persisted) |

`App.tsx` reads `gameState` from the store and renders the matching screen — state is **not** passed via props.

**GameState flow:**
```
MENU → LOADING → DIFFICULTY_SELECT → READY → PLAYING|PAUSED → FINISHED
```

### Frontend Screens

| State | Component | Location |
|---|---|---|
| MENU | `MenuScreen` | `features/menu/components/MenuScreen.tsx` |
| LOADING | `LoadingScreen` | `features/menu/components/LoadingScreen.tsx` |
| DIFFICULTY_SELECT | `DifficultySelectScreen` | `features/menu/components/DifficultySelectScreen.tsx` |
| READY | `ReadyScreen` | `features/menu/components/ReadyScreen.tsx` |
| PLAYING / PAUSED | `GameScreen` | `features/game/components/GameScreen.tsx` |
| FINISHED | `ResultsScreen` | `features/results/components/ResultsScreen.tsx` |

### Gameplay Loop

Game loop uses `requestAnimationFrame` at 60 FPS in `GameScreen`. 5-second visible window. Arrow Y-position calculated dynamically from BPM via `getArrowSpeed(tempo)`.

### Game Constants (`features/game/types/game.types.ts`)

| Constant | Value |
|---|---|
| `TIMING.PERFECT` / `.GOOD` / `.OK` / `.MISS` | 50 / 100 / 150 / 200 ms |
| `POINTS.PERFECT` / `.GOOD` / `.OK` / `.MISS` | 100 / 50 / 25 / 0 |
| `COMBO_MULTIPLIER` | 10→1.1, 25→1.25, 50→1.5, 100→2.0 |
| `HOLD_SCORING.TICK_INTERVAL` / `.POINTS_PER_TICK` / `.COMPLETION_BONUS` | 0.1s / 10 / 50 |
| `VISUAL_CONFIG.HIT_ZONE_Y` / `.ARROW_SIZE` / `.VISIBLE_WINDOW` / `.SPAWN_Y` | 80px / 90px / 5s / 700px |
| `getArrowSpeed(tempo)` | `tempo * getSpeedMod(tempo) * 0.9` (~400–600 px/s) |
| `KEY_MAP` | ArrowLeft/Down/Up/Right → LEFT/DOWN/UP/RIGHT |

### Hit Detection (`features/game/utils/scoring.ts`)

- `evaluateHit(timeDiffMs)` → `Judgment` (PERFECT/GOOD/OK/MISS)
- `calculatePoints(judgment, combo)` → base points × combo multiplier
- `getComboMultiplier(combo)` → multiplier at threshold, else 1.0

### API Contract

```http
POST /api/generate — multipart: file + difficulty (beginner|intermediate|expert)
POST /api/generate-url — JSON: { url, difficulty }
```
Response: `{ song_id, charts: { beginner: { steps: [...] }, ... }, song_info, audio_url }`

## Environment Variables (backend/.env)

| Variable | Default | Purpose |
|---|---|---|
| `ENV` | `development` | `production` hides docs/reload/error detail |
| `ML_MODEL_PATH` | `./ml/checkpoints/best_model.pt` | ML checkpoint path |
| `GENERATION_TIMING_OFFSET_MS` | `-30` | Calibration for ML notes (ms) |
| `AUDIUS_APP_NAME` | `stepageddon` | Audius app identifier |
| `JAMENDO_CLIENT_ID` | (empty) | Jamendo API key; empty = disabled |
| `STORAGE_BACKEND` | `local` | `local` or `s3` |
| `AUDIO_STORAGE_PATH` | `~/Desktop/ddr/stepageddon-data/audio` | Local file storage |
| `S3_BUCKET` / `S3_*` | (empty) | S3/R2 credentials (when `STORAGE_BACKEND=s3`) |
| `CORS_ORIGINS` | `localhost:3000,localhost:5173` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Server bind |
| `MAX_FILE_SIZE_MB` / `MAX_DURATION_SECONDS` | 50 / 600 | Upload limits |
| `DATABASE_URL` | `postgresql+asyncpg://...localhost:5432/stepageddon` | Postgres connection |

## Conventions

- **Frontend**: TypeScript everywhere — `.tsx` components, `.ts` modules/utilities. No `.jsx`/`.js` in source.
- **Backend**: Python 3.12+. Feature modules under `src/` with consistent layout: router.py, schemas.py, service.py, models.py.
- **State**: zustand store with slices, `useGameStore(s => s.field)` selectors. No prop drilling.
- **ML**: Model trained offline; API runs inference only. Checkpoint loaded at startup.
- **Imports**: `from src.xxx import ...` (absolute within backend package). Must run from `backend/` dir or set `PYTHONPATH=.`.
- **Audio**: All loaded at 22050 Hz via librosa. Files saved to `AUDIO_STORAGE_PATH` (never auto-deleted).
- **No test suite exists** — no pytest configuration, no test files.

## Notes

(Quick-add section for later)

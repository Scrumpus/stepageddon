# Beat Sync - Quick Start Guide

Get up and running with Beat Sync in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- Anthropic API key ([get one here](https://console.anthropic.com/))

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd beat-sync
```

## Step 2: Configure API Keys

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:

```env
ANTHROPIC_API_KEY=sk-ant-your-api-key-here

# Optional: Add Spotify credentials for Spotify URL support
SPOTIFY_CLIENT_ID=your-spotify-client-id
SPOTIFY_CLIENT_SECRET=your-spotify-client-secret
```

## Step 3: Start the Application

```bash
docker-compose up -d
```

This will:
- Build the Docker containers
- Start the backend server on port 8000
- Start the frontend server on port 80

## Step 4: Open Your Browser

Navigate to: http://localhost

You should see the Beat Sync main menu!

## Step 5: Create Your First Chart

### Option A: Upload a File

1. Select your desired difficulty (Beginner, Intermediate, or Expert)
2. Click "Upload File" tab
3. Drop an audio file (MP3, WAV, OGG, or FLAC)
4. Wait 20-30 seconds for generation
5. Press "Start Game" when ready
6. Play using arrow keys (←↓↑→)!

### Option B: Use a URL

1. Select difficulty
2. Click "URL" tab
3. Paste a YouTube or Spotify URL
4. Click "Generate Steps"
5. Wait for processing
6. Start playing!

## Gameplay Tips

- **Perfect timing**: Hit arrows within ±50ms for maximum points
- **Build combos**: Chain perfect hits for score multipliers
- **Watch the tempo**: Each song has its own rhythm
- **Practice**: Try Beginner first to learn the mechanics

## Common Issues

### "API key not configured"
- Make sure you added `ANTHROPIC_API_KEY` to your `.env` file
- Restart Docker: `docker-compose down && docker-compose up -d`

### "Could not download from YouTube"
- Some videos may be restricted
- Try a different video
- Check your internet connection

### "Spotify track doesn't have a preview"
- Not all tracks have 30-second previews available
- Try a different song or use YouTube instead

## Useful Commands

```bash
# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Rebuild containers (after code changes)
docker-compose build
docker-compose up -d

# Remove all data and start fresh
docker-compose down -v
```

## What's Next?

- Try different difficulty levels
- Experiment with various music genres
- Check out the [full documentation](README.md)
- Explore [customization options](docs/customization.md)
- Contribute to the project (see [CONTRIBUTING.md](CONTRIBUTING.md))

## Local Development (Without Docker)

If you prefer local development:

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python main.py
```

Backend runs at: http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:3000

## Getting Help

- **Issues**: Check [GitHub Issues](https://github.com/yourusername/beat-sync/issues)
- **Documentation**: See [README.md](README.md) for detailed info
- **Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md) for technical details

## Have Fun!

That's it! You're ready to create AI-generated rhythm charts and play Beat Sync! 🎮🎵

Remember:
- ←↓↑→ to hit arrows
- ESC to pause
- Practice makes perfect!

Enjoy the game!

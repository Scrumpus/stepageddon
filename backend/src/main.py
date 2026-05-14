"""
Beat Sync Backend - FastAPI Application
Main entry point for the rhythm game backend API
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.config import settings
from src.database import engine
from src.songs.router import router as songs_router
from src.charts.router import router as charts_router
from src.playlists.router import router as playlists_router
from src.generation.router import router as generation_router
from src.audio.router import router as audio_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🎮 Beat Sync Backend Starting...")

    # Create audio storage directory
    os.makedirs(settings.AUDIO_STORAGE_PATH, exist_ok=True)
    logger.info(f"📁 Audio storage: {settings.AUDIO_STORAGE_PATH}")

    if not settings.SPOTIFY_CLIENT_ID or not settings.SPOTIFY_CLIENT_SECRET:
        logger.warning("⚠️  Spotify credentials not set - Spotify URLs won't work")
    else:
        logger.info("✓ Spotify credentials configured")

    logger.info("✓ Beat Sync Backend Ready!")

    yield

    # Shutdown
    logger.info("🛑 Beat Sync Backend Shutting Down...")
    await engine.dispose()


# Create FastAPI app
app = FastAPI(
    title="Beat Sync API",
    description="AI-powered DDR-style rhythm game backend",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "online",
        "service": "Beat Sync API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "api": "operational",
            "spotify": "configured" if settings.SPOTIFY_CLIENT_ID else "not configured"
        }
    }


# Include routers
app.include_router(generation_router, prefix="/api", tags=["generation"])
app.include_router(audio_router, prefix="/api", tags=["audio"])
app.include_router(songs_router, prefix="/api", tags=["songs"])
app.include_router(charts_router, prefix="/api", tags=["charts"])
app.include_router(playlists_router, prefix="/api", tags=["playlists"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )

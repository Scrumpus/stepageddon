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

from routers import generation, audio
from core.config import settings

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
    
    # Verify API keys
    if not settings.ANTHROPIC_API_KEY:
        logger.warning("⚠️  ANTHROPIC_API_KEY not set - AI generation will fail")
    else:
        logger.info("✓ Anthropic API key configured")
    
    if not settings.SPOTIFY_CLIENT_ID or not settings.SPOTIFY_CLIENT_SECRET:
        logger.warning("⚠️  Spotify credentials not set - Spotify URLs won't work")
    else:
        logger.info("✓ Spotify credentials configured")
    
    logger.info("✓ Beat Sync Backend Ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Beat Sync Backend Shutting Down...")


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
            "anthropic": "configured" if settings.ANTHROPIC_API_KEY else "not configured",
            "spotify": "configured" if settings.SPOTIFY_CLIENT_ID else "not configured"
        }
    }


# Include routers
app.include_router(generation.router, prefix="/api", tags=["generation"])
app.include_router(audio.router, prefix="/api", tags=["audio"])


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
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )

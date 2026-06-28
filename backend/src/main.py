"""
Stepageddon Backend - FastAPI Application
Main entry point for the rhythm game backend API
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.config import settings
from src.database import engine
from src.rate_limit import limiter
from src.songs.router import router as songs_router
from src.charts.router import router as charts_router
from src.playlists.router import router as playlists_router
from src.generation.router import router as generation_router
from src.audio.router import router as audio_router
from src.search.router import router as search_router
from src.discover.router import router as discover_router

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
    logger.info("🎮 Stepageddon Backend Starting...")

    # Create audio storage directory
    os.makedirs(settings.AUDIO_STORAGE_PATH, exist_ok=True)
    logger.info(f"📁 Audio storage: {settings.AUDIO_STORAGE_PATH}")

    logger.info(f"✓ Audius enabled (app_name={settings.AUDIUS_APP_NAME})")
    if settings.JAMENDO_CLIENT_ID:
        logger.info("✓ Jamendo client_id configured")
    else:
        logger.warning("⚠️  JAMENDO_CLIENT_ID not set - Jamendo search/links disabled")

    logger.info("✓ Stepageddon Backend Ready!")

    yield

    # Shutdown
    logger.info("🛑 Stepageddon Backend Shutting Down...")
    await engine.dispose()


_is_prod = settings.ENV == "production"

# Create FastAPI app — hide OpenAPI surface in production to reduce
# information disclosure to unauthenticated scanners.
app = FastAPI(
    title="Stepageddon API",
    description="AI-powered DDR-style rhythm game backend",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _is_prod else "/docs",
    redoc_url=None if _is_prod else "/redoc",
    openapi_url=None if _is_prod else "/openapi.json",
)

# Configure CORS — restrict methods/headers to the actual API surface.
# Wildcard methods + allow_credentials=True is a CSRF foot-gun.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Rate limiting — anonymous public endpoints (generation) need per-IP throttling.
# Per-route limits are declared on the handlers via @limiter.limit(...).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "online",
        "service": "Stepageddon API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "api": "operational",
            "audius": "enabled",
            "jamendo": "configured" if settings.JAMENDO_CLIENT_ID else "not configured"
        }
    }


# Include routers
app.include_router(generation_router, prefix="/api", tags=["generation"])
app.include_router(audio_router, prefix="/api", tags=["audio"])
app.include_router(songs_router, prefix="/api", tags=["songs"])
app.include_router(charts_router, prefix="/api", tags=["charts"])
app.include_router(playlists_router, prefix="/api", tags=["playlists"])
app.include_router(search_router, prefix="/api", tags=["search"])
app.include_router(discover_router, prefix="/api", tags=["discover"])


# Global exception handler — never leak exception detail to clients.
# Full traceback goes to the server log.
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    body = {"error": "Internal server error"}
    if not _is_prod:
        body["detail"] = str(exc)
    return JSONResponse(status_code=500, content=body)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=not _is_prod,
        log_level=settings.LOG_LEVEL.lower()
    )

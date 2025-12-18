"""
Configuration settings for Beat Sync backend
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    SPOTIFY_CLIENT_ID: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    SPOTIFY_CLIENT_SECRET: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # File Storage
    AUDIO_STORAGE_PATH: str = os.getenv("AUDIO_STORAGE_PATH", "./audio_storage")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_DURATION_SECONDS: int = int(os.getenv("MAX_DURATION_SECONDS", "600"))
    
    # CORS
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:3000,http://localhost:5173"
    ).split(",")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Generation Settings
    AI_MODEL: str = "claude-sonnet-4-20250514"
    MAX_GENERATION_TIME: int = 30
    
    class Config:
        env_file = ".env"


settings = Settings()

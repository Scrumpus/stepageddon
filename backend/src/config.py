"""
Configuration settings for Beat Sync backend
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Keys
    SPOTIFY_CLIENT_ID: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    SPOTIFY_CLIENT_SECRET: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # File Storage
    # STORAGE_BACKEND selects the storage implementation (see src/storage/).
    # "local" reads/writes AUDIO_STORAGE_PATH. "s3" uses S3_* settings.
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")
    AUDIO_STORAGE_PATH: str = os.getenv("AUDIO_STORAGE_PATH", "~/stepageddon-data/audio")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_DURATION_SECONDS: int = int(os.getenv("MAX_DURATION_SECONDS", "600"))

    # S3-compatible storage (used when STORAGE_BACKEND=s3)
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")
    S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL", "")
    S3_REGION: str = os.getenv("S3_REGION", "auto")
    S3_ACCESS_KEY_ID: str = os.getenv("S3_ACCESS_KEY_ID", "")
    S3_SECRET_ACCESS_KEY: str = os.getenv("S3_SECRET_ACCESS_KEY", "")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://stepageddon:stepageddon@localhost:5432/stepageddon",
    )

    # CORS
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:3001,http://localhost:5173"
    ).split(",")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ML Model Settings
    ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", "./ml/checkpoints/best_model.pt")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

"""
Configuration settings for Beat Sync backend
"""

import os
from typing import Annotated, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode


class Settings(BaseSettings):
    """Application settings"""

    # Deployment environment: "development" | "production".
    # Gates auto-reload, OpenAPI docs exposure, and exception-detail leakage.
    ENV: str = os.getenv("ENV", "development")

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

    # CORS — empty by default so misconfigured prod fails closed.
    # Set CORS_ORIGINS="http://localhost:3000,http://localhost:5173" in your
    # dev .env (see .env.example). NoDecode disables pydantic-settings'
    # default JSON parsing for List[str] so a plain comma-separated string
    # in the env is accepted.
    CORS_ORIGINS: Annotated[List[str], NoDecode] = []

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_csv_origins(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ML Model Settings
    ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", "./ml/checkpoints/best_model.pt")

    # YouTube cookies file (Netscape format) handed to yt-dlp to get past
    # bot/age checks. Empty falls back to backend/cookies.txt if present.
    # yt-dlp rewrites the jar on exit, so the client always copies this to a
    # writable temp file first (the source is often a read-only mount).
    YT_COOKIES_PATH: str = os.getenv("YT_COOKIES_PATH", "")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

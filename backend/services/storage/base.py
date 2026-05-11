"""Abstract storage backend.

All audio assets (user-uploaded MP3s, YouTube downloads, seeded DDR files,
generated previews) live behind this interface. Routers and services should
never touch the filesystem directly — they hold a storage key (e.g.
``"abc-123.mp3"`` or ``"ddr/butterfly/Butterfly.ogg"``) and ask the backend
to produce a local path, write bytes, or serve a response.

This is the seam where a future ``S3Storage`` (R2, etc.) plugs in without
changing call sites.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from fastapi import Response


class StorageBackend(ABC):
    """Storage backend for audio + related assets, keyed by string."""

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> Path:
        """Store ``data`` under ``key`` and return a local path the caller can
        read from (e.g. to hand to librosa). For remote backends this may be a
        temp path; for local it is the canonical storage path.
        """

    @abstractmethod
    def reserve_path(self, key: str) -> Path:
        """Return a local path the caller will write to externally (e.g. yt-dlp
        output). The caller MUST call :meth:`commit` once the write succeeds,
        or :meth:`delete` to abandon it.
        """

    @abstractmethod
    def commit(self, key: str, local_path: Path) -> None:
        """Finalize a write produced via :meth:`reserve_path`. No-op for local
        backends; uploads + cleanup for remote backends.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if ``key`` is present in storage."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key`` from storage. Silently no-op if missing."""

    @abstractmethod
    def serve(self, key: str) -> Response:
        """Return a FastAPI response that delivers the asset to the client."""

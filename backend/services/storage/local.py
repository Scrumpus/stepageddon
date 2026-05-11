"""Local filesystem storage backend."""

import logging
import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

from .base import StorageBackend

logger = logging.getLogger(__name__)

MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class LocalStorage(StorageBackend):
    """Stores assets under a single root directory on the local filesystem.

    Keys may contain forward slashes; those become subdirectories. Path
    traversal (``..``) is rejected at :meth:`_resolve`.
    """

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info("LocalStorage root: %s", self.root)

    def _resolve(self, key: str) -> Path:
        target = (self.root / key).resolve()
        if self.root != target and self.root not in target.parents:
            raise HTTPException(status_code=403, detail="Forbidden")
        return target

    def write_bytes(self, key: str, data: bytes) -> Path:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def reserve_path(self, key: str) -> Path:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def commit(self, key: str, local_path: Path) -> None:
        # File was written in place by the caller; nothing to do.
        return

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to delete %s: %s", key, e)

    def serve(self, key: str):
        try:
            target = self._resolve(key)
        except HTTPException:
            logger.warning("Rejected traversal attempt: %s", key)
            raise
        if not target.is_file():
            logger.warning("Asset not found: %s", key)
            raise HTTPException(status_code=404, detail="Asset not found")
        media_type = MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
        return FileResponse(
            str(target),
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
        )

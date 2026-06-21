"""Storage backends: abstract interface + local filesystem + S3-compatible.

All audio assets (user-uploaded MP3s, YouTube downloads, seeded DDR files,
generated previews) live behind :class:`StorageBackend`. Routers and services
should never touch the filesystem directly — they hold a storage key (e.g.
``"abc-123.mp3"`` or ``"ddr/butterfly/Butterfly.ogg"``) and ask the backend
to produce a local path, write bytes, or serve a response.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from fastapi import HTTPException, Response
from fastapi.responses import FileResponse, RedirectResponse

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

PRESIGNED_URL_TTL_SECONDS = 900  # 15 minutes


def _content_type(key: str) -> str:
    return MEDIA_TYPES.get(Path(key).suffix.lower(), "application/octet-stream")


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
    def read_bytes(self, key: str) -> bytes:
        """Return the raw bytes stored under ``key``. Raises 404 if missing."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if ``key`` is present in storage."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key`` from storage. Silently no-op if missing."""

    @abstractmethod
    def serve(self, key: str) -> Response:
        """Return a FastAPI response that delivers the asset to the client."""


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

    def read_bytes(self, key: str) -> bytes:
        target = self._resolve(key)
        if not target.is_file():
            logger.warning("Asset not found: %s", key)
            raise HTTPException(status_code=404, detail="Asset not found")
        return target.read_bytes()

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


class S3Storage(StorageBackend):
    """S3-compatible backend (Cloudflare R2, AWS S3, MinIO, etc.).

    Keeps a small on-disk cache so callers that just wrote bytes can hand the
    returned path to librosa / yt-dlp without an extra round-trip. ``serve``
    returns a 307 redirect to a short-lived presigned GET URL so the API
    isn't on the bandwidth path.
    """

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        region: str = "auto",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        cache_dir: str | os.PathLike | None = None,
    ):
        if not bucket:
            raise ValueError("S3Storage requires a bucket name")

        self.bucket = bucket
        self._lock = threading.Lock()
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            region_name=region or "auto",
            aws_access_key_id=access_key_id or None,
            aws_secret_access_key=secret_access_key or None,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

        cache_root = cache_dir or Path(tempfile.gettempdir()) / "stepageddon-r2-cache"
        self.cache_dir = Path(cache_root).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("S3Storage bucket=%s endpoint=%s cache=%s", bucket, endpoint_url, self.cache_dir)

    def _cache_path(self, key: str) -> Path:
        target = (self.cache_dir / key).resolve()
        if self.cache_dir != target and self.cache_dir not in target.parents:
            raise HTTPException(status_code=403, detail="Forbidden")
        return target

    def write_bytes(self, key: str, data: bytes) -> Path:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=_content_type(key),
        )
        return path

    def reserve_path(self, key: str) -> Path:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def commit(self, key: str, local_path: Path) -> None:
        self._client.upload_file(
            str(local_path),
            self.bucket,
            key,
            ExtraArgs={"ContentType": _content_type(key)},
        )

    def read_bytes(self, key: str) -> bytes:
        cache = self._cache_path(key)
        if cache.is_file():
            return cache.read_bytes()
        try:
            obj = self._client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                raise HTTPException(status_code=404, detail="Asset not found") from e
            raise
        data = obj["Body"].read()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        return data

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            logger.warning("R2 delete failed for %s: %s", key, e)
        try:
            self._cache_path(key).unlink(missing_ok=True)
        except (OSError, HTTPException):
            pass

    def serve(self, key: str):
        try:
            url = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=PRESIGNED_URL_TTL_SECONDS,
            )
        except ClientError as e:
            logger.error("Failed to presign %s: %s", key, e)
            raise HTTPException(status_code=500, detail="Failed to sign asset URL") from e
        return RedirectResponse(url=url, status_code=307)

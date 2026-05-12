"""S3-compatible storage backend.

Built for Cloudflare R2 (also works with AWS S3, MinIO, etc.). The backend
keeps a small on-disk cache so callers that just wrote bytes can hand the
returned path to librosa / yt-dlp without an extra round-trip.

Delivery model: :meth:`serve` returns a 307 redirect to a short-lived
presigned GET URL so the API isn't on the bandwidth path.
"""

import logging
import os
import tempfile
import threading
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from fastapi import HTTPException
from fastapi.responses import RedirectResponse

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

PRESIGNED_URL_TTL_SECONDS = 900  # 15 minutes


def _content_type(key: str) -> str:
    return MEDIA_TYPES.get(Path(key).suffix.lower(), "application/octet-stream")


class S3Storage(StorageBackend):
    """S3-compatible backend.

    Args:
        bucket: Target bucket name.
        endpoint_url: S3 endpoint. For R2: ``https://<account>.r2.cloudflarestorage.com``.
        region: ``"auto"`` for R2, otherwise the AWS region.
        access_key_id / secret_access_key: API credentials.
        cache_dir: Local directory used to mirror writes so the caller can
            immediately read the file (librosa, yt-dlp). Defaults to a stable
            subdir under the system tempdir.
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

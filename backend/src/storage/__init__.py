"""Storage backends for audio + related assets.

Use :func:`get_storage` to obtain the configured singleton. Selection is
driven by ``settings.STORAGE_BACKEND`` (``"local"`` today; ``"s3"`` later).
"""

from functools import lru_cache

from src.config import settings
from src.storage.client import LocalStorage, S3Storage, StorageBackend


@lru_cache(maxsize=1)
def get_storage() -> StorageBackend:
    backend = (settings.STORAGE_BACKEND or "local").lower()
    if backend == "local":
        return LocalStorage(settings.AUDIO_STORAGE_PATH)
    if backend == "s3":
        return S3Storage(
            bucket=settings.S3_BUCKET,
            endpoint_url=settings.S3_ENDPOINT_URL or None,
            region=settings.S3_REGION or "auto",
            access_key_id=settings.S3_ACCESS_KEY_ID or None,
            secret_access_key=settings.S3_SECRET_ACCESS_KEY or None,
        )
    raise ValueError(f"Unknown STORAGE_BACKEND: {backend!r}")


__all__ = ["StorageBackend", "LocalStorage", "S3Storage", "get_storage"]

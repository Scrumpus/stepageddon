"""S3-compatible storage backend (stub).

Intended to drop in for Cloudflare R2, AWS S3, MinIO, etc. — anything with an
S3 API. Phase 2: implement against ``boto3`` / ``aioboto3``.

Sketch of intended behavior:

- :meth:`write_bytes` → ``put_object``; returns a temp path the caller can read.
- :meth:`reserve_path` → temp file under ``/tmp``.
- :meth:`commit` → ``upload_file`` then unlink the temp.
- :meth:`serve` → redirect to a presigned GET URL so the backend isn't a
  bandwidth bottleneck.
"""

from pathlib import Path

from .base import StorageBackend


class S3Storage(StorageBackend):
    def __init__(self, bucket: str, endpoint_url: str | None = None, **_):
        raise NotImplementedError("S3Storage is not implemented yet (phase 2)")

    def write_bytes(self, key: str, data: bytes) -> Path:
        raise NotImplementedError

    def reserve_path(self, key: str) -> Path:
        raise NotImplementedError

    def commit(self, key: str, local_path: Path) -> None:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def serve(self, key: str):
        raise NotImplementedError

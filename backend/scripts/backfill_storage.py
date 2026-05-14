"""Upload every file under AUDIO_STORAGE_PATH into the configured storage backend.

Used after switching ``STORAGE_BACKEND`` from ``local`` to ``s3`` to push the
local mirror up to R2/S3. Keys are computed as the path relative to
``AUDIO_STORAGE_PATH``, matching what ``seed_ddr`` writes to Postgres.

Usage:
    uv run python -m scripts.backfill_storage
    uv run python -m scripts.backfill_storage --root /custom/path --dry-run
    uv run python -m scripts.backfill_storage --overwrite   # re-upload existing keys

Idempotent: skips keys already present in the backend unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from core.config import settings
from services import get_storage

logger = logging.getLogger(__name__)


def iter_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            yield path


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        logger.error("Root not found: %s", root)
        return 1

    storage = get_storage()
    logger.info("Backfilling from %s into %s", root, type(storage).__name__)

    files = list(iter_files(root))
    if not files:
        logger.warning("No files under %s", root)
        return 0

    uploaded = skipped = failed = 0
    for path in tqdm(files, desc="uploading", unit="file"):
        key = str(path.relative_to(root))

        if not args.overwrite and storage.exists(key):
            skipped += 1
            continue

        if args.dry_run:
            logger.info("[dry-run] %s → %s", path, key)
            uploaded += 1
            continue

        try:
            storage.commit(key, path)
            uploaded += 1
        except Exception:
            logger.exception("Upload failed for %s", key)
            failed += 1

    logger.info(
        "✓ Done. uploaded=%d skipped=%d failed=%d", uploaded, skipped, failed
    )
    return 0 if failed == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload AUDIO_STORAGE_PATH contents into the configured backend"
    )
    parser.add_argument(
        "--root",
        default=settings.AUDIO_STORAGE_PATH,
        help="Local directory to mirror (default: settings.AUDIO_STORAGE_PATH)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-upload files even if the key already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be uploaded without sending anything",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

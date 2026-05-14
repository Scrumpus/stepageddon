"""One-shot: strip the legacy ``ddr/`` prefix from all path columns on ``songs``.

The R2 bucket was populated with keys at ``<slug>/<file>`` but the seed script
recorded them as ``ddr/<slug>/<file>``. This aligns the DB to the bucket so
``/api/audio/<key>`` resolves.

Usage:
    uv run python -m scripts.strip_ddr_prefix [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from sqlalchemy import func, select, update

from src.database import AsyncSessionLocal
from src.songs.models import Song

logger = logging.getLogger(__name__)

PATH_COLS = ("audio_path", "banner_path", "jacket_path", "bg_path")
PREFIX = "ddr/"


async def run(dry_run: bool) -> int:
    async with AsyncSessionLocal() as session:
        for col_name in PATH_COLS:
            col = getattr(Song, col_name)
            count = (
                await session.execute(
                    select(func.count()).select_from(Song).where(col.like(f"{PREFIX}%"))
                )
            ).scalar()
            logger.info("%s with prefix: %d", col_name, count)
            if dry_run or count == 0:
                continue
            await session.execute(
                update(Song)
                .where(col.like(f"{PREFIX}%"))
                .values({col_name: func.substr(col, len(PREFIX) + 1)})
            )
        if not dry_run:
            await session.commit()
            logger.info("✓ Committed")
        else:
            logger.info("(dry-run, no changes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args.dry_run))


if __name__ == "__main__":
    sys.exit(main())

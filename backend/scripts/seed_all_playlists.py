"""Seed every DDR game folder as both songs and a playlist.

Walks the DDR root, and for each ``[NN] Game Name/`` subfolder:
  1. Imports every song into the ``songs`` + ``charts`` tables (re-uses
     :func:`scripts.seed_ddr.seed_game`).
  2. Builds a playlist named after the game with all those songs as members
     (re-uses :func:`scripts.seed_playlist_from_game.build_playlist_for_game`).

Usage:
    uv run python -m scripts.seed_all_playlists
    uv run python -m scripts.seed_all_playlists --root "DDR 1st-A20 Plus" --limit-per-game 5
    uv run python -m scripts.seed_all_playlists --skip-songs   # only build playlists
    uv run python -m scripts.seed_all_playlists --skip-playlists  # only import songs
    uv run python -m scripts.seed_all_playlists --keep-prefix   # keep "[01] " prefix in names
    uv run python -m scripts.seed_all_playlists --dry-run       # parse only, no DB writes

Idempotent: re-running is safe.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from db.session import AsyncSessionLocal
from scripts.seed_ddr import DEFAULT_ROOT, seed_game
from scripts.seed_playlist_from_game import build_playlist_for_game
from services import get_storage

logger = logging.getLogger(__name__)

PREFIX_RE = re.compile(r"^\[\d+\]\s*")


def stripped_name(game: str) -> str:
    return PREFIX_RE.sub("", game).strip()


async def run(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.is_dir():
        logger.error("DDR root not found: %s", root)
        return 1

    storage = get_storage()

    games = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not games:
        logger.error("No game folders under %s", root)
        return 1

    logger.info("Found %d game folders under %s", len(games), root)

    if args.dry_run:
        for game in games:
            logger.info("[dry-run] would import + bundle: %s", game)
        return 0

    songs_total = 0
    playlists_total = 0

    async with AsyncSessionLocal() as session:
        for game in games:
            try:
                if not args.skip_songs:
                    songs_added = await seed_game(
                        session,
                        root=root,
                        game=game,
                        storage=storage,
                        limit=args.limit_per_game,
                        dry_run=False,
                    )
                    songs_total += songs_added
                    await session.commit()

                if not args.skip_playlists:
                    name: Optional[str] = None if args.keep_prefix else stripped_name(game)
                    playlist = await build_playlist_for_game(
                        session,
                        game=game,
                        name=name,
                        description=game,
                    )
                    if playlist is not None:
                        playlists_total += 1
                    await session.commit()
            except Exception:
                await session.rollback()
                logger.exception("Failed on game %r — continuing", game)

    logger.info(
        "✓ Done. Songs persisted across runs: %d (this invocation), playlists built: %d",
        songs_total,
        playlists_total,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import all DDR songs and build a playlist per game folder"
    )
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help='Path to the DDR root (default: "DDR 1st-A20 Plus")')
    parser.add_argument("--limit-per-game", type=int, default=None,
                        help="Cap the number of songs imported per game (debug)")
    parser.add_argument("--skip-songs", action="store_true",
                        help="Only build playlists from already-seeded songs")
    parser.add_argument("--skip-playlists", action="store_true",
                        help="Only import songs; don't create playlists")
    parser.add_argument("--keep-prefix", action="store_true",
                        help='Keep the "[NN] " prefix in playlist names (default: stripped)')
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without touching DB or filesystem")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())

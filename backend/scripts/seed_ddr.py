"""Seed Postgres from the DDR 1st-A20 Plus song collection.

Usage:
    uv run python -m scripts.seed_ddr [--game "[01] DDR 1st Mix" | --game all]
                                       [--root "DDR 1st-A20 Plus"]
                                       [--limit N]
                                       [--dry-run]

Idempotent: re-running updates rows in place via ON CONFLICT and skips
asset uploads that already exist in the configured storage backend.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import librosa
from slugify import slugify
from tqdm import tqdm

from db.session import AsyncSessionLocal
from services import get_storage
from services.chart_persistence import persist_ddr_song
from services.sm_parser import ParsedSong, parse_sm
from services.storage.base import StorageBackend

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "DDR 1st-A20 Plus"
DEFAULT_GAME = "[01] DDR 1st Mix"
ASSET_EXTS = {".ogg", ".mp3", ".wav", ".png", ".jpg", ".jpeg"}
AUDIO_EXTS = (".ogg", ".mp3", ".wav", ".flac", ".m4a")

GAME_INDEX_RE = re.compile(r"^\[(\d+)\]")


def parse_game_index(game_dir_name: str) -> Optional[int]:
    m = GAME_INDEX_RE.match(game_dir_name)
    return int(m.group(1)) if m else None


def find_chart_file(song_dir: Path) -> Optional[Path]:
    for pattern in ("*.sm", "*.ssc"):
        files = sorted(song_dir.glob(pattern))
        if files:
            return files[0]
    return None


def upload_asset(src: Path, key: str, storage: StorageBackend) -> None:
    """Copy ``src`` into storage under ``key``. No-op if already present."""
    if storage.exists(key):
        return
    dst = storage.reserve_path(key)
    shutil.copy2(src, dst)
    storage.commit(key, dst)


def upload_assets(song_dir: Path, slug: str, storage: StorageBackend) -> None:
    for src in song_dir.iterdir():
        if not src.is_file() or src.suffix.lower() not in ASSET_EXTS:
            continue
        upload_asset(src, f"{slug}/{src.name}", storage)


def key_or_none(
    filename: Optional[str], slug: str, storage: StorageBackend
) -> Optional[str]:
    if not filename:
        return None
    key = f"{slug}/{filename}"
    return key if storage.exists(key) else None


async def seed_one_song(
    session,
    song_dir: Path,
    game: str,
    game_index: Optional[int],
    storage: StorageBackend,
    dry_run: bool,
) -> bool:
    chart_file = find_chart_file(song_dir)
    if not chart_file:
        logger.debug("Skipping %s — no .sm/.ssc", song_dir)
        return False

    try:
        parsed: ParsedSong = parse_sm(chart_file)
    except Exception as e:
        logger.error("Parse failed for %s: %s", chart_file, e)
        return False

    if not parsed.charts:
        logger.warning("No usable dance-single charts in %s — skipping", chart_file)
        return False
    if not parsed.music_file:
        logger.warning("No #MUSIC declared in %s — skipping", chart_file)
        return False

    slug = slugify(song_dir.name) or slugify(parsed.title) or song_dir.name.lower()

    if dry_run:
        logger.info(
            "[dry-run] %s | %s | charts=%d | music=%s",
            parsed.title,
            slug,
            len(parsed.charts),
            parsed.music_file,
        )
        for c in parsed.charts:
            logger.info(
                "    %s L%d steps=%d holds=%d",
                c.difficulty_name,
                c.difficulty_level,
                len(c.steps),
                sum(1 for s in c.steps if s.step_type.value == "hold"),
            )
        return True

    upload_assets(song_dir, slug, storage)

    music_key = f"{slug}/{parsed.music_file}"
    if not storage.exists(music_key):
        # .sm sometimes declares a wrong extension (e.g. .wav when only .ogg
        # was shipped). Fall back to any audio file in the source folder.
        candidates = [
            p for p in song_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS
        ]
        if not candidates:
            logger.error("No audio file found for %s in %s", parsed.title, song_dir)
            return False
        fallback = candidates[0]
        fallback_key = f"{slug}/{fallback.name}"
        upload_asset(fallback, fallback_key, storage)
        parsed.music_file = fallback.name
        music_key = fallback_key
        logger.warning(
            "#MUSIC pointed at missing file; falling back to %s", fallback.name
        )

    music_local = storage.reserve_path(music_key)
    try:
        duration = float(librosa.get_duration(path=str(music_local)))
    except Exception as e:
        logger.warning(
            "Could not probe duration for %s (%s); falling back to 0", music_local, e
        )
        duration = 0.0

    banner_relpath = key_or_none(parsed.banner, slug, storage)
    bg_relpath = key_or_none(parsed.background, slug, storage)
    jacket_relpath = key_or_none(parsed.jacket, slug, storage)
    if jacket_relpath is None:
        # Common DDR pack convention: "<song>-jacket.png" not declared in #JACKET
        for f in song_dir.glob("*-jacket.*"):
            candidate_key = f"{slug}/{f.name}"
            if storage.exists(candidate_key):
                jacket_relpath = candidate_key
                break

    await persist_ddr_song(
        session,
        parsed=parsed,
        slug=slug,
        audio_relpath=music_key,
        banner_relpath=banner_relpath,
        jacket_relpath=jacket_relpath,
        bg_relpath=bg_relpath,
        duration=duration,
        ddr_game=game,
        ddr_release_index=game_index,
    )
    return True


async def seed_game(
    session,
    root: Path,
    game: str,
    storage: StorageBackend,
    limit: Optional[int],
    dry_run: bool,
) -> int:
    game_dir = root / game
    if not game_dir.is_dir():
        logger.error("Game folder not found: %s", game_dir)
        return 0
    game_index = parse_game_index(game)
    song_dirs = sorted(d for d in game_dir.iterdir() if d.is_dir())
    if limit is not None:
        song_dirs = song_dirs[:limit]
    count = 0
    for song_dir in tqdm(song_dirs, desc=game, unit="song"):
        ok = await seed_one_song(
            session, song_dir, game, game_index, storage, dry_run
        )
        if ok:
            count += 1
    return count


async def run(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.is_dir():
        logger.error("DDR root not found: %s", root)
        return 1

    storage = get_storage()

    games: list[str]
    if args.game == "all":
        games = sorted(d.name for d in root.iterdir() if d.is_dir())
    else:
        games = [args.game]

    total = 0
    if args.dry_run:
        for game in games:
            total += await seed_game(None, root, game, storage, args.limit, True)
        logger.info("✓ Dry run complete. Songs seen: %d", total)
        return 0

    async with AsyncSessionLocal() as session:
        for game in games:
            try:
                total += await seed_game(
                    session, root, game, storage, args.limit, False
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    logger.info("✓ Seed complete. Songs persisted: %d", total)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Postgres from DDR .sm packs")
    parser.add_argument("--game", default=DEFAULT_GAME,
                        help="DDR game folder (e.g. '[01] DDR 1st Mix') or 'all'")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="Path to the DDR root folder")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit songs per game")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse only; no DB writes, no asset upload")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())

"""Seed Postgres from the DDR 1st-A20 Plus song collection.

Usage:
    uv run python -m scripts.seed_ddr [--game "[01] DDR 1st Mix" | --game all]
                                       [--root "DDR 1st-A20 Plus"]
                                       [--limit N]
                                       [--dry-run]

Idempotent: re-running updates rows in place via ON CONFLICT and skips
asset copies that already exist.
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

from core.config import settings
from db.session import AsyncSessionLocal
from services.chart_persistence import persist_ddr_song
from services.sm_parser import ParsedSong, parse_sm

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "DDR 1st-A20 Plus"
DEFAULT_GAME = "[01] DDR 1st Mix"
ASSET_EXTS = {".ogg", ".mp3", ".wav", ".png", ".jpg", ".jpeg"}

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


def copy_assets(song_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in song_dir.iterdir():
        if not src.is_file():
            continue
        if src.suffix.lower() not in ASSET_EXTS:
            continue
        dst = dest_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def relpath_or_none(filename: Optional[str], dest_dir: Path) -> Optional[str]:
    """Return the path relative to AUDIO_STORAGE_PATH if the file exists in dest_dir."""
    if not filename:
        return None
    candidate = dest_dir / filename
    if not candidate.exists():
        return None
    storage_root = Path(settings.AUDIO_STORAGE_PATH).resolve()
    try:
        return str(candidate.resolve().relative_to(storage_root))
    except ValueError:
        return None


async def seed_one_song(
    session,
    song_dir: Path,
    game: str,
    game_index: Optional[int],
    storage_root: Path,
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
    dest_dir = storage_root / "ddr" / slug

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

    copy_assets(song_dir, dest_dir)

    music_path = dest_dir / parsed.music_file
    if not music_path.exists():
        # .sm sometimes declares a wrong extension (e.g. .wav when only .ogg
        # was shipped). Fall back to any audio file in the dest folder.
        audio_exts = (".ogg", ".mp3", ".wav", ".flac", ".m4a")
        candidates = [p for p in dest_dir.iterdir() if p.suffix.lower() in audio_exts]
        if not candidates:
            logger.error("No audio file found for %s in %s", parsed.title, dest_dir)
            return False
        music_path = candidates[0]
        parsed.music_file = music_path.name
        logger.warning(
            "#MUSIC pointed at missing file; falling back to %s", music_path.name
        )

    try:
        duration = float(librosa.get_duration(path=str(music_path)))
    except Exception as e:
        logger.warning("Could not probe duration for %s (%s); falling back to 0", music_path, e)
        duration = 0.0

    audio_relpath = f"ddr/{slug}/{parsed.music_file}"
    banner_relpath = relpath_or_none(parsed.banner, dest_dir)
    bg_relpath = relpath_or_none(parsed.background, dest_dir)
    jacket_relpath = relpath_or_none(parsed.jacket, dest_dir)
    if jacket_relpath is None:
        # Common DDR pack convention: "<song>-jacket.png" not declared in #JACKET
        for f in dest_dir.glob("*-jacket.*"):
            jacket_relpath = f"ddr/{slug}/{f.name}"
            break

    await persist_ddr_song(
        session,
        parsed=parsed,
        slug=slug,
        audio_relpath=audio_relpath,
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
    storage_root: Path,
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
            session, song_dir, game, game_index, storage_root, dry_run
        )
        if ok:
            count += 1
    return count


async def run(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.is_dir():
        logger.error("DDR root not found: %s", root)
        return 1

    storage_root = Path(settings.AUDIO_STORAGE_PATH).resolve()
    storage_root.mkdir(parents=True, exist_ok=True)

    games: list[str]
    if args.game == "all":
        games = sorted(d.name for d in root.iterdir() if d.is_dir())
    else:
        games = [args.game]

    total = 0
    if args.dry_run:
        # No DB needed for dry run.
        class _NoSession:
            async def __aenter__(self): return None
            async def __aexit__(self, *a): return False
        for game in games:
            total += await seed_game(None, root, game, storage_root, args.limit, True)
        logger.info("✓ Dry run complete. Songs seen: %d", total)
        return 0

    async with AsyncSessionLocal() as session:
        for game in games:
            try:
                total += await seed_game(
                    session, root, game, storage_root, args.limit, False
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
                        help="Parse only; no DB writes, no asset copy")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())

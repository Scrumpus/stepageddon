"""Build a playlist out of every DDR song from a given game folder.

The DDR songs must already be seeded (run ``scripts.seed_ddr`` first). This
script finds-or-creates a playlist by name and replaces its membership with
the full set of songs whose ``ddr_game`` matches.

Usage:
    uv run python -m scripts.seed_playlist_from_game --game "[01] DDR 1st Mix"
    uv run python -m scripts.seed_playlist_from_game --game "[01] DDR 1st Mix" --name "Konami Classics"

Idempotent: re-runs reset the membership to match current DB state.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Optional

from sqlalchemy import delete, select

from src.database import AsyncSessionLocal
from src.playlists.models import Playlist, PlaylistSong
from src.songs.models import Song, SongSource

logger = logging.getLogger(__name__)


async def build_playlist_for_game(
    session,
    *,
    game: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    log_songs: bool = False,
) -> Optional[Playlist]:
    """Find-or-create a playlist named after the game and replace membership
    with every DDR song whose ``ddr_game`` matches.

    Caller commits. Returns the Playlist (or None if no songs were found).
    """
    playlist_name = name or game

    songs = list(
        (
            await session.execute(
                select(Song)
                .where(Song.source == SongSource.DDR, Song.ddr_game == game)
                .order_by(Song.title)
            )
        ).scalars()
    )
    if not songs:
        logger.warning(
            "No DDR songs found for game=%r — run scripts.seed_ddr first", game
        )
        return None

    playlist = (
        await session.execute(select(Playlist).where(Playlist.name == playlist_name))
    ).scalar_one_or_none()
    if playlist is None:
        playlist = Playlist(name=playlist_name, description=description or game)
        session.add(playlist)
        await session.flush()
        logger.info("Created playlist %s (%s)", playlist_name, playlist.id)
    else:
        if description is not None and description != playlist.description:
            playlist.description = description
        logger.info("Reusing playlist %s (%s)", playlist_name, playlist.id)

    await session.execute(
        delete(PlaylistSong).where(PlaylistSong.playlist_id == playlist.id)
    )
    await session.flush()

    for position, song in enumerate(songs):
        session.add(
            PlaylistSong(
                playlist_id=playlist.id,
                song_id=song.id,
                position=position,
            )
        )

    logger.info("✓ %s: %d songs", playlist_name, len(songs))
    if log_songs:
        for s in songs:
            logger.info("  %s — %s", s.title, s.artist or "Unknown")
    return playlist


async def run(game: str, name: Optional[str], description: Optional[str]) -> int:
    async with AsyncSessionLocal() as session:
        playlist = await build_playlist_for_game(
            session,
            game=game,
            name=name,
            description=description,
            log_songs=True,
        )
        if playlist is None:
            return 1
        await session.commit()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a playlist from every song in a DDR game folder"
    )
    parser.add_argument("--game", required=True, help='e.g. "[01] DDR 1st Mix"')
    parser.add_argument("--name", default=None,
                        help="Playlist name (defaults to --game)")
    parser.add_argument("--description", default=None,
                        help="Playlist description (defaults to game name)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args.game, args.name, args.description))


if __name__ == "__main__":
    sys.exit(main())

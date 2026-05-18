"""YouTube Music search via the unofficial ytmusicapi.

We only use the search endpoint — playback still goes through the existing
yt-dlp pipeline since YouTube Music videoIds resolve to normal youtube.com
watch URLs.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ytmusicapi import YTMusic

from src.ytmusic.schemas import SongSearchResult

logger = logging.getLogger(__name__)


def _pick_thumbnail(thumbnails: Optional[list]) -> str:
    """Return the largest thumbnail URL from a ytmusicapi `thumbnails` array."""
    if not thumbnails:
        return ""
    # ytmusicapi returns thumbnails sorted small → large.
    return thumbnails[-1].get("url", "")


def _parse_duration(raw: Optional[str]) -> Optional[int]:
    """Convert a "M:SS" or "H:MM:SS" string from ytmusicapi to seconds."""
    if not raw:
        return None
    parts = raw.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    seconds = 0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


def _join_artists(artists: Optional[list]) -> str:
    if not artists:
        return "Unknown"
    names = [a.get("name") for a in artists if a.get("name")]
    return ", ".join(names) if names else "Unknown"


class YTMusicClient:
    """Thin wrapper around ytmusicapi for the search-songs endpoint."""

    def __init__(self) -> None:
        self._ytm: Optional[YTMusic] = None
        try:
            self._ytm = YTMusic()
            logger.info("✓ YouTube Music client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize YouTube Music client: {e}")

    def search(self, query: str, limit: int = 10) -> List[SongSearchResult]:
        """Run a song search and return mapped results.

        Raises ``RuntimeError`` if the client couldn't be initialized.
        Any exception from ytmusicapi propagates — callers should treat
        that as a 502.
        """
        if self._ytm is None:
            raise RuntimeError("YouTube Music client not initialized")

        raw = self._ytm.search(query, filter="songs", limit=limit) or []

        results: List[SongSearchResult] = []
        for item in raw[:limit]:
            video_id = item.get("videoId")
            if not video_id:
                # Songs filter occasionally returns rows without a videoId
                # (e.g. album/artist headers). Skip them.
                continue
            album_obj = item.get("album") or {}
            results.append(
                SongSearchResult(
                    videoId=video_id,
                    title=item.get("title", "Unknown"),
                    artist=_join_artists(item.get("artists")),
                    album=album_obj.get("name"),
                    duration_seconds=_parse_duration(item.get("duration")),
                    thumbnail=_pick_thumbnail(item.get("thumbnails")),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                )
            )
        return results

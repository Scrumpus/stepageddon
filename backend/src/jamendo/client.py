"""Jamendo audio search + full-track download via the v3.0 REST API.

Jamendo serves a large Creative-Commons / royalty-free catalog and returns
direct MP3 URLs. It requires a free ``client_id`` (devportal.jamendo.com); when
that's unset the client degrades gracefully — search returns nothing and
download raises — so the app still boots without it (mirrors the old Spotify
client's optional-credentials behaviour).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import aiohttp

from src.config import settings
from src.search.schemas import SongSearchResult

logger = logging.getLogger(__name__)

_API_BASE = "https://api.jamendo.com/v3.0"
_TIMEOUT = aiohttp.ClientTimeout(total=30)
# Share URLs look like https://www.jamendo.com/track/{id}/{slug}
_TRACK_ID_RE = re.compile(r"/track/(\d+)")


class JamendoClient:
    """Thin async wrapper over the Jamendo /tracks endpoint."""

    def __init__(self) -> None:
        self.client_id = settings.JAMENDO_CLIENT_ID
        if self.client_id:
            logger.info("✓ Jamendo client configured")
        else:
            logger.warning(
                "Jamendo client_id not set — Jamendo search/links disabled"
            )

    @property
    def enabled(self) -> bool:
        return bool(self.client_id)

    def _map_track(self, t: dict) -> Optional[SongSearchResult]:
        track_id = t.get("id")
        share_url = t.get("shareurl")
        if not track_id or not share_url:
            return None
        duration = t.get("duration")
        return SongSearchResult(
            id=str(track_id),
            source="jamendo",
            title=t.get("name", "Unknown"),
            artist=t.get("artist_name") or "Unknown",
            album=t.get("album_name"),
            duration_seconds=int(duration) if duration else None,
            thumbnail=t.get("image") or t.get("album_image") or "",
            url=share_url,
        )

    async def _get_tracks(
        self, session: aiohttp.ClientSession, params: dict
    ) -> list:
        full_params = {"client_id": self.client_id, "format": "json", **params}
        async with session.get(
            f"{_API_BASE}/tracks/", params=full_params
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return payload.get("results") or []

    async def search(self, query: str, limit: int = 10) -> List[SongSearchResult]:
        if not self.enabled:
            return []
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            rows = await self._get_tracks(
                session, {"search": query, "limit": limit}
            )
        results: List[SongSearchResult] = []
        for t in rows:
            mapped = self._map_track(t)
            if mapped is not None:
                results.append(mapped)
        return results

    @staticmethod
    def extract_track_id(url: str) -> str:
        m = _TRACK_ID_RE.search(url)
        if not m:
            raise ValueError("Invalid Jamendo URL format")
        return m.group(1)

    async def download(self, url: str, output_path: str) -> Dict:
        """Look up a Jamendo track by URL and stream its MP3 to disk.

        Returns ``{"file_path", "metadata"}``. Prefers the download URL when the
        artist allows it, else falls back to the stream URL.
        """
        if not self.enabled:
            raise ValueError("Jamendo is not configured.")
        logger.info(f"Downloading from Jamendo: {url}")

        track_id = self.extract_track_id(url)
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            rows = await self._get_tracks(session, {"id": track_id})
            if not rows:
                raise ValueError("Jamendo track not found.")
            t = rows[0]

            duration = t.get("duration")
            metadata = {
                "title": t.get("name", "Unknown"),
                "artist": t.get("artist_name") or "Unknown",
                "duration": int(duration) if duration else 0,
                "thumbnail": t.get("image") or t.get("album_image") or "",
                "source": "jamendo",
            }

            audio_url = (
                t.get("audiodownload")
                if t.get("audiodownload_allowed") and t.get("audiodownload")
                else t.get("audio")
            )
            if not audio_url:
                raise ValueError("Jamendo track has no playable audio URL.")

            async with session.get(audio_url) as resp:
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        f.write(chunk)

        logger.info(f"✓ Downloaded: {metadata['title']}")
        return {"file_path": output_path, "metadata": metadata}

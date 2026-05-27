"""Audius audio search + full-track download via the public REST API.

Audius is an open music protocol; its catalog is streamable without an API key
— requests only need an ``app_name`` query param. For downloads we resolve a
track's ``audius.co`` permalink to its id, then stream the full MP3 from the
network.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import aiohttp

from src.config import settings
from src.search.schemas import SongSearchResult

logger = logging.getLogger(__name__)

# Stable entry point — it routes requests to a healthy discovery node for us,
# so we don't have to do manual host discovery against https://api.audius.co.
_API_BASE = "https://api.audius.co/v1"
_HEADERS = {"Accept": "application/json"}
_TIMEOUT = aiohttp.ClientTimeout(total=30)


def _pick_artwork(artwork: Optional[dict]) -> str:
    """Largest available square artwork URL from an Audius `artwork` object."""
    if not artwork:
        return ""
    for size in ("1000x1000", "480x480", "150x150"):
        if artwork.get(size):
            return artwork[size]
    return ""


class AudiusClient:
    """Thin async wrapper over the Audius tracks search + stream endpoints.

    A fresh ``aiohttp`` session is opened per call (rather than held on the
    instance) so the client isn't bound to a particular event loop — matching
    how the old Spotify client fetched previews.
    """

    def __init__(self) -> None:
        self.app_name = settings.AUDIUS_APP_NAME or "stepageddon"
        logger.info(f"✓ Audius client ready (app_name={self.app_name})")

    def _map_track(self, t: dict) -> Optional[SongSearchResult]:
        track_id = t.get("id")
        permalink = t.get("permalink")
        if not track_id or not permalink:
            return None
        user = t.get("user") or {}
        return SongSearchResult(
            id=str(track_id),
            source="audius",
            title=t.get("title", "Unknown"),
            artist=user.get("name") or "Unknown",
            album=None,
            duration_seconds=t.get("duration"),
            thumbnail=_pick_artwork(t.get("artwork")),
            url=f"https://audius.co{permalink}",
        )

    async def search(self, query: str, limit: int = 10) -> List[SongSearchResult]:
        params = {"query": query, "app_name": self.app_name}
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            async with session.get(
                f"{_API_BASE}/tracks/search", params=params, headers=_HEADERS
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()

        results: List[SongSearchResult] = []
        for t in payload.get("data") or []:
            if t.get("is_streamable") is False:
                continue
            mapped = self._map_track(t)
            if mapped is not None:
                results.append(mapped)
            if len(results) >= limit:
                break
        return results

    async def download(self, url: str, output_path: str) -> Dict:
        """Resolve an audius.co URL to a track and stream its MP3 to disk.

        Returns ``{"file_path", "metadata"}`` — full tracks, no previews.
        """
        logger.info(f"Downloading from Audius: {url}")
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            track = await self._resolve_track(session, url)
            track_id = track.get("id")
            if not track_id:
                raise ValueError("Could not resolve Audius track from URL.")

            user = track.get("user") or {}
            metadata = {
                "title": track.get("title", "Unknown"),
                "artist": user.get("name") or "Unknown",
                "duration": track.get("duration", 0) or 0,
                "thumbnail": _pick_artwork(track.get("artwork")),
                "source": "audius",
            }

            stream_url = f"{_API_BASE}/tracks/{track_id}/stream"
            # The stream endpoint 302-redirects to the content node; aiohttp
            # follows it by default and we read the MP3 bytes off the response.
            async with session.get(
                stream_url, params={"app_name": self.app_name}
            ) as resp:
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        f.write(chunk)

        logger.info(f"✓ Downloaded: {metadata['title']}")
        return {"file_path": output_path, "metadata": metadata}

    async def _resolve_track(
        self, session: aiohttp.ClientSession, url: str
    ) -> dict:
        """Use /v1/resolve to turn an audius.co URL into a track object."""
        try:
            async with session.get(
                f"{_API_BASE}/resolve",
                params={"url": url, "app_name": self.app_name},
                headers=_HEADERS,
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        except Exception as e:
            logger.error(f"Audius resolve failed: {e}")
            raise ValueError(f"Could not resolve Audius URL: {e}")

        data = payload.get("data")
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict):
            raise ValueError("Audius URL did not resolve to a track.")
        return data

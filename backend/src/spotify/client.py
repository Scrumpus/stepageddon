"""Spotify metadata + 30-second preview download.

Spotify never exposes full audio via its API — we resolve track metadata
through the official client and (if available) pull the short preview MP3 to
feed the generation pipeline. Callers should branch on the ``is_preview``
flag in the result.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from src.config import settings

logger = logging.getLogger(__name__)


class SpotifyClient:
    """Wrap spotipy with our credentials + a preview-download helper."""

    def __init__(self):
        self.client: Optional[spotipy.Spotify] = None
        self._init_client()

    def _init_client(self) -> None:
        if settings.SPOTIFY_CLIENT_ID and settings.SPOTIFY_CLIENT_SECRET:
            try:
                auth_manager = SpotifyClientCredentials(
                    client_id=settings.SPOTIFY_CLIENT_ID,
                    client_secret=settings.SPOTIFY_CLIENT_SECRET,
                )
                self.client = spotipy.Spotify(auth_manager=auth_manager)
                logger.info("✓ Spotify client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Spotify: {e}")

    @staticmethod
    def extract_track_id(url: str) -> str:
        """Extract track ID from an open.spotify.com URL or spotify: URI."""
        if "open.spotify.com/track/" in url:
            return url.split("track/")[1].split("?")[0]
        if "spotify:track:" in url:
            return url.split("spotify:track:")[1]
        raise ValueError("Invalid Spotify URL format")

    async def download_preview(self, url: str, output_path: str) -> Dict:
        """Download the 30s preview for a Spotify track URL.

        Returns ``{"file_path", "metadata", "is_preview": True}``.
        Raises ``ValueError`` if Spotify is not configured or the track has no
        preview_url.
        """
        if not self.client:
            raise ValueError("Spotify is not configured. Please add Spotify credentials.")

        logger.info(f"Fetching Spotify metadata: {url}")

        try:
            track_id = self.extract_track_id(url)
            track = self.client.track(track_id)

            metadata = {
                "title": track["name"],
                "artist": ", ".join([artist["name"] for artist in track["artists"]]),
                "duration": track["duration_ms"] / 1000,
                "thumbnail": track["album"]["images"][0]["url"] if track["album"]["images"] else "",
                "source": "spotify",
                "spotify_id": track_id,
            }

            preview_url = track.get("preview_url")
            if not preview_url:
                raise ValueError(
                    "This Spotify track doesn't have a preview available. "
                    "Please try another song or use YouTube."
                )

            logger.info("Downloading Spotify preview...")
            await self._fetch_preview(preview_url, output_path)

            return {
                "file_path": output_path,
                "metadata": metadata,
                "is_preview": True,
            }
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Spotify handling failed: {e}")
            raise ValueError(f"Could not process Spotify URL: {str(e)}")

    async def _fetch_preview(self, preview_url: str, output_path: str) -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(preview_url) as response:
                if response.status == 200:
                    with open(output_path, 'wb') as f:
                        f.write(await response.read())
                else:
                    raise ValueError("Failed to download preview")

    def get_metadata(self, url: str) -> Optional[Dict]:
        """Lightweight metadata-only lookup (no audio fetch)."""
        if not self.client:
            return None
        try:
            track_id = self.extract_track_id(url)
            track = self.client.track(track_id)
            return {
                "title": track["name"],
                "duration": track["duration_ms"] / 1000,
                "source": "spotify",
            }
        except Exception as e:
            logger.warning(f"Could not fetch Spotify metadata: {e}")
            return None

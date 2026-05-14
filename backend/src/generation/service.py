"""Generation orchestration helpers shared across router endpoints."""

from __future__ import annotations

import logging
from typing import Dict

from src.spotify.client import SpotifyClient
from src.youtube.client import YouTubeClient

logger = logging.getLogger(__name__)


async def download_audio(
    url: str,
    output_path: str,
    *,
    spotify_client: SpotifyClient,
    youtube_client: YouTubeClient,
) -> Dict:
    """Dispatch to the right external client based on URL host.

    Returns the same shape both clients produce:
    ``{"file_path", "metadata", "is_preview"?: bool}``.
    """
    try:
        if "youtube.com" in url or "youtu.be" in url:
            return await youtube_client.download(url, output_path)
        if "spotify.com" in url:
            return await spotify_client.download_preview(url, output_path)
        raise ValueError("Unsupported URL. Please use YouTube or Spotify links.")
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        raise

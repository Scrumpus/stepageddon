"""Generation orchestration helpers shared across router endpoints."""

from __future__ import annotations

import logging
from typing import Dict

from src.audius.client import AudiusClient
from src.jamendo.client import JamendoClient

logger = logging.getLogger(__name__)


async def download_audio(
    url: str,
    output_path: str,
    *,
    audius_client: AudiusClient,
    jamendo_client: JamendoClient,
) -> Dict:
    """Dispatch to the right source client based on URL host.

    Both clients return the same shape: ``{"file_path", "metadata"}`` (full
    tracks — no previews).
    """
    try:
        if "audius.co" in url:
            return await audius_client.download(url, output_path)
        if "jamendo.com" in url:
            return await jamendo_client.download(url, output_path)
        raise ValueError("Unsupported URL. Please use Audius or Jamendo links.")
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        raise

"""YouTube audio download via yt-dlp."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yt_dlp

logger = logging.getLogger(__name__)

# backend/src/youtube/client.py → backend/
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_COOKIES_PATH = _BACKEND_DIR / "cookies.txt"


class YouTubeClient:
    """yt-dlp wrapper that downloads + transcodes to MP3."""

    async def download(self, url: str, output_path: str) -> Dict:
        """Download audio from a YouTube URL to ``output_path``.

        Returns ``{"file_path", "metadata"}`` where ``metadata`` contains
        title, artist (uploader), duration, thumbnail, and ``source="youtube"``.
        """
        logger.info(f"Downloading from YouTube: {url}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_path.replace('.mp3', ''),
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {'youtube': {'player_client': ['android_vr']}},
        }

        if os.path.exists(_COOKIES_PATH):
            ydl_opts['cookiefile'] = str(_COOKIES_PATH)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                metadata = {
                    "title": info.get("title", "Unknown"),
                    "artist": info.get("uploader", "Unknown"),
                    "duration": info.get("duration", 0),
                    "thumbnail": info.get("thumbnail", ""),
                    "source": "youtube",
                }

                logger.info(f"✓ Downloaded: {metadata['title']}")
                return {
                    "file_path": output_path,
                    "metadata": metadata,
                }
        except Exception as e:
            logger.error(f"YouTube download failed: {e}")
            raise ValueError(f"Could not download from YouTube: {str(e)}")

    def get_metadata(self, url: str) -> Optional[Dict]:
        """Lightweight metadata-only lookup (no audio fetch)."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "source": "youtube",
                }
        except Exception as e:
            logger.warning(f"Could not fetch YouTube metadata: {e}")
            return None

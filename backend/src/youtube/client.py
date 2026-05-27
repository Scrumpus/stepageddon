"""YouTube audio download via yt-dlp."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

import yt_dlp

from src.config import settings

logger = logging.getLogger(__name__)

# backend/src/youtube/client.py → backend/
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_COOKIES_PATH = _BACKEND_DIR / "cookies.txt"


def _resolve_cookies_path() -> Optional[Path]:
    """Cookies file to use, or None. YT_COOKIES_PATH wins; else the bundled file."""
    if settings.YT_COOKIES_PATH:
        path = Path(settings.YT_COOKIES_PATH).expanduser()
        if path.exists():
            return path
        logger.warning(f"YT_COOKIES_PATH set but not found: {path}")
        return None
    return _DEFAULT_COOKIES_PATH if _DEFAULT_COOKIES_PATH.exists() else None


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

        # yt-dlp rewrites the cookie jar back to ``cookiefile`` when the session
        # ends. The source is often mounted read-only (or sits on a read-only
        # rootfs), so hand yt-dlp a writable copy and discard it after.
        cookies_tmp: Optional[str] = None
        cookies_path = _resolve_cookies_path()
        if cookies_path is not None:
            fd, cookies_tmp = tempfile.mkstemp(prefix="ytcookies_", suffix=".txt")
            os.close(fd)
            shutil.copyfile(cookies_path, cookies_tmp)
            ydl_opts['cookiefile'] = cookies_tmp

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
        finally:
            if cookies_tmp and os.path.exists(cookies_tmp):
                os.remove(cookies_tmp)

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

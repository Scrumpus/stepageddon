"""
Audio Downloader Service
Handles downloading audio from YouTube and Spotify
"""

import os
import logging
from typing import Dict, Optional
import yt_dlp
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from core.config import settings

logger = logging.getLogger(__name__)


class AudioDownloader:
    """Download audio from various sources"""
    
    def __init__(self):
        self.spotify_client = None
        self._init_spotify()
    
    def _init_spotify(self):
        """Initialize Spotify client"""
        if settings.SPOTIFY_CLIENT_ID and settings.SPOTIFY_CLIENT_SECRET:
            try:
                auth_manager = SpotifyClientCredentials(
                    client_id=settings.SPOTIFY_CLIENT_ID,
                    client_secret=settings.SPOTIFY_CLIENT_SECRET
                )
                self.spotify_client = spotipy.Spotify(auth_manager=auth_manager)
                logger.info("✓ Spotify client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Spotify: {e}")
    
    async def download_from_url(self, url: str, output_path: str) -> Dict:
        """
        Download audio from URL (YouTube or Spotify)
        
        Args:
            url: YouTube or Spotify URL
            output_path: Where to save the audio file
            
        Returns:
            Dictionary with file info and metadata
        """
        try:
            if "youtube.com" in url or "youtu.be" in url:
                return await self._download_youtube(url, output_path)
            elif "spotify.com" in url:
                return await self._handle_spotify(url, output_path)
            else:
                raise ValueError("Unsupported URL. Please use YouTube or Spotify links.")
                
        except Exception as e:
            logger.error(f"Download failed: {e}", exc_info=True)
            raise
    
    async def _download_youtube(self, url: str, output_path: str) -> Dict:
        """Download audio from YouTube"""
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
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                metadata = {
                    "title": info.get("title", "Unknown"),
                    "artist": info.get("uploader", "Unknown"),
                    "duration": info.get("duration", 0),
                    "thumbnail": info.get("thumbnail", ""),
                    "source": "youtube"
                }
                
                logger.info(f"✓ Downloaded: {metadata['title']}")
                return {
                    "file_path": output_path,
                    "metadata": metadata
                }
                
        except Exception as e:
            logger.error(f"YouTube download failed: {e}")
            raise ValueError(f"Could not download from YouTube: {str(e)}")
    
    async def _handle_spotify(self, url: str, output_path: str) -> Dict:
        """Handle Spotify URLs"""
        if not self.spotify_client:
            raise ValueError("Spotify is not configured. Please add Spotify credentials.")
        
        logger.info(f"Fetching Spotify metadata: {url}")
        
        try:
            # Extract track ID from URL
            track_id = self._extract_spotify_id(url)
            
            # Get track info
            track = self.spotify_client.track(track_id)
            
            metadata = {
                "title": track["name"],
                "artist": ", ".join([artist["name"] for artist in track["artists"]]),
                "duration": track["duration_ms"] / 1000,
                "thumbnail": track["album"]["images"][0]["url"] if track["album"]["images"] else "",
                "source": "spotify",
                "spotify_id": track_id
            }
            
            # Check if preview is available
            preview_url = track.get("preview_url")
            
            if preview_url:
                # Download preview (30s clip)
                logger.info("Downloading Spotify preview...")
                await self._download_preview(preview_url, output_path)
                
                return {
                    "file_path": output_path,
                    "metadata": metadata,
                    "is_preview": True
                }
            else:
                # No preview available
                raise ValueError(
                    "This Spotify track doesn't have a preview available. "
                    "Please try another song or use YouTube."
                )
                
        except Exception as e:
            logger.error(f"Spotify handling failed: {e}")
            raise ValueError(f"Could not process Spotify URL: {str(e)}")
    
    def _extract_spotify_id(self, url: str) -> str:
        """Extract track ID from Spotify URL"""
        # Handle both open.spotify.com and spotify: URIs
        if "open.spotify.com/track/" in url:
            return url.split("track/")[1].split("?")[0]
        elif "spotify:track:" in url:
            return url.split("spotify:track:")[1]
        else:
            raise ValueError("Invalid Spotify URL format")
    
    async def _download_preview(self, preview_url: str, output_path: str):
        """Download Spotify preview URL"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(preview_url) as response:
                if response.status == 200:
                    with open(output_path, 'wb') as f:
                        f.write(await response.read())
                else:
                    raise ValueError("Failed to download preview")
    
    def get_metadata_only(self, url: str) -> Optional[Dict]:
        """Get metadata without downloading (for validation)"""
        try:
            if "youtube.com" in url or "youtu.be" in url:
                ydl_opts = {'quiet': True, 'no_warnings': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    return {
                        "title": info.get("title", "Unknown"),
                        "duration": info.get("duration", 0),
                        "source": "youtube"
                    }
            
            elif "spotify.com" in url and self.spotify_client:
                track_id = self._extract_spotify_id(url)
                track = self.spotify_client.track(track_id)
                return {
                    "title": track["name"],
                    "duration": track["duration_ms"] / 1000,
                    "source": "spotify"
                }
        
        except Exception as e:
            logger.warning(f"Could not fetch metadata: {e}")
            return None

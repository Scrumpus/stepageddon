"""Canonical genre list with per-provider translations for discovery.

Audius expects genre *names* (e.g. ``Hip-Hop/Rap``); Jamendo expects lowercase
*tags* (e.g. ``hiphop``). The frontend works in stable ``key``s and shows
``label``s; this module is the single source of truth mapping a key to each
provider so the two never drift.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class Genre(TypedDict):
    key: str
    label: str
    audius: str
    jamendo: str


# Order here is the order shown in the UI chips.
CANONICAL: List[Genre] = [
    {"key": "electronic", "label": "Electronic", "audius": "Electronic", "jamendo": "electronic"},
    {"key": "hiphop", "label": "Hip-Hop", "audius": "Hip-Hop/Rap", "jamendo": "hiphop"},
    {"key": "rock", "label": "Rock", "audius": "Rock", "jamendo": "rock"},
    {"key": "pop", "label": "Pop", "audius": "Pop", "jamendo": "pop"},
    {"key": "jazz", "label": "Jazz", "audius": "Jazz", "jamendo": "jazz"},
    {"key": "classical", "label": "Classical", "audius": "Classical", "jamendo": "classical"},
    {"key": "ambient", "label": "Ambient", "audius": "Ambient", "jamendo": "ambient"},
    {"key": "house", "label": "House", "audius": "House", "jamendo": "house"},
]

_BY_KEY = {g["key"]: g for g in CANONICAL}


def list_genres() -> List[dict]:
    """Public genre list for the frontend chips — just ``key`` + ``label``."""
    return [{"key": g["key"], "label": g["label"]} for g in CANONICAL]


def audius_genre(key: Optional[str]) -> Optional[str]:
    """Audius genre name for a canonical key, or None (→ no genre filter)."""
    g = _BY_KEY.get(key) if key else None
    return g["audius"] if g else None


def jamendo_tag(key: Optional[str]) -> Optional[str]:
    """Jamendo tag for a canonical key, or None (→ no tag filter)."""
    g = _BY_KEY.get(key) if key else None
    return g["jamendo"] if g else None

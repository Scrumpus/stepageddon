"""SQLAlchemy 2.0 ORM models for songs and charts."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class SongSource(str, enum.Enum):
    USER = "USER"
    DDR = "DDR"


song_source_enum = ENUM(
    SongSource,
    name="songsource",
    values_callable=lambda x: [m.value for m in x],
    create_type=False,
)


class Song(Base):
    __tablename__ = "songs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    artist: Mapped[Optional[str]] = mapped_column(Text)
    source: Mapped[SongSource] = mapped_column(song_source_enum, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)

    audio_path: Mapped[str] = mapped_column(Text, nullable=False)
    banner_path: Mapped[Optional[str]] = mapped_column(Text)
    jacket_path: Mapped[Optional[str]] = mapped_column(Text)
    bg_path: Mapped[Optional[str]] = mapped_column(Text)

    tempo: Mapped[float] = mapped_column(Float, nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    offset: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0"))
    bpms: Mapped[Optional[list]] = mapped_column(JSONB)

    ddr_game: Mapped[Optional[str]] = mapped_column(Text)
    ddr_release_index: Mapped[Optional[int]] = mapped_column(SmallInteger)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    charts: Mapped[List["Chart"]] = relationship(
        back_populates="song",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index(
            "uq_songs_ddr_identity",
            "source",
            "ddr_game",
            "title",
            unique=True,
            postgresql_where=text("source = 'DDR'"),
        ),
        Index("ix_songs_browse", "source", "ddr_release_index", "title"),
        CheckConstraint(
            "(source = 'DDR' AND ddr_game IS NOT NULL) "
            "OR (source = 'USER' AND ddr_game IS NULL)",
            name="ddr_game_matches_source",
        ),
    )


class Playlist(Base):
    __tablename__ = "playlists"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    items: Mapped[List["PlaylistSong"]] = relationship(
        back_populates="playlist",
        cascade="all, delete-orphan",
        order_by="PlaylistSong.position",
    )

    __table_args__ = (Index("ix_playlists_name", "name"),)


class PlaylistSong(Base):
    __tablename__ = "playlist_songs"

    playlist_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("playlists.id", ondelete="CASCADE"),
        primary_key=True,
    )
    song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    playlist: Mapped[Playlist] = relationship(back_populates="items")
    song: Mapped[Song] = relationship()

    __table_args__ = (
        UniqueConstraint(
            "playlist_id", "position", name="uq_playlist_songs_position"
        ),
        Index("ix_playlist_songs_song_id", "song_id"),
    )


class Chart(Base):
    __tablename__ = "charts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        nullable=False,
    )
    difficulty_name: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty_level: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    chart_type: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'dance-single'"),
    )
    steps: Mapped[list] = mapped_column(JSONB, nullable=False)
    step_count: Mapped[int] = mapped_column(Integer, nullable=False)
    hold_count: Mapped[int] = mapped_column(Integer, nullable=False)
    radar_values: Mapped[Optional[list]] = mapped_column(JSONB)
    generator: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )

    song: Mapped[Song] = relationship(back_populates="charts")

    __table_args__ = (
        UniqueConstraint(
            "song_id",
            "difficulty_name",
            "chart_type",
            name="uq_charts_song_difficulty_type",
        ),
        Index("ix_charts_song_id", "song_id"),
    )

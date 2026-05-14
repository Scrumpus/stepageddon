"""SQLAlchemy 2.0 ORM models for songs."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Index,
    SmallInteger,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.charts.models import Chart


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
        "Chart",
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

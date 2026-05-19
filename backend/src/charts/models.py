"""SQLAlchemy 2.0 ORM model for charts."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base

if TYPE_CHECKING:
    from src.songs.models import Song


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

    song: Mapped["Song"] = relationship("Song", back_populates="charts")

    __table_args__ = (
        UniqueConstraint(
            "song_id",
            "difficulty_name",
            "chart_type",
            name="uq_charts_song_difficulty_type",
        ),
        Index("ix_charts_song_id", "song_id"),
    )

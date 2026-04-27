"""init songs and charts tables

Revision ID: 0001_init_songs_charts
Revises:
Create Date: 2026-04-25

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_init_songs_charts"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute(
        "CREATE TYPE songsource AS ENUM ('USER','DDR')"
    )

    op.create_table(
        "songs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("artist", sa.Text(), nullable=True),
        sa.Column(
            "source",
            postgresql.ENUM(
                "USER", "DDR", name="songsource", create_type=False
            ),
            nullable=False,
        ),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("audio_path", sa.Text(), nullable=False),
        sa.Column("banner_path", sa.Text(), nullable=True),
        sa.Column("jacket_path", sa.Text(), nullable=True),
        sa.Column("bg_path", sa.Text(), nullable=True),
        sa.Column("tempo", sa.Float(), nullable=False),
        sa.Column("duration", sa.Float(), nullable=False),
        sa.Column(
            "offset",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("bpms", postgresql.JSONB(), nullable=True),
        sa.Column("ddr_game", sa.Text(), nullable=True),
        sa.Column("ddr_release_index", sa.SmallInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "(source = 'DDR' AND ddr_game IS NOT NULL) "
            "OR (source = 'USER' AND ddr_game IS NULL)",
            name="ck_songs_ddr_game_matches_source",
        ),
    )

    op.create_index(
        "uq_songs_ddr_identity",
        "songs",
        ["source", "ddr_game", "title"],
        unique=True,
        postgresql_where=sa.text("source = 'DDR'"),
    )
    op.create_index(
        "ix_songs_browse",
        "songs",
        ["source", "ddr_release_index", "title"],
    )

    op.create_table(
        "charts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "song_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("songs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("difficulty_name", sa.Text(), nullable=False),
        sa.Column("difficulty_level", sa.SmallInteger(), nullable=False),
        sa.Column(
            "chart_type",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'dance-single'"),
        ),
        sa.Column("steps", postgresql.JSONB(), nullable=False),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("hold_count", sa.Integer(), nullable=False),
        sa.Column("radar_values", postgresql.JSONB(), nullable=True),
        sa.Column("generator", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "song_id",
            "difficulty_name",
            "chart_type",
            name="uq_charts_song_difficulty_type",
        ),
    )
    op.create_index("ix_charts_song_id", "charts", ["song_id"])


def downgrade() -> None:
    op.drop_index("ix_charts_song_id", table_name="charts")
    op.drop_table("charts")
    op.drop_index("ix_songs_browse", table_name="songs")
    op.drop_index("uq_songs_ddr_identity", table_name="songs")
    op.drop_table("songs")
    op.execute("DROP TYPE IF EXISTS songsource")

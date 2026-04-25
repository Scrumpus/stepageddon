"""playlists + playlist_songs

Revision ID: 0002_playlists
Revises: 0001_init_songs_charts
Create Date: 2026-04-25

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_playlists"
down_revision: Union[str, None] = "0001_init_songs_charts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "playlists",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_playlists_name", "playlists", ["name"])

    op.create_table(
        "playlist_songs",
        sa.Column(
            "playlist_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("playlists.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "song_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("songs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column(
            "added_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "playlist_id", "position", name="uq_playlist_songs_position"
        ),
    )
    op.create_index(
        "ix_playlist_songs_song_id", "playlist_songs", ["song_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_playlist_songs_song_id", table_name="playlist_songs")
    op.drop_table("playlist_songs")
    op.drop_index("ix_playlists_name", table_name="playlists")
    op.drop_table("playlists")

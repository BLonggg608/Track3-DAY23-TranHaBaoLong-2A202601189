"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    Supports:
    - "none": No checkpointer
    - "memory": In-memory checkpointer (default)
    - "sqlite": SQLite checkpointer with WAL mode
    - "postgres": PostgreSQL checkpointer (extension)

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Uses WAL mode for better concurrency
    - Database stored at outputs/checkpoints.db
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "Install: pip install langgraph-checkpoint-sqlite"
            ) from exc

        # Create outputs directory if needed
        db_path = Path("outputs") / "checkpoints.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect with WAL mode for better concurrency
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.commit()

        return SqliteSaver(conn=conn)

    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "Install: pip install langgraph-checkpoint-postgres"
            ) from exc

        if not database_url:
            raise ValueError("database_url required for postgres checkpointer")

        import psycopg2

        conn = psycopg2.connect(database_url)
        return PostgresSaver(conn=conn)

    raise ValueError(f"Unknown checkpointer kind: {kind}")

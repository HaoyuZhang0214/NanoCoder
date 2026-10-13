"""LangGraph checkpointer backends for state persistence.

Supports two backends:
- MemorySaver: In-memory storage (default, for testing/development)
- SqliteSaver: SQLite persistence (production, supports resume)

The checkpointer enables automatic state persistence at each graph step,
allowing agents to resume from breakpoints after crashes or interruptions.

Note: This is separate from the file-edit checkpoints in checkpoints.py.
This module is specifically for LangGraph state persistence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def get_checkpointer(backend: str | None = None, db_path: str | None = None) -> BaseCheckpointSaver:
    """Create a LangGraph checkpointer instance based on configuration.
    
    Args:
        backend: Checkpoint backend to use. Options: "memory", "sqlite".
                 If None, reads from NANOCODER_CHECKPOINT_BACKEND env var.
                 Defaults to "memory" if not specified.
        db_path: Path to SQLite database file. Only used when backend="sqlite".
                 If None, reads from NANOCODER_LANGGRAPH_CHECKPOINT_PATH env var.
                 Defaults to ~/.nanocoder/langgraph_checkpoints.db
    
    Returns:
        A LangGraph checkpointer instance
    
    Examples:
        >>> checkpointer = get_checkpointer("memory")
        >>> checkpointer = get_checkpointer("sqlite", "~/.nanocoder/langgraph.db")
    """
    # Resolve backend from env if not specified
    if backend is None:
        backend = os.getenv("NANOCODER_CHECKPOINT_BACKEND", "memory").lower()
    
    if backend == "sqlite":
        # Resolve db_path from env if not specified
        if db_path is None:
            db_path = os.getenv(
                "NANOCODER_LANGGRAPH_CHECKPOINT_PATH",
                str(Path.home() / ".nanocoder" / "langgraph_checkpoints.db")
            )
        # Expand user home directory
        db_path = str(Path(db_path).expanduser().resolve())
        
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            return SqliteSaver.from_conn_string(db_path)
        except ImportError as e:
            raise ImportError(
                "SqliteSaver requires langgraph-checkpoint-sqlite. "
                "Install with: pip install langgraph-checkpoint-sqlite"
            ) from e
    
    # Default to memory backend
    try:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    except ImportError as e:
        raise ImportError(
            "MemorySaver requires langgraph. "
            "Install with: pip install langgraph"
        ) from e


def list_checkpoints(backend: str = "sqlite", db_path: str | None = None) -> list[dict]:
    """List available LangGraph checkpoints (SQLite backend only).
    
    Args:
        backend: Must be "sqlite" for persistent checkpoints
        db_path: Path to SQLite database file
    
    Returns:
        List of checkpoint metadata dicts with keys: thread_id, checkpoint_id
    """
    if backend != "sqlite":
        return []
    
    if db_path is None:
        db_path = os.getenv(
            "NANOCODER_LANGGRAPH_CHECKPOINT_PATH",
            str(Path.home() / ".nanocoder" / "langgraph_checkpoints.db")
        )
    db_path = str(Path(db_path).expanduser().resolve())
    
    if not Path(db_path).exists():
        return []
    
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query checkpoint metadata (LangGraph schema)
        cursor.execute("""
            SELECT DISTINCT thread_id, checkpoint_id 
            FROM checkpoints 
            ORDER BY checkpoint_id DESC
            LIMIT 50
        """)
        
        checkpoints = []
        for thread_id, checkpoint_id in cursor.fetchall():
            checkpoints.append({
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            })
        
        conn.close()
        return checkpoints
    except Exception:
        return []

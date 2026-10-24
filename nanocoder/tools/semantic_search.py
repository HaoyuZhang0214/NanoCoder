"""Semantic + keyword code search over Python AST chunks."""

from pathlib import Path
from typing import ClassVar

from ..retrieval import format_hits, get_index
from .base import Tool


class SemanticSearchTool(Tool):
    """Locate relevant Python functions/classes with hybrid retrieval."""

    name = "semantic_search"
    description = (
        "Locate relevant Python functions/classes with hybrid retrieval "
        "(FAISS semantic search + BM25 keyword search). Use when you do not "
        "know the exact symbol name; then read_file the returned paths for full context."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language or keywords describing the code to find",
            },
            "path": {
                "type": "string",
                "description": "Directory to index and search (default: cwd)",
            },
            "top_k": {
                "type": "integer",
                "description": "How many hits to return (default 8, max 50)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, embedder=None, cache_root: Path | None = None):
        # injectable for tests so search never hits a real embeddings API
        self._embedder = embedder
        self._cache_root = cache_root

    def execute(self, query: str, path: str = ".", top_k: int = 8) -> str:
        if not (query or "").strip():
            return "Error: query must be non-empty"
        root = Path(path).expanduser().resolve()
        if not root.exists():
            return f"Error: {path} not found"
        if root.is_file():
            root = root.parent
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 8
        try:
            index = get_index(
                root,
                embedder=self._embedder,
                cache_root=self._cache_root,
            )
            hits = index.search(query.strip(), top_k=top_k)
        except Exception as e:  # noqa: BLE001 - surface to the model, keep the loop alive
            return f"Error building/searching code index: {e}"
        return format_hits(hits, root)

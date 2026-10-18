"""Hybrid code retrieval: AST chunks + FAISS dense + BM25 sparse + RRF.

Indexes Python sources under a root into function/class chunks, embeds them
with an OpenAI-compatible embeddings API, and fuses FAISS ranks with BM25 via
Reciprocal Rank Fusion. The on-disk cache under ~/.nanocoder/code_index is
incremental: unchanged files keep their vectors.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import pickle
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}
_MAX_EMBED_CHARS = 6000
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
_INDEX_ROOT = Path.home() / ".nanocoder" / "code_index"
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    symbol: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}:{self.symbol}"


def tokenize(text: str) -> list[str]:
    """Split code into identifier tokens, expanding snake_case and CamelCase."""
    tokens: list[str] = []
    for ident in _IDENT_RE.findall(text):
        tokens.append(ident.lower())
        parts = ident.split("_") if "_" in ident else [ident]
        for part in parts:
            if not part:
                continue
            tokens.append(part.lower())
            camel = _CAMEL_RE.findall(part)
            if len(camel) > 1:
                tokens.extend(p.lower() for p in camel)
    return tokens


def chunk_python_source(path: Path, source: str, root: Path) -> list[CodeChunk]:
    """AST-split one file into function/class chunks (methods included)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    rel = str(path.relative_to(root)).replace("\\", "/") if path.is_relative_to(root) else str(path)
    chunks: list[CodeChunk] = []

    def slice_node(node: ast.AST, symbol: str) -> CodeChunk | None:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            return None
        text = "\n".join(lines[start - 1 : end])
        return CodeChunk(path=rel, start_line=start, end_line=end, symbol=symbol, text=text)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            c = slice_node(node, node.name)
            if c:
                chunks.append(c)
        elif isinstance(node, ast.ClassDef):
            c = slice_node(node, node.name)
            if c:
                chunks.append(c)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m = slice_node(child, f"{node.name}.{child.name}")
                    if m:
                        chunks.append(m)

    if not chunks and source.strip():
        chunks.append(CodeChunk(
            path=rel,
            start_line=1,
            end_line=max(1, len(lines)),
            symbol="<module>",
            text=source,
        ))
    return chunks


def iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    root = root.resolve()
    for item in root.rglob("*.py"):
        try:
            rel_parts = item.relative_to(root).parts
        except ValueError:
            continue
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        if item.is_file():
            files.append(item)
        if len(files) >= 5000:
            break
    return files


def chunk_tree(root: Path) -> list[CodeChunk]:
    root = root.resolve()
    chunks: list[CodeChunk] = []
    for path in iter_python_files(root):
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        chunks.extend(chunk_python_source(path, source, root))
    return chunks


class BM25:
    """Minimal Okapi BM25 over an in-memory corpus of token lists."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0
        self.df: Counter[str] = Counter()
        for doc in corpus:
            self.df.update(set(doc))

    def score(self, query: list[str], index: int) -> float:
        if self.N == 0 or self.avgdl == 0:
            return 0.0
        doc = self.corpus[index]
        tf = Counter(doc)
        score = 0.0
        dl = self.doc_len[index]
        for term in query:
            if term not in tf:
                continue
            n_qi = self.df.get(term, 0)
            idf = math.log(1 + (self.N - n_qi + 0.5) / (n_qi + 0.5))
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def top_k(self, query: list[str], k: int) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(self.N)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 0]


def rrf_fuse(
    ranked_lists: list[list[int]],
    k: int = 60,
    top_n: int = 8,
) -> list[int]:
    """Reciprocal Rank Fusion over several ranked id lists."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def default_embedder(texts: list[str]) -> list[list[float]]:
    """OpenAI-compatible embeddings using the same key/base_url as the chat client."""
    from openai import OpenAI

    from .config import Config

    cfg = Config.from_env()
    if not cfg.api_key:
        raise RuntimeError(
            "No API key for embeddings. Set OPENAI_API_KEY or NANOCODER_API_KEY."
        )
    model = os.getenv("NANOCODER_EMBED_MODEL", _DEFAULT_EMBED_MODEL)
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    # truncate before sending; keep order
    payload = [t[:_MAX_EMBED_CHARS] for t in texts]
    out: list[list[float]] = []
    batch_size = 64
    for i in range(0, len(payload), batch_size):
        batch = payload[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        by_idx = {item.index: item.embedding for item in resp.data}
        out.extend(by_idx[j] for j in range(len(batch)))
    return out


def _root_key(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]


def _file_fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"mtime": st.st_mtime_ns, "size": st.st_size}


class CodeIndex:
    """Per-root hybrid index with FAISS + BM25 and incremental disk cache."""

    def __init__(
        self,
        root: Path,
        embedder: EmbedFn | None = None,
        cache_root: Path | None = None,
        embed_model: str | None = None,
    ):
        self.root = root.resolve()
        self.embedder = embedder or default_embedder
        self.embed_model = embed_model or os.getenv("NANOCODER_EMBED_MODEL", _DEFAULT_EMBED_MODEL)
        self.cache_dir = (cache_root or _INDEX_ROOT) / _root_key(self.root)
        self.chunks: list[CodeChunk] = []
        self._vectors: np.ndarray | None = None
        self._faiss = None
        self._bm25: BM25 | None = None
        self._lock = threading.Lock()
        self._manifest: dict = {"embed_model": self.embed_model, "files": {}, "dim": None}

    def ensure(self) -> None:
        with self._lock:
            self._ensure_unlocked()

    def _ensure_unlocked(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        loaded = self._try_load()
        current_files = {str(p.relative_to(self.root)).replace("\\", "/"): p
                         for p in iter_python_files(self.root)}
        if loaded and self._manifest.get("embed_model") == self.embed_model:
            changed = self._incremental_update(current_files)
            if not changed and self._faiss is not None:
                return
        else:
            self._full_rebuild(current_files)
        self._rebuild_faiss_bm25()
        self._save()

    def _try_load(self) -> bool:
        meta_path = self.cache_dir / "meta.json"
        chunks_path = self.cache_dir / "chunks.pkl"
        vec_path = self.cache_dir / "vectors.npy"
        if not (meta_path.exists() and chunks_path.exists() and vec_path.exists()):
            return False
        try:
            self._manifest = json.loads(meta_path.read_text(encoding="utf-8"))
            with chunks_path.open("rb") as f:
                raw = pickle.load(f)
            self.chunks = [CodeChunk(**c) if isinstance(c, dict) else c for c in raw]
            self._vectors = np.load(vec_path).astype(np.float32)
            if len(self.chunks) != len(self._vectors):
                return False
            return True
        except (OSError, ValueError, pickle.UnpicklingError, KeyError, TypeError):
            return False

    def _save(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "meta.json").write_text(
            json.dumps(self._manifest, indent=2), encoding="utf-8"
        )
        with (self.cache_dir / "chunks.pkl").open("wb") as f:
            pickle.dump([asdict(c) for c in self.chunks], f)
        if self._vectors is not None:
            np.save(self.cache_dir / "vectors.npy", self._vectors)

    def _full_rebuild(self, current_files: dict[str, Path]) -> None:
        self.chunks = []
        texts: list[str] = []
        new_manifest_files: dict = {}
        for rel, path in sorted(current_files.items()):
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            file_chunks = chunk_python_source(path, source, self.root)
            self.chunks.extend(file_chunks)
            texts.extend(c.text for c in file_chunks)
            new_manifest_files[rel] = {
                **_file_fingerprint(path),
                "chunk_count": len(file_chunks),
            }
        vectors = self._embed_texts(texts) if texts else np.zeros((0, 0), dtype=np.float32)
        self._vectors = vectors
        self._manifest = {
            "embed_model": self.embed_model,
            "files": new_manifest_files,
            "dim": int(vectors.shape[1]) if vectors.size else None,
        }

    def _incremental_update(self, current_files: dict[str, Path]) -> bool:
        """Re-embed only new/changed files; drop deleted ones. Returns True if mutated."""
        old_files = self._manifest.get("files") or {}
        dim = self._manifest.get("dim")
        if self._vectors is None:
            return True

        # group existing chunks by path
        by_path: dict[str, list[tuple[int, CodeChunk]]] = {}
        for i, c in enumerate(self.chunks):
            by_path.setdefault(c.path, []).append((i, c))

        keep_indices: list[int] = []
        new_chunks: list[CodeChunk] = []
        new_texts: list[str] = []
        new_manifest_files: dict = {}
        changed = False

        for rel, path in sorted(current_files.items()):
            fp = _file_fingerprint(path)
            old = old_files.get(rel)
            if old and old.get("mtime") == fp["mtime"] and old.get("size") == fp["size"]:
                idxs = by_path.get(rel, [])
                keep_indices.extend(i for i, _ in idxs)
                new_manifest_files[rel] = old
                continue
            changed = True
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            file_chunks = chunk_python_source(path, source, self.root)
            new_chunks.extend(file_chunks)
            new_texts.extend(c.text for c in file_chunks)
            new_manifest_files[rel] = {**fp, "chunk_count": len(file_chunks)}

        deleted = set(old_files) - set(current_files)
        if deleted:
            changed = True

        if not changed and set(old_files) == set(current_files):
            return False

        kept_vecs = self._vectors[keep_indices] if keep_indices else np.zeros((0, dim or 0), dtype=np.float32)
        kept_chunks = [self.chunks[i] for i in keep_indices]
        if new_texts:
            fresh = self._embed_texts(new_texts)
            if dim is None and fresh.size:
                dim = int(fresh.shape[1])
            if kept_vecs.size == 0 and fresh.size:
                kept_vecs = np.zeros((0, fresh.shape[1]), dtype=np.float32)
            vectors = np.vstack([kept_vecs, fresh]) if kept_vecs.size or fresh.size else kept_vecs
        else:
            vectors = kept_vecs

        self.chunks = kept_chunks + new_chunks
        self._vectors = vectors.astype(np.float32)
        self._manifest = {
            "embed_model": self.embed_model,
            "files": new_manifest_files,
            "dim": int(vectors.shape[1]) if vectors.size else dim,
        }
        return True

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self.embedder([t[:_MAX_EMBED_CHARS] for t in texts])
        arr = np.asarray(vectors, dtype=np.float32)
        # L2-normalize for inner-product = cosine
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return arr / norms

    def _rebuild_faiss_bm25(self) -> None:
        if self._vectors is None or self._vectors.size == 0:
            self._faiss = None
            self._bm25 = BM25([])
            return
        dim = self._vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self._vectors)
        self._faiss = index
        self._bm25 = BM25([tokenize(c.text) for c in self.chunks])

    def search(self, query: str, top_k: int = 8) -> list[CodeChunk]:
        self.ensure()
        if not self.chunks:
            return []
        top_k = max(1, min(top_k, 50))
        candidate_n = min(20, len(self.chunks))

        q_tokens = tokenize(query)
        bm25_ranked = [i for i, _ in (self._bm25.top_k(q_tokens, candidate_n) if self._bm25 else [])]

        dense_ranked: list[int] = []
        if self._faiss is not None and self._faiss.ntotal > 0:
            q_vec = self._embed_texts([query])
            scores, idxs = self._faiss.search(q_vec, candidate_n)
            dense_ranked = [int(i) for i in idxs[0] if i >= 0]

        if not bm25_ranked and not dense_ranked:
            return []
        fused = rrf_fuse([dense_ranked, bm25_ranked], top_n=top_k)
        return [self.chunks[i] for i in fused]


def format_hits(hits: list[CodeChunk], root: Path) -> str:
    if not hits:
        return "No matching code found."
    parts = []
    for i, c in enumerate(hits, 1):
        abs_path = (root / c.path).resolve() if not Path(c.path).is_absolute() else Path(c.path)
        snippet = c.text if len(c.text) <= 800 else c.text[:700] + "\n... (truncated)"
        parts.append(
            f"{i}. {abs_path}:{c.start_line}-{c.end_line}  [{c.symbol}]\n{snippet}"
        )
    return "\n\n".join(parts)


# process-wide indexes keyed by cache dir + root, so parallel tool calls share work
_INDEXES: dict[tuple[str, str], CodeIndex] = {}
_INDEXES_LOCK = threading.Lock()


def get_index(
    root: Path,
    embedder: EmbedFn | None = None,
    cache_root: Path | None = None,
    embed_model: str | None = None,
) -> CodeIndex:
    root = root.resolve()
    model = embed_model or os.getenv("NANOCODER_EMBED_MODEL", _DEFAULT_EMBED_MODEL)
    # custom embedders (tests) are never shared across calls
    if embedder is not None:
        return CodeIndex(root, embedder=embedder, cache_root=cache_root, embed_model=model)
    cache = str((cache_root or _INDEX_ROOT).resolve())
    idx_key = (cache, str(root), model)
    with _INDEXES_LOCK:
        existing = _INDEXES.get(idx_key)
        if existing is not None:
            return existing
        index = CodeIndex(root, cache_root=cache_root, embed_model=model)
        _INDEXES[idx_key] = index
        return index

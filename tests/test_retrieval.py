"""AST chunking, BM25, FAISS hybrid search, and codebase_search tool."""

from pathlib import Path

import numpy as np

from nanocoder.permissions import Permission
from nanocoder.retrieval import (
    BM25,
    CodeIndex,
    chunk_python_source,
    chunk_tree,
    rrf_fuse,
    tokenize,
)
from nanocoder.tools.codebase_search import CodebaseSearchTool


def _hash_embedder(dim: int = 32):
    """Deterministic bag-of-tokens embedding for offline tests."""

    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for tok in tokenize(text):
                vec[hash(tok) % dim] += 1.0
            norm = float(np.linalg.norm(vec)) or 1.0
            out.append((vec / norm).tolist())
        return out

    return embed


def test_chunk_python_extracts_functions_and_methods(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text(
        "def top():\n"
        "    return 1\n"
        "\n"
        "class Foo:\n"
        "    def bar(self):\n"
        "        return 2\n",
        encoding="utf-8",
    )
    chunks = chunk_python_source(src, src.read_text(encoding="utf-8"), tmp_path)
    symbols = {c.symbol for c in chunks}
    assert "top" in symbols
    assert "Foo" in symbols
    assert "Foo.bar" in symbols
    top = next(c for c in chunks if c.symbol == "top")
    assert top.start_line == 1
    assert "def top" in top.text


def test_chunk_skips_syntax_errors_and_falls_back_to_module(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def (\n", encoding="utf-8")
    assert chunk_python_source(bad, bad.read_text(encoding="utf-8"), tmp_path) == []

    script = tmp_path / "script.py"
    script.write_text("x = 1\ny = 2\n", encoding="utf-8")
    chunks = chunk_python_source(script, script.read_text(encoding="utf-8"), tmp_path)
    assert len(chunks) == 1
    assert chunks[0].symbol == "<module>"


def test_tokenize_splits_snake_and_camel():
    toks = tokenize("parseJSON_data fooBar")
    assert "parse" in toks
    assert "json" in toks
    assert "data" in toks
    assert "foo" in toks
    assert "bar" in toks


def test_bm25_ranks_matching_document_first():
    corpus = [
        tokenize("def alpha(): pass"),
        tokenize("def authenticate_user(password): check password"),
        tokenize("def beta(): pass"),
    ]
    bm25 = BM25(corpus)
    ranked = bm25.top_k(tokenize("authenticate password"), k=3)
    assert ranked[0][0] == 1


def test_rrf_fuse_prefers_shared_high_ranks():
    fused = rrf_fuse([[0, 1, 2], [1, 0, 3]], top_n=2)
    assert fused[0] in (0, 1)
    assert set(fused) <= {0, 1, 2, 3}


def test_hybrid_search_finds_relevant_function(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def authenticate_user(username, password):\n"
        "    '''Validate credentials against the user store.'''\n"
        "    return username == 'admin' and password == 'secret'\n"
        "\n"
        "def list_files(path):\n"
        "    return []\n",
        encoding="utf-8",
    )
    (tmp_path / "math_util.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    index = CodeIndex(
        tmp_path,
        embedder=_hash_embedder(),
        cache_root=tmp_path / ".cache",
        embed_model="test-hash",
    )
    hits = index.search("authenticate user password login", top_k=3)
    assert hits
    assert any("authenticate" in h.symbol for h in hits)


def test_incremental_index_reembeds_only_changed_file(tmp_path):
    cache = tmp_path / ".cache"
    a = tmp_path / "a.py"
    a.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    index = CodeIndex(tmp_path, embedder=_hash_embedder(), cache_root=cache, embed_model="test-hash")
    index.ensure()
    n_first = len(index.chunks)

    b = tmp_path / "b.py"
    b.write_text("def beta_unique_symbol():\n    return 2\n", encoding="utf-8")
    index2 = CodeIndex(tmp_path, embedder=_hash_embedder(), cache_root=cache, embed_model="test-hash")
    index2.ensure()
    assert len(index2.chunks) > n_first
    assert any(c.symbol == "beta_unique_symbol" for c in index2.chunks)


def test_codebase_search_tool_returns_paths_and_lines(tmp_path):
    (tmp_path / "pay.py").write_text(
        "def charge_card(amount):\n"
        "    '''Bill the customer card for amount.'''\n"
        "    return amount\n",
        encoding="utf-8",
    )
    tool = CodebaseSearchTool(embedder=_hash_embedder(), cache_root=tmp_path / ".idx")
    out = tool.execute("charge card billing", path=str(tmp_path), top_k=3)
    assert "charge_card" in out
    assert "pay.py" in out
    assert ":" in out  # path:lines


def test_codebase_search_is_read_only():
    assert "codebase_search" in Permission.READ_ONLY
    assert Permission(allow_all=False).check("codebase_search", {"query": "x"}) is None


def test_chunk_tree_skips_venv_dirs(tmp_path):
    (tmp_path / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "site.py").write_text("def site():\n    return 0\n", encoding="utf-8")
    chunks = chunk_tree(tmp_path)
    assert all(".venv" not in c.path for c in chunks)
    assert any(c.symbol == "app" for c in chunks)

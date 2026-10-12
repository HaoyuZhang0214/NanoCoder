"""Session-scoped undo for file mutations.

edit_file and write_file record a checkpoint before touching a file; /undo
pops the latest one and restores the previous bytes (or removes the file if
it did not exist). The stack persists to ~/.nanocoder/checkpoints.json so
/undo still reaches back into earlier sessions after a restart. Bash side
effects are not tracked, only the two file-writing tools.
"""

import base64
import json
from pathlib import Path

CHECKPOINTS_FILE = Path.home() / ".nanocoder" / "checkpoints.json"

# (path, prior bytes or None if the file did not exist)
_stack: list[tuple[str, bytes | None]] = []
_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        entries = json.loads(CHECKPOINTS_FILE.read_text(encoding="utf-8"))
        for e in entries:
            prior = e["prior"]
            _stack.append((e["path"], base64.b64decode(prior) if prior is not None else None))
    except (OSError, ValueError, KeyError, TypeError):
        # a missing or corrupt stack just means nothing to undo
        pass


def _save() -> None:
    entries = [
        {"path": path, "prior": base64.b64encode(prior).decode("ascii") if prior is not None else None}
        for path, prior in _stack
    ]
    try:
        CHECKPOINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CHECKPOINTS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries), encoding="utf-8")
        tmp.replace(CHECKPOINTS_FILE)
    except OSError:
        # undo is a safety net; losing its persistence must not break the edit
        pass


def record(path: Path) -> None:
    """Capture the pre-mutation state of path. Call right before writing."""
    _ensure_loaded()
    _stack.append((str(path), path.read_bytes() if path.exists() else None))
    _save()


def undo() -> str:
    """Restore the most recent checkpoint."""
    _ensure_loaded()
    if not _stack:
        return "Nothing to undo."
    path_str, prior = _stack.pop()
    p = Path(path_str)
    if prior is None:
        p.unlink(missing_ok=True)
        msg = f"Removed {path_str} (created this session)."
    else:
        # the parent tree may be gone by now: bash side effects are untracked,
        # so recreate it instead of dying on FileNotFoundError
        recreated = not p.parent.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(prior)
        msg = f"Restored {path_str}." if not recreated else (
            f"Restored {path_str} (recreated missing parent directories)."
        )
    _save()
    return msg


def pending() -> int:
    _ensure_loaded()
    return len(_stack)


def clear() -> None:
    global _loaded
    _stack.clear()
    _loaded = True  # the stack is known-empty; don't reload the file we just removed
    try:
        CHECKPOINTS_FILE.unlink(missing_ok=True)
    except OSError:
        pass

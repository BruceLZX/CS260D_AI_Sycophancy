"""Small file helpers shared by every step."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import threading
from pathlib import Path

_LOCK = threading.Lock()


def sha12(text: str) -> str:
    """Short, stable fingerprint of a text (first 12 hex characters of its SHA-256)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path) -> list[dict]:
    """All records in a JSONL file ([] if the file does not exist). A half-written line left
    by an interrupted run is skipped with a warning."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: skipping unreadable line {lineno} in {path}")
    return records


def append_jsonl(path, record: dict) -> None:
    """Append one record and close the file, so an interrupted run loses at most the calls
    still in flight. Thread-safe."""
    path = Path(path)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        # If an earlier run died mid-line, start on a fresh line so this record stays readable.
        if path.exists() and path.stat().st_size > 0:
            with open(path, "rb") as f:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    line = "\n" + line
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)

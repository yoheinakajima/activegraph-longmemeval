from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FALSEY_ENV = {"0", "false", "off", "no", "disabled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_embedding_cache_path() -> Path | None:
    """Resolve the durable embedding cache path.

    The default is intentionally local and gitignored. Set
    ``AGLME_EMBEDDING_CACHE=off`` to disable, or point the variable at a
    different SQLite file to share cache state across checkouts.
    """
    raw = os.environ.get("AGLME_EMBEDDING_CACHE")
    if raw is not None:
        if raw.strip().lower() in FALSEY_ENV:
            return None
        return Path(raw).expanduser()
    return Path(".embedding_cache/embeddings.sqlite3")


def embedding_cache_key(model: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model}:{digest}"


@dataclass
class EmbeddingCacheStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    stores: int = 0
    store_conflicts: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "store_conflicts": self.store_conflicts,
        }


class PersistentEmbeddingCache:
    """SQLite-backed, content-addressed store for normalized embeddings."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=60)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self.stats = EmbeddingCacheStats()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                dim INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                vector BLOB NOT NULL,
                text_preview TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                key TEXT,
                model TEXT,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_events_key ON events(key)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get(self, model: str, text: str) -> np.ndarray | None:
        self.stats.requests += 1
        key = embedding_cache_key(model, text)
        row = self._conn.execute(
            "SELECT vector, dim, dtype FROM embeddings WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            self.stats.misses += 1
            return None
        blob, dim, dtype = row
        if dtype != "float32":
            raise RuntimeError(f"Unsupported cached embedding dtype: {dtype!r}")
        arr = np.frombuffer(blob, dtype=np.float32).copy()
        if int(dim) != int(arr.shape[0]):
            raise RuntimeError(
                f"Corrupt cached embedding {key}: dim={dim}, bytes={arr.shape[0]}"
            )
        self.stats.hits += 1
        return arr

    def put(
        self,
        model: str,
        text: str,
        vector: np.ndarray,
        *,
        source: str = "openai",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        arr = np.asarray(vector, dtype=np.float32)
        key = embedding_cache_key(model, text)
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        now = utc_now()
        preview = " ".join(text[:240].split())
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO embeddings
                    (key, model, text_sha256, dim, dtype, vector, text_preview,
                     created_at, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    model,
                    text_sha,
                    int(arr.shape[0]),
                    "float32",
                    arr.tobytes(),
                    preview,
                    now,
                    source,
                ),
            )
            if cur.rowcount:
                self.stats.stores += 1
                self._append_event(
                    "embedding.stored",
                    key,
                    model,
                    {
                        "dim": int(arr.shape[0]),
                        "source": source,
                        **(metadata or {}),
                    },
                )
            else:
                self.stats.store_conflicts += 1

    def _append_event(
        self,
        event_type: str,
        key: str | None,
        model: str | None,
        metadata: dict[str, Any],
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO events (event_id, type, created_at, key, model, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                event_type,
                utc_now(),
                key,
                model,
                json.dumps(metadata, sort_keys=True),
            ),
        )

    def to_manifest(self) -> dict[str, Any]:
        n_embeddings = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]
        n_events = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {
            "path": str(self.path),
            "n_embeddings": int(n_embeddings),
            "n_events": int(n_events),
            "stats": self.stats.to_dict(),
        }

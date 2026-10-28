"""Prediction history in SQLite (stdlib only, one connection per call)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    filename    TEXT,
    class_key   TEXT    NOT NULL,
    crop        TEXT,
    name        TEXT,
    status      TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    top_k       TEXT    NOT NULL,
    thumbnail   TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at DESC);
"""


class HistoryStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add(self, *, filename, class_key, crop, name, status, confidence, top_k, thumbnail) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO predictions (created_at, filename, class_key, crop, name, status,"
                " confidence, top_k, thumbnail) VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), filename, class_key,
                 crop, name, status, confidence, json.dumps(top_k), thumbnail))
            return int(cur.lastrowid)

    def list(self, limit: int = 50, offset: int = 0, status: str | None = None) -> tuple[list[dict], int]:
        where, args = ("WHERE status = ?", [status]) if status else ("", [])
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM predictions {where}", args).fetchone()[0]
            rows = c.execute(f"SELECT * FROM predictions {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                             [*args, limit, offset]).fetchall()
        return [self._row(r) for r in rows], int(total)

    def delete(self, pid: int) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM predictions WHERE id = ?", (pid,)).rowcount > 0

    def clear(self) -> int:
        with self._conn() as c:
            return c.execute("DELETE FROM predictions").rowcount

    def stats(self) -> dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            by_status = dict(c.execute("SELECT status, COUNT(*) FROM predictions GROUP BY status").fetchall())
            by_class = [dict(r) for r in c.execute(
                "SELECT class_key, crop, name, COUNT(*) AS count, AVG(confidence) AS avg_confidence "
                "FROM predictions WHERE status IN ('diseased','healthy','uncertain') "
                "GROUP BY class_key ORDER BY count DESC LIMIT 10").fetchall()]
            by_crop = dict(c.execute(
                "SELECT crop, COUNT(*) FROM predictions WHERE status != 'no_leaf' "
                "GROUP BY crop ORDER BY COUNT(*) DESC").fetchall())
        return {"total": total, "by_status": by_status, "top_classes": by_class, "by_crop": by_crop}

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["top_k"] = json.loads(d["top_k"])
        return d

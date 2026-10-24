"""SQLite-Persistenz. Eine kurze Verbindung pro Aufruf, Schreibzugriffe serialisiert."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS creators(
  id INTEGER PRIMARY KEY, slug TEXT UNIQUE NOT NULL, name TEXT NOT NULL, source_dir TEXT NOT NULL,
  notes TEXT DEFAULT '', rights INTEGER NOT NULL DEFAULT 0, rights_note TEXT DEFAULT '', rights_date REAL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tracks(
  id INTEGER PRIMARY KEY, creator_id INTEGER NOT NULL REFERENCES creators(id) ON DELETE CASCADE,
  path TEXT NOT NULL, title TEXT NOT NULL, size INTEGER, mtime REAL, duration REAL, features TEXT, error TEXT, analyzed REAL,
  UNIQUE(creator_id, path));
CREATE TABLE IF NOT EXISTS profiles(
  creator_id INTEGER PRIMARY KEY REFERENCES creators(id) ON DELETE CASCADE, data TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS adapters(
  creator_id INTEGER PRIMARY KEY REFERENCES creators(id) ON DELETE CASCADE, path TEXT NOT NULL, base_model TEXT NOT NULL,
  clips INTEGER, epochs INTEGER, steps INTEGER, loss REAL, seconds REAL, trained REAL NOT NULL);
CREATE TABLE IF NOT EXISTS pieces(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, params TEXT NOT NULL, status TEXT NOT NULL, audio_path TEXT, error TEXT,
  created REAL NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, params TEXT, status TEXT NOT NULL,
  progress REAL DEFAULT 0, log TEXT DEFAULT '', created REAL NOT NULL, started REAL, finished REAL, priority INTEGER NOT NULL DEFAULT 0);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path, self._lock = path, threading.RLock()
        self._conn: sqlite3.Connection | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn: conn.executescript(SCHEMA)

    PATH_COLUMNS = (("creators", "source_dir"), ("tracks", "path"), ("adapters", "path"), ("pieces", "audio_path"))

    def rebase_paths(self, root: Path) -> int:
        """Absolute Pfade aus einem frueheren Speicherort des Projekts auf den aktuellen Projektordner umschreiben (Projekt ist verschiebbar)."""
        pattern = re.compile(r"^(?:[A-Za-z]:|\\\\|/)[^\r\n]*?[\\/]" + re.escape(root.name) + r"[\\/](.+)$")
        changed = 0
        with self._lock, self.connect() as conn:
            for table, column in self.PATH_COLUMNS:
                for row in conn.execute(f"SELECT rowid AS rid, {column} AS value FROM {table} WHERE {column} IS NOT NULL").fetchall():
                    match = pattern.match(row["value"])
                    if not match: continue
                    new = str(root.joinpath(*re.split(r"[\\/]", match.group(1))))
                    if new == row["value"]: continue
                    try: conn.execute(f"UPDATE {table} SET {column}=? WHERE rowid=?", (new, row["rid"])); changed += 1
                    except sqlite3.IntegrityError: pass
        return changed

    @contextmanager
    def connect(self):
        """Eine Prozessverbindung, serialisiert ueber alle Worker-Threads.

        Das wiederholte Oeffnen/Schliessen von SQLite-Dateien auf SMB erzeugt
        kurzlebige Byte-Locks und war die Quelle der sporadischen I/O-Fehler.
        Eine Verbindung pro MusicAI-Prozess vermeidet diese Locks; die
        Ein-Instanz-Sperre im Starter verhindert gleichzeitig Parallelzugriffe.
        """
        with self._lock:
            if self._conn is None:
                conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA busy_timeout=30000")
                self._conn = conn
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _reset_connection(self) -> None:
        with self._lock:
            if self._conn is not None:
                try: self._conn.close()
                except sqlite3.Error: pass
                self._conn = None

    @staticmethod
    def _transient(error: sqlite3.OperationalError) -> bool:
        text = str(error).lower()
        return "locked" in text or "busy" in text or "disk i/o" in text

    def query(self, sql: str, args: Iterable = ()) -> list[dict[str, Any]]:
        for attempt in range(4):
            try:
                with self.connect() as conn: return [dict(row) for row in conn.execute(sql, tuple(args)).fetchall()]
            except sqlite3.OperationalError as error:
                if not self._transient(error) or attempt == 3: raise
                self._reset_connection()
                time.sleep(0.25 * (attempt + 1))
        raise AssertionError("unerreichbar")

    def one(self, sql: str, args: Iterable = ()) -> dict[str, Any] | None:
        rows = self.query(sql, args); return rows[0] if rows else None

    def execute(self, sql: str, args: Iterable = ()) -> int:
        for attempt in range(4):
            try:
                with self._lock, self.connect() as conn: return conn.execute(sql, tuple(args)).lastrowid or 0
            except sqlite3.OperationalError as error:
                if not self._transient(error) or attempt == 3: raise
                self._reset_connection()
                time.sleep(0.25 * (attempt + 1))
        raise AssertionError("unerreichbar")

    def setting(self, key: str, default: str | None = None) -> str | None:
        row = self.one("SELECT value FROM settings WHERE key=?", (key,)); return row["value"] if row else default

    def set_setting(self, key: str, value: str | None) -> None:
        if value is None: self.execute("DELETE FROM settings WHERE key=?", (key,))
        else: self.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def dumps(value: Any) -> str: return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

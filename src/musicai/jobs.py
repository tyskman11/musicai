"""Hintergrund-Jobs: je Art ein Worker mit Prioritaets-Warteschlange (hoehere Prioritaet zuerst, danach FIFO); verschiedene Arten laufen parallel."""
from __future__ import annotations

import itertools
import json
import logging
import queue
import threading
import time
import traceback
from typing import Any, Callable

from .db import Database, dumps
from .logging_setup import diagnostic

log = logging.getLogger("musicai.jobs")


class Cancelled(Exception):
    pass


class Yield(Exception):
    """Der laufende Job pausiert zwischen zwei Arbeitsschritten, weil ein Job hoeherer Prioritaet wartet (er wird danach fortgesetzt)."""


class JobContext:
    def __init__(self, manager: "JobManager", job_id: int, kind: str = "", title: str = "") -> None:
        self.manager, self.job_id, self.cancel_event = manager, job_id, threading.Event()
        self.kind, self.title = kind, title
        self._last_flush, self._log, self._progress = 0.0, [], 0.0
        self.last_activity = self.started = time.monotonic()
        self.current = ""        # letzte Schrittmeldung (fuer den Live-Status im Terminal)

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Eine Zeile: sofort im Terminal (Logger), in logs/*.log und im Job-Log der WebUI."""
        self._log.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        self._log = self._log[-400:]
        self.last_activity = time.monotonic(); self.current = message.strip()
        log.log(level, "Job %s %s", self.job_id, message)
        self._flush()

    def progress(self, value: float, message: str | None = None) -> None:
        self._progress = max(0.0, min(1.0, value))
        if message: self.log(message)
        else: self._flush()

    def check(self) -> None:
        if self.cancel_event.is_set(): raise Cancelled()

    def yield_to_higher_priority(self) -> None:
        """Zwischen zwei Arbeitsschritten aufrufen (z. B. nach jedem Video). Wartet ein Job derselben Art mit hoeherer Prioritaet,
        pausiert dieser Job; bereits erledigte Schritte bleiben erhalten und werden beim Fortsetzen uebersprungen."""
        db = self.manager.db; mine = db.one("SELECT priority FROM jobs WHERE id=?", (self.job_id,))
        waiting = db.one("SELECT id,title FROM jobs WHERE kind=? AND status='queued' AND priority>? ORDER BY priority DESC, id LIMIT 1", (self.kind, mine["priority"] if mine else 0))
        if waiting: raise Yield(waiting["title"])

    def _flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_flush > 0.7:
            self._last_flush = now
            self.manager.db.execute("UPDATE jobs SET progress=?, log=? WHERE id=?", (self._progress, "\n".join(self._log), self.job_id))


class JobManager:
    def __init__(self, db: Database) -> None:
        self.db, self.handlers = db, {}
        self.queues: dict[str, queue.PriorityQueue] = {}
        self._seq = itertools.count()
        self.contexts: dict[int, JobContext] = {}
        self._lock = threading.Lock()

    def register(self, kind: str, handler: Callable[[JobContext, dict], Any]) -> None:
        self.handlers[kind] = handler

    def resume(self) -> list[int]:
        """Nach einem Neustart: laufende/wartende Jobs der alten Sitzung wieder einreihen (Analyse-Stufen sind fortsetzbar)."""
        rows = self.db.query("SELECT id, kind FROM jobs WHERE status IN ('running','queued') ORDER BY id")
        resumed = []
        for row in rows:
            if row["kind"] not in self.handlers or row["id"] in self.contexts: continue
            self.db.execute("UPDATE jobs SET status='queued', progress=0, started=NULL, log=log || ? WHERE id=?",
                            (f"\n[{time.strftime('%H:%M:%S')}] Nach Neustart automatisch wieder eingereiht.", row["id"]))
            self._enqueue(row["kind"], row["id"]); resumed.append(row["id"])
        return resumed

    def retry(self, job_id: int) -> int:
        row = self.db.one("SELECT kind,title,params,status,priority FROM jobs WHERE id=?", (job_id,))
        if not row: raise KeyError("Job nicht gefunden")
        if row["status"] in ("queued", "running"): raise ValueError("Job laeuft noch.")
        return self.submit(row["kind"], row["title"], json.loads(row["params"] or "{}"), row.get("priority") or 0)

    def delete(self, job_id: int | None = None, statuses: list[str] | None = None) -> int:
        """Loescht nur beendete Jobs (done/error/canceled/interrupted) - niemals laufende oder wartende."""
        allowed = {"done", "error", "canceled", "interrupted"}
        if job_id is not None:
            row = self.db.one("SELECT status FROM jobs WHERE id=?", (job_id,))
            if not row: raise KeyError("Job nicht gefunden")
            if row["status"] not in allowed: raise ValueError("Laufende oder wartende Jobs zuerst abbrechen.")
            self.db.execute("DELETE FROM jobs WHERE id=?", (job_id,)); return 1
        wanted = [s for s in (statuses or []) if s in allowed]
        if not wanted: raise ValueError("Kein loeschbarer Status angegeben.")
        marks = ",".join("?" * len(wanted))
        count = self.db.one(f"SELECT COUNT(*) AS n FROM jobs WHERE status IN ({marks})", wanted)["n"]
        self.db.execute(f"DELETE FROM jobs WHERE status IN ({marks})", wanted)
        return count

    def _enqueue(self, kind: str, job_id: int) -> None:
        row = self.db.one("SELECT priority FROM jobs WHERE id=?", (job_id,)); priority = row["priority"] if row else 0
        with self._lock:
            if kind not in self.queues:
                self.queues[kind] = queue.PriorityQueue()
                threading.Thread(target=self._worker, args=(kind,), daemon=True, name=f"job-{kind}").start()
            self.queues[kind].put((-priority, next(self._seq), job_id))          # hoehere Prioritaet zuerst, gleiche: Reihenfolge des Eintreffens

    def submit(self, kind: str, title: str, params: dict | None = None, priority: int = 0) -> int:
        if kind not in self.handlers: raise ValueError(f"Unbekannte Job-Art: {kind}")
        job_id = self.db.execute("INSERT INTO jobs(kind,title,params,status,created,priority) VALUES(?,?,?,?,?,?)",
                                 (kind, title, dumps(params or {}), "queued", time.time(), priority))
        self._enqueue(kind, job_id)
        return job_id

    def prioritize(self, job_id: int, priority: int = 20) -> None:
        """Einen wartenden Job vorziehen. Laeuft gerade ein Job derselben Art, pausiert er nach seinem aktuellen Schritt."""
        row = self.db.one("SELECT kind,status FROM jobs WHERE id=?", (job_id,))
        if not row: raise KeyError("Job nicht gefunden")
        if row["status"] != "queued": raise ValueError("Nur wartende Jobs lassen sich vorziehen.")
        self.db.execute("UPDATE jobs SET priority=? WHERE id=?", (priority, job_id))
        self._enqueue(row["kind"], job_id)           # zusaetzlicher Eintrag mit neuer Prioritaet; der alte wird beim Abholen uebersprungen

    def cancel(self, job_id: int) -> bool:
        context = self.contexts.get(job_id)
        if context: context.cancel_event.set(); return True
        row = self.db.one("SELECT status FROM jobs WHERE id=?", (job_id,))
        if row and row["status"] == "queued":
            self.db.execute("UPDATE jobs SET status='canceled', finished=? WHERE id=?", (time.time(), job_id)); return True
        return False

    def active(self, kind: str | None = None) -> list[dict]:
        sql = "SELECT * FROM jobs WHERE status IN ('queued','running')" + (" AND kind=?" if kind else "")
        return self.db.query(sql, (kind,) if kind else ())

    def _worker(self, kind: str) -> None:
        while True:
            job_id = self.queues[kind].get()[2]
            row = self.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
            if not row or row["status"] != "queued": continue
            context = JobContext(self, job_id, kind, row["title"]); self.contexts[job_id] = context
            self.db.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (time.time(), job_id))
            log.info("=== Job %s gestartet: %s (%s) ===", job_id, row["title"], kind)
            diagnostic("job_started", job_id=job_id, kind=kind, title=row["title"], priority=row.get("priority", 0))
            started, status, paused = time.monotonic(), "done", False
            try:
                self.handlers[kind](context, json.loads(row["params"] or "{}"))
                context.progress(1.0)
            except Yield as pause:
                status, paused = "queued", True
                context.log(f"Pausiert - Job mit hoeherer Prioritaet wartet: {pause}. Danach geht es automatisch weiter.", logging.WARNING)
            except Cancelled:
                status = "canceled"; context.log("Abgebrochen.", logging.WARNING)
            except Exception as error:  # noqa: BLE001 - jobs must never kill the worker
                status = "error"; context._log.append(f"[{time.strftime('%H:%M:%S')}] FEHLER: {error}"); context._log.append(traceback.format_exc(limit=6))
                log.error("Job %s (%s) fehlgeschlagen: %s", job_id, row["title"], error, exc_info=True)      # Traceback -> logs/errors.log
                diagnostic("job_failed", job_id=job_id, kind=kind, title=row["title"], error_type=type(error).__name__, error=str(error)[:500], step=context.current[:500])
            finally:
                context._flush(force=True); self.contexts.pop(job_id, None)
                self.db.execute("UPDATE jobs SET status=?, finished=? WHERE id=?", (status, None if paused else time.time(), job_id))
                log.info("=== Job %s %s nach %.1f min ===", job_id, {"done": "beendet: fertig", "error": "beendet: FEHLER", "canceled": "beendet: abgebrochen", "queued": "pausiert"}.get(status, status), (time.monotonic() - started) / 60)
                diagnostic("job_finished", job_id=job_id, kind=kind, status=status, seconds=round(time.monotonic() - started, 3))
                if paused: self._enqueue(kind, job_id)

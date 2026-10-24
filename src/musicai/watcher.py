"""Beobachtet den Eingangsordner: neue Creator-Ordner und neue Dateien werden erkannt; die Analyse startet per Button (oder auto_analyze)."""
from __future__ import annotations

import logging
import threading

from . import pipeline

log = logging.getLogger("musicai.watcher")


class Watcher:
    def __init__(self, app) -> None:
        self.app = app
        self.last: dict = {"new_creators": [], "new_tracks": {}}

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="inbox-watcher").start()

    def check(self) -> dict:
        db, config = self.app.db, self.app.config
        settle = float(config.get("inbox", "settle_seconds", 30)); auto = bool(config.get("inbox", "auto_analyze", False))
        new_creators = [c["name"] for c in pipeline.sync_inbox(db, config)]; new_tracks: dict[str, int] = {}
        for creator in db.query("SELECT * FROM creators"):
            try: added, _ = pipeline.scan_creator(db, creator, settle)
            except FileNotFoundError: continue
            if added:
                new_tracks[creator["name"]] = added
                if auto and not any(j["title"] == f"Analyse: {creator['name']}" for j in self.app.jobs.active("analyze")): self.app.start_analyze(creator["id"])
        if new_creators or new_tracks: log.info("Eingang: neue Creator %s, neue Titel %s", new_creators or "-", new_tracks or "-")
        self.last = {"new_creators": new_creators, "new_tracks": new_tracks}
        return self.last

    def status(self) -> dict:
        pending = self.app.db.query("SELECT c.name AS creator, COUNT(*) AS n FROM tracks t JOIN creators c ON c.id=t.creator_id WHERE t.features IS NULL AND t.error IS NULL GROUP BY c.id")
        return {"pending": pending, "auto_analyze": bool(self.app.config.get("inbox", "auto_analyze", False)), "last": self.last}

    def _run(self) -> None:
        interval = float(self.app.config.get("inbox", "interval_seconds", 60))
        while True:
            try: self.check()
            except Exception:  # noqa: BLE001 - der Waechter darf nie sterben
                log.error("Eingangsordner-Pruefung fehlgeschlagen", exc_info=True)
            threading.Event().wait(interval)

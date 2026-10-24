"""Ingest + fortsetzbare Analyse: Creator-Ordner -> Titel -> Merkmale -> Stil-Profil."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

from . import analysis, media
from .config import Config
from .db import Database, dumps
from .jobs import JobContext
from .profile import build_profile

alog = logging.getLogger("musicai.analysis")
_scan_lock = threading.Lock()      # Waechter und Analyse-Job duerfen nie gleichzeitig denselben Ordner eintragen


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "creator"


def clean_title(path: Path) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\s*\[[0-9A-Za-z_-]{8,}\]\s*$", "", path.stem)).strip()      # Download-IDs in eckigen Klammern entfernen


def sync_inbox(db: Database, config: Config) -> list[dict]:
    """Jeder Unterordner von input/ ist ein Creator. Neue Ordner werden angelegt (ohne Rechte-Haken: Training bleibt gesperrt)."""
    created = []
    for folder in sorted(p for p in config.input.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        if db.one("SELECT id FROM creators WHERE source_dir=?", (str(folder),)): continue
        slug = slugify(folder.name)
        if db.one("SELECT id FROM creators WHERE slug=?", (slug,)): slug = f"{slug}-{int(time.time()) % 10000}"
        db.execute("INSERT INTO creators(slug,name,source_dir,created) VALUES(?,?,?,?)", (slug, folder.name, str(folder), time.time()))
        created.append(db.one("SELECT * FROM creators WHERE slug=?", (slug,)))
    return created


def scan_creator(db: Database, creator: dict, settle_seconds: float = 0) -> tuple[int, int]:
    """Registriert neue Audio-/Videodateien des Quellordners. -> (neu, gesamt)"""
    root = Path(creator["source_dir"])
    if not root.is_dir(): raise FileNotFoundError(f"Quellordner nicht gefunden: {root}")
    with _scan_lock:
        known = {row["path"] for row in db.query("SELECT path FROM tracks WHERE creator_id=?", (creator["id"],))}
        added = 0
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in media.MEDIA_EXT and path.is_file() and str(path) not in known:
                stat = path.stat()
                if settle_seconds and time.time() - stat.st_mtime < settle_seconds: continue      # wird evtl. noch kopiert
                db.execute("INSERT OR IGNORE INTO tracks(creator_id,path,title,size,mtime) VALUES(?,?,?,?,?)", (creator["id"], str(path), clean_title(path), stat.st_size, stat.st_mtime)); added += 1
        return added, db.one("SELECT COUNT(*) AS n FROM tracks WHERE creator_id=?", (creator["id"],))["n"]


def analyze_creator(ctx: JobContext, db: Database, config: Config, creator_id: int, force: bool = False, limit: int | None = None, only_failed: bool = False) -> dict:
    creator = db.one("SELECT * FROM creators WHERE id=?", (creator_id,))
    if not creator: raise KeyError("Creator nicht gefunden")
    added, total = scan_creator(db, creator)
    ctx.log(f"{creator['name']}: {total} Titel ({added} neu)")
    rows = db.query("SELECT * FROM tracks WHERE creator_id=? ORDER BY id", (creator_id,))
    todo = [r for r in rows if r["error"] is not None and r["features"] is None] if only_failed else [r for r in rows if force or r["features"] is None]
    if only_failed: ctx.log(f"Nur fehlgeschlagene Titel wiederholen: {len(todo)}")
    if limit: todo = todo[:limit]
    sr = int(config.get("analysis", "analysis_sample_rate", 22050)); max_seconds = float(config.get("analysis", "max_seconds", 1200))
    done = failed = 0
    for i, row in enumerate(todo, 1):
        ctx.check(); ctx.progress((i - 1) / max(len(todo), 1), f"[{i}/{len(todo)}] {row['title']}")
        started = time.monotonic()
        try:
            info = media.probe(config, row["path"])
            audio = media.decode(config, row["path"], sr, seconds=max_seconds)
            feat = analysis.analyze(audio, sr)
            feat["duration"] = min(info["duration"], max_seconds) if info["duration"] else feat["duration"]
            db.execute("UPDATE tracks SET duration=?, features=?, error=NULL, analyzed=? WHERE id=?", (info["duration"], dumps(feat), time.time(), row["id"]))
            alog.info("%s: %.0f BPM, %s %s, %.1fs", row["title"], feat["bpm"], feat["tonic"], feat["mode"], time.monotonic() - started); done += 1
        except Exception as error:  # noqa: BLE001 - ein defekter Titel stoppt die Analyse nicht
            db.execute("UPDATE tracks SET error=? WHERE id=?", (str(error)[:300], row["id"])); failed += 1
            ctx.log(f"  uebersprungen: {error}", logging.WARNING)
    profile = None
    if db.one("SELECT 1 AS x FROM tracks WHERE creator_id=? AND features IS NOT NULL", (creator_id,)):
        profile = build_profile(db, creator_id); ctx.log(f"Stil-Profil: {profile['caption']} ({profile['tracks']} Titel, {profile['hours']} h)")
    return {"analyzed": done, "failed": failed, "profile": bool(profile)}

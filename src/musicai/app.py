"""Anwendungskern: Datenbank, Jobs, Creator-Verwaltung, Analyse/Training/Erzeugung als Jobs."""
from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from . import pipeline
from .config import ROOT, Config
from .db import Database, dumps
from .generate import render_piece, resolve_styles
from .jobs import JobManager
from .profile import get_profile, mix_profiles, normalize_weights
from .train import adapter_dir, train_creator

log = logging.getLogger("musicai.app")


class App:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.load()
        self.db = Database(self.config.db_path)  # ausschliesslich projektnah auf dem NAS
        rebased = self.db.rebase_paths(ROOT)
        if rebased: log.info("%d gespeicherte Pfade auf den aktuellen Projektordner umgeschrieben.", rebased)
        merged = self._deduplicate_creators()
        if merged: log.warning("%d doppelte Creator-Verknuepfung(en) nach Pfadvereinheitlichung zusammengefuehrt.", merged)
        self.jobs = JobManager(self.db)
        self.watcher = None
        self.jobs.register("analyze", lambda ctx, p: pipeline.analyze_creator(ctx, self.db, self.config, p["creator_id"], bool(p.get("force")), p.get("limit"), bool(p.get("only_failed"))))
        self.jobs.register("train", lambda ctx, p: train_creator(ctx, self.db, self.config, p["creator_id"]))
        self.jobs.register("render", lambda ctx, p: render_piece(ctx, self.db, self.config, p["piece_id"]))
        for job in self.db.query("SELECT id FROM jobs WHERE status='running'"):
            self.db.execute("UPDATE jobs SET status='interrupted', finished=? WHERE id=?", (time.time(), job["id"]))
        pipeline.sync_inbox(self.db, self.config)
        self.jobs.resume()

    def _deduplicate_creators(self) -> int:
        """Fasst nur DB-Dubletten desselben NAS-Ordners zusammen.

        Dies kann nach einem frueheren Proxy-/Containerpfad entstehen. Die
        Quelldateien werden nie veraendert; bevorzugt wird der Creator mit den
        meisten analysierten Titeln und bestaetigten Rechten.
        """
        groups: dict[str, list[dict]] = {}
        for creator in self.db.query("SELECT id,source_dir,rights FROM creators"):
            groups.setdefault(str(creator["source_dir"]).replace("/", "\\\\").casefold(), []).append(creator)
        merged = 0
        for creators in groups.values():
            if len(creators) < 2: continue
            def score(c):
                stats = self.db.one("SELECT SUM(features IS NOT NULL) AS analyzed FROM tracks WHERE creator_id=?", (c["id"],))
                return (stats["analyzed"] or 0, int(bool(c["rights"])), -c["id"])
            keep = max(creators, key=score)
            for extra in creators:
                if extra["id"] == keep["id"]: continue
                # Gleiche Datei nicht doppelt umhaengen; die besser bewertete
                # Verknuepfung bleibt beim Ziel-Creator erhalten.
                self.db.execute("DELETE FROM tracks WHERE creator_id=? AND path IN (SELECT path FROM tracks WHERE creator_id=?)", (extra["id"], keep["id"]))
                self.db.execute("UPDATE tracks SET creator_id=? WHERE creator_id=?", (keep["id"], extra["id"]))
                if not self.db.one("SELECT 1 AS x FROM profiles WHERE creator_id=?", (keep["id"],)):
                    self.db.execute("UPDATE profiles SET creator_id=? WHERE creator_id=?", (keep["id"], extra["id"]))
                if not self.db.one("SELECT 1 AS x FROM adapters WHERE creator_id=?", (keep["id"],)):
                    self.db.execute("UPDATE adapters SET creator_id=? WHERE creator_id=?", (keep["id"], extra["id"]))
                self.db.execute("DELETE FROM creators WHERE id=?", (extra["id"],)); merged += 1
        return merged

    # Creators ----------------------------------------------------------------------------------
    def nas_folders(self, relative: str = "") -> dict:
        """Liefert einen Ordner des erlaubten NAS-Baums fuer den Dashboard-Auswahldialog."""
        root = self.config.nas_root
        parts = Path(relative).parts
        if Path(relative).is_absolute() or any(part in ("", ".", "..") for part in parts):
            raise ValueError("Ungueltiger NAS-Ordner.")
        folder = root.joinpath(*parts)
        try: folder.relative_to(root)
        except ValueError as error: raise ValueError("Ordner liegt nicht auf dem erlaubten NAS.") from error
        if not folder.is_dir(): raise FileNotFoundError("NAS-Ordner nicht gefunden.")
        try:
            children = sorted((p for p in folder.iterdir() if p.is_dir() and not p.name.startswith((".", "$"))), key=lambda p: p.name.casefold())[:500]
        except OSError as error: raise ValueError(f"NAS-Ordner kann nicht gelesen werden: {error}") from error
        return {"root": str(root), "path": str(folder.relative_to(root)), "display": str(folder),
                "folders": [{"name": p.name, "path": str(p.relative_to(root))} for p in children]}

    def _nas_folder(self, source_dir: str) -> Path:
        """Normalisiert und begrenzt einen Creator-Ordner auf die NAS-Freigabe."""
        root = self.config.nas_root
        folder = Path(source_dir).expanduser()
        folder = folder if folder.is_absolute() else ROOT / folder
        try: folder.relative_to(root)
        except ValueError as error: raise ValueError(f"Ordner muss auf dem lokalen NAS liegen ({root}).") from error
        if not folder.is_dir(): raise ValueError(f"Ordner nicht gefunden: {folder}")
        return folder

    def creators(self) -> list[dict]:
        out = []
        for c in self.db.query("SELECT * FROM creators ORDER BY name"):
            stats = self.db.one("SELECT COUNT(*) AS tracks, COALESCE(SUM(duration),0)/3600.0 AS hours, SUM(features IS NOT NULL) AS analyzed, SUM(error IS NOT NULL) AS errors "
                                "FROM tracks WHERE creator_id=?", (c["id"],))
            adapter = self.db.one("SELECT clips,epochs,steps,loss,trained,base_model FROM adapters WHERE creator_id=?", (c["id"],))
            out.append({**c, "tracks": stats["tracks"], "hours": round(stats["hours"], 2), "analyzed": stats["analyzed"] or 0, "errors": stats["errors"] or 0,
                        "has_profile": get_profile(self.db, c["id"]) is not None, "adapter": adapter})
        return out

    def add_creator(self, name: str, source_dir: str, notes: str = "") -> dict:
        name = name.strip()
        if not name: raise ValueError("Name fehlt.")
        folder = Path(source_dir).expanduser() if source_dir else self.config.input / pipeline.slugify(name)
        if not source_dir: folder.mkdir(parents=True, exist_ok=True)
        folder = self._nas_folder(str(folder))
        slug = pipeline.slugify(name)
        if self.db.one("SELECT id FROM creators WHERE slug=?", (slug,)): raise ValueError(f"Creator '{slug}' existiert schon.")
        cid = self.db.execute("INSERT INTO creators(slug,name,source_dir,notes,created) VALUES(?,?,?,?,?)", (slug, name, str(folder), notes, time.time()))
        creator = self.db.one("SELECT * FROM creators WHERE id=?", (cid,)); added, total = pipeline.scan_creator(self.db, creator)
        return {**creator, "tracks": total}

    def delete_creator(self, creator_id: int, remove_adapter: bool = False) -> None:
        """Entfernt nur Datenbank-Eintraege (Quelldateien bleiben unveraendert). Der trainierte Adapter wird nur auf ausdruecklichen Wunsch geloescht."""
        creator = self.db.one("SELECT slug FROM creators WHERE id=?", (creator_id,))
        if not creator: raise KeyError("Creator nicht gefunden")
        self.db.execute("DELETE FROM creators WHERE id=?", (creator_id,))
        if remove_adapter: shutil.rmtree(self.config.models / creator["slug"], ignore_errors=True)

    def set_rights(self, creator_id: int, confirmed: bool, note: str = "") -> dict:
        if confirmed and not note.strip(): raise ValueError("Bitte kurz festhalten, woraus sich die Rechte ergeben (z. B. 'eigene Aufnahmen', 'Lizenz CC-BY', 'Vertrag vom ...').")
        self.db.execute("UPDATE creators SET rights=?, rights_note=?, rights_date=? WHERE id=?", (1 if confirmed else 0, note.strip() if confirmed else "", time.time() if confirmed else None, creator_id))
        return self.db.one("SELECT * FROM creators WHERE id=?", (creator_id,))

    def tracks(self, creator_id: int) -> list[dict]:
        rows = self.db.query("SELECT id,title,duration,features,error FROM tracks WHERE creator_id=? ORDER BY title", (creator_id,))
        out = []
        for r in rows:
            f = json.loads(r["features"]) if r["features"] else None
            out.append({"id": r["id"], "title": r["title"], "duration": r["duration"], "error": r["error"],
                        "bpm": f and f["bpm"], "key": f and f"{f['tonic']} {f['mode']}", "centroid_hz": f and f["centroid_hz"], "analyzed": f is not None})
        return out

    # Jobs --------------------------------------------------------------------------------------
    def start_analyze(self, creator_id: int, force: bool = False, limit: int | None = None) -> int:
        creator = self.db.one("SELECT name FROM creators WHERE id=?", (creator_id,))
        if not creator: raise KeyError("Creator nicht gefunden")
        return self.jobs.submit("analyze", f"Analyse: {creator['name']}", {"creator_id": creator_id, "force": force, "limit": limit})

    def failures(self) -> dict:
        """Alles, was fehlgeschlagen ist - Titel (Analyse), Stuecke (Erzeugen), Jobs (Analyse-/Trainings-Abbrueche). Nur diese werden wiederholt."""
        tracks = self.db.query("SELECT t.id,t.title,t.error,c.id AS creator_id,c.name AS creator FROM tracks t JOIN creators c ON c.id=t.creator_id WHERE t.error IS NOT NULL AND t.features IS NULL ORDER BY c.name,t.title")
        pieces = self.db.query("SELECT id,name,error FROM pieces WHERE status='error' ORDER BY id DESC")
        jobs = [{k: v for k, v in j.items() if k != "log"} for j in self.db.query("SELECT * FROM jobs WHERE status IN ('error','interrupted') AND kind IN ('analyze','train') ORDER BY id DESC")]
        for j in jobs:
            last = [l for l in (self.db.one("SELECT log FROM jobs WHERE id=?", (j["id"],))["log"] or "").splitlines() if "FEHLER" in l]
            j["error"] = last[-1].split("FEHLER:", 1)[-1].strip() if last else j["status"]
        return {"tracks": tracks, "pieces": pieces, "jobs": jobs, "total": len(tracks) + len(pieces) + len(jobs)}

    def retry_failures(self, kinds: list[str] | None = None) -> dict:
        """Wiederholt nur Fehlgeschlagenes: Titel (nur diese), Stuecke (neuer Render-Job), abgebrochene Analyse-/Trainings-Jobs. kinds: tracks|pieces|jobs (leer = alle)."""
        kinds = set(kinds or ("tracks", "pieces", "jobs")); fail = self.failures(); started: dict[str, list[int]] = {"tracks": [], "pieces": [], "jobs": []}
        covered: set[int] = set()
        if "jobs" in kinds:
            for j in fail["jobs"]:
                params = json.loads(j["params"] or "{}")
                if j["kind"] == "analyze":
                    if params.get("creator_id") in covered: self.jobs.delete(job_id=j["id"]); continue
                    covered.add(params.get("creator_id"))
                try: started["jobs"].append(self.jobs.retry(j["id"])); self.jobs.delete(job_id=j["id"])
                except (KeyError, ValueError) as error: log.warning("Job %s nicht wiederholbar: %s", j["id"], error)
        if "tracks" in kinds:
            for creator_id in sorted({t["creator_id"] for t in fail["tracks"]} - covered):
                name = next(t["creator"] for t in fail["tracks"] if t["creator_id"] == creator_id)
                started["tracks"].append(self.jobs.submit("analyze", f"Analyse (nur Fehlgeschlagene): {name}", {"creator_id": creator_id, "only_failed": True}))
        if "pieces" in kinds:
            for piece in fail["pieces"]:
                self.db.execute("UPDATE pieces SET status='queued', error=NULL, updated=? WHERE id=?", (time.time(), piece["id"]))
                started["pieces"].append(self.jobs.submit("render", f"Erzeugen (Wiederholung): {piece['name']}", {"piece_id": piece["id"]}))
            for j in self.db.query("SELECT id FROM jobs WHERE kind='render' AND status IN ('error','interrupted')"): self.jobs.delete(job_id=j["id"])
        return {"started": started, "count": sum(len(v) for v in started.values())}

    def start_train(self, creator_id: int) -> int:
        creator = self.db.one("SELECT name,rights FROM creators WHERE id=?", (creator_id,))
        if not creator: raise KeyError("Creator nicht gefunden")
        if not creator["rights"]: raise ValueError("Training gesperrt: Rechte fuer diesen Creator sind nicht bestaetigt.")
        if not self.db.one("SELECT 1 AS x FROM tracks WHERE creator_id=? AND features IS NOT NULL", (creator_id,)): raise ValueError("Zuerst die Analyse laufen lassen.")
        return self.jobs.submit("train", f"Training: {creator['name']}", {"creator_id": creator_id})

    # Stuecke ------------------------------------------------------------------------------------
    def mix_preview(self, weights: dict[str, float]) -> dict:
        styles = resolve_styles(self.db, self.config, weights)
        return {"weights": styles["weights"], "profile": styles["profile"], "warnings": styles["warnings"], "adapters": [s for s, _, _ in styles["adapters"]]}

    def create_piece(self, params: dict) -> dict:
        weights = {k: float(v) for k, v in (params.get("styles") or {}).items() if float(v) > 0}
        if weights: resolve_styles(self.db, self.config, weights)      # frueh melden, wenn ein Stil unbekannt ist
        seconds = max(5.0, min(float(params.get("seconds", 30)), 1800.0))
        clean = {"styles": normalize_weights(weights) if weights else {}, "prompt": str(params.get("prompt", ""))[:500], "seconds": seconds,
                 "strength": max(0.0, min(float(params.get("strength", 1.0)), 2.0)), "use_profile": bool(params.get("use_profile", True)),
                 "seed": int(params["seed"]) if params.get("seed") not in (None, "") else None}
        for key in ("guidance_scale", "temperature"):
            if params.get(key): clean[key] = float(params[key])
        name = (params.get("name") or "").strip() or (", ".join(clean["styles"]) or "Stueck")[:60]
        now = time.time(); pid = self.db.execute("INSERT INTO pieces(name,params,status,created,updated) VALUES(?,?,?,?,?)", (name, dumps(clean), "queued", now, now))
        job = self.jobs.submit("render", f"Erzeugen: {name} ({seconds:.0f} s)", {"piece_id": pid})
        return {"id": pid, "job": job, "name": name}

    def pieces(self) -> list[dict]:
        return [{**p, "params": json.loads(p["params"])} for p in self.db.query("SELECT * FROM pieces ORDER BY id DESC")]

    def delete_piece(self, piece_id: int) -> None:
        shutil.rmtree(self.config.tracks / str(piece_id), ignore_errors=True)
        self.db.execute("DELETE FROM pieces WHERE id=?", (piece_id,))

    # Status ------------------------------------------------------------------------------------
    def overview(self) -> dict:
        creators = self.creators()
        return {"creators": len(creators), "tracks": sum(c["tracks"] for c in creators), "hours": round(sum(c["hours"] for c in creators), 1),
                "trained": sum(1 for c in creators if c["adapter"]), "pieces": self.db.one("SELECT COUNT(*) AS n FROM pieces")["n"],
                "jobs": len(self.jobs.active()), "base_model": self.config.get("music", "base_model")}

    def gpu(self) -> dict | None:
        """GPU-Auslastung per nvidia-smi (importiert kein torch, ist daher schnell); 2 s zwischengespeichert."""
        cached = getattr(self, "_gpu", None)
        if cached and time.monotonic() - cached[0] < 2: return cached[1]
        value = None
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=4, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if out.returncode == 0 and out.stdout.strip():
                name, util, used, total, temp, power = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
                num = lambda x: float(x) if x.replace(".", "", 1).isdigit() else None
                value = {"name": name, "util": num(util), "mem_used": num(used), "mem_total": num(total), "temp": num(temp), "power": num(power)}
        except (OSError, subprocess.SubprocessError): pass
        self._gpu = (time.monotonic(), value)
        return value

    def dashboard(self) -> dict:
        creators = []
        for c in self.creators():
            profile = get_profile(self.db, c["id"]) if c["has_profile"] else None
            creators.append({"id": c["id"], "slug": c["slug"], "name": c["name"], "tracks": c["tracks"], "hours": c["hours"], "analyzed": c["analyzed"], "errors": c["errors"],
                             "rights": bool(c["rights"]), "adapter": c["adapter"],
                             "profile": profile and {k: profile[k] for k in ("bpm", "bpm_min", "bpm_max", "keys", "centroid_hz", "dynamic_range_db", "onset_rate", "bands", "energy_arc", "caption")}})
        jobs = [{k: v for k, v in j.items() if k != "log"} for j in self.db.query("SELECT * FROM jobs ORDER BY id DESC LIMIT 8")]
        for j in jobs:
            ctx = self.jobs.contexts.get(j["id"])
            if ctx and j["status"] == "running": j["progress"] = ctx._progress; j["current"] = ctx.current; j["elapsed"] = time.monotonic() - ctx.started
        pieces = [{"id": p["id"], "name": p["name"], "status": p["status"], "seconds": json.loads(p["params"]).get("seconds"), "styles": json.loads(p["params"]).get("styles", {}), "created": p["created"]}
                  for p in self.db.query("SELECT * FROM pieces ORDER BY id DESC LIMIT 6")]
        return {"overview": self.overview(), "gpu": self.gpu(), "creators": creators, "jobs": jobs, "pieces": pieces,
                "inbox": self.watcher.status() if self.watcher else {"pending": [], "auto_analyze": False}, "failures": self.failures(), "time": time.time()}

    def doctor(self) -> dict:
        from .version import VERSION
        have = lambda mod: importlib.util.find_spec(mod) is not None
        out = {"MusicAI-Version": VERSION,
               "python-pakete (numpy/scipy/soundfile/yaml)": all(have(m) for m in ("numpy", "scipy", "soundfile", "yaml")),
               "ffmpeg": Path(self.config.ffmpeg).exists() or shutil.which(self.config.ffmpeg), "ffprobe": Path(self.config.ffprobe).exists() or shutil.which(self.config.ffprobe),
               "torch": have("torch"), "transformers": have("transformers"), "bitsandbytes (4-Bit fuer large)": have("bitsandbytes")}
        if have("torch"):
            import torch
            out["cuda"] = torch.cuda.is_available() and f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)"
        base = str(self.config.get("music", "base_model", "")); cache = self.config.cache
        out[f"basismodell {base}"] = any(cache.glob(f"models--{base.replace('/', '--')}")) or "wird beim ersten Training/Erzeugen geladen" if self.config.get("music", "allow_download", True) else any(cache.glob(f"models--{base.replace('/', '--')}"))
        out["eingangsordner"] = str(self.config.input)
        return {k: v for k, v in out.items()}

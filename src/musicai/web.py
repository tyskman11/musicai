"""WebUI-Server (nur Standardbibliothek): JSON-API + statische Single-Page-App, standardmaessig nur auf 127.0.0.1."""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .app import App
from .logging_setup import tail
from .profile import build_profile, get_profile

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("musicai.web")
ROUTES: list[tuple[str, re.Pattern, str]] = []


def route(method: str, pattern: str):
    def decorator(fn):
        ROUTES.append((method, re.compile("^" + pattern + "$"), fn.__name__)); return fn
    return decorator


class Api:
    """Alle Endpunkte. Rueckgabe: JSON-serialisierbar oder (status, dict)."""
    def __init__(self, app: App) -> None: self.app = app

    @route("GET", "/api/overview")
    def overview(self, q, body): return self.app.overview()

    @route("GET", "/api/dashboard")
    def dashboard(self, q, body): return self.app.dashboard()

    @route("GET", "/api/failures")
    def failures(self, q, body): return self.app.failures()

    @route("POST", "/api/failures/retry")
    def retry_failures(self, q, body): return self.app.retry_failures(body.get("kinds"))

    @route("GET", "/api/doctor")
    def doctor(self, q, body): return self.app.doctor()

    @route("GET", "/api/logs")
    def logs(self, q, body):
        name = (q.get("name") or ["errors"])[0]; lines = int((q.get("lines") or ["200"])[0])
        return {"name": name, "text": tail(name, lines)}

    @route("GET", "/api/inbox")
    def inbox(self, q, body): return self.app.watcher.status() if self.app.watcher else {"pending": [], "auto_analyze": False}

    @route("POST", "/api/inbox/scan")
    def inbox_scan(self, q, body):
        if not self.app.watcher: raise ValueError("Waechter laeuft nicht")
        return self.app.watcher.check()

    # Creators ----------------------------------------------------------------------------------
    @route("GET", "/api/creators")
    def creators(self, q, body): return self.app.creators()

    @route("POST", "/api/folders/list")
    def folders(self, q, body): return self.app.nas_folders(str(body.get("path") or ""))

    @route("POST", "/api/creators")
    def add_creator(self, q, body): return self.app.add_creator(body.get("name", ""), body.get("folder", ""), body.get("notes", ""))

    @route("DELETE", r"/api/creators/(\d+)")
    def delete_creator(self, q, body, cid): self.app.delete_creator(int(cid), (q.get("adapter") or ["0"])[0] == "1"); return {"ok": True}

    @route("POST", r"/api/creators/(\d+)/rights")
    def rights(self, q, body, cid): return self.app.set_rights(int(cid), bool(body.get("confirmed")), body.get("note", ""))

    @route("POST", r"/api/creators/(\d+)/analyze")
    def analyze(self, q, body, cid): return {"job": self.app.start_analyze(int(cid), bool(body.get("force")), body.get("limit"))}

    @route("POST", r"/api/creators/(\d+)/train")
    def train(self, q, body, cid): return {"job": self.app.start_train(int(cid))}

    @route("GET", r"/api/creators/(\d+)/tracks")
    def tracks(self, q, body, cid): return self.app.tracks(int(cid))

    @route("GET", r"/api/creators/(\d+)/profile")
    def profile(self, q, body, cid): return get_profile(self.app.db, int(cid)) or {}

    @route("POST", r"/api/creators/(\d+)/profile")
    def rebuild_profile(self, q, body, cid): return build_profile(self.app.db, int(cid))

    # Mischen / Stuecke -------------------------------------------------------------------------
    @route("POST", "/api/mix")
    def mix(self, q, body): return self.app.mix_preview({k: float(v) for k, v in (body.get("styles") or {}).items()})

    @route("GET", "/api/pieces")
    def pieces(self, q, body): return self.app.pieces()

    @route("POST", "/api/pieces")
    def create_piece(self, q, body): return self.app.create_piece(body)

    @route("DELETE", r"/api/pieces/(\d+)")
    def delete_piece(self, q, body, pid): self.app.delete_piece(int(pid)); return {"ok": True}

    # Jobs --------------------------------------------------------------------------------------
    @route("GET", "/api/jobs")
    def jobs(self, q, body):
        limit = int((q.get("limit") or ["50"])[0])
        return [{k: v for k, v in row.items() if k != "log"} for row in self.app.db.query("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]

    @route("GET", r"/api/jobs/(\d+)")
    def job(self, q, body, jid): return self.app.db.one("SELECT * FROM jobs WHERE id=?", (int(jid),)) or (404, {"error": "nicht gefunden"})

    @route("DELETE", r"/api/jobs/(\d+)")
    def delete_job(self, q, body, jid): return {"deleted": self.app.jobs.delete(job_id=int(jid))}

    @route("POST", "/api/restart")
    def restart(self, q, body):
        """Beendet den Server mit Code 75; start.py (im Terminal) startet dann sich selbst und alles neu - Code-Aenderungen gelten sofort."""
        if os.environ.get("MUSICAI_START_CHILD") != "1":
            return 409, {"error": "Nur moeglich, wenn MusicAI ueber start.py gestartet wurde."}
        threading.Timer(0.5, lambda: os._exit(75)).start()
        return {"ok": True}

    @route("POST", "/api/jobs/diagnose")
    def diagnose_jobs(self, q, body):
        """Nur fehlgeschlagene Jobs (Status 'error'): gruppiert, mit Ursache/Gegenmassnahme; mit {llm:true, group:id} zusaetzlich LLM-Erklaerung."""
        from . import diagnose
        groups = diagnose.analyze(self.app.db)
        if body.get("llm"):
            groups = [g for g in groups if g["id"] == body.get("group")][:1]
            for g in groups: g["explanation"] = diagnose.explain(self.app.config, g)
        return {"groups": groups}

    @route("POST", "/api/jobs/clear")
    def clear_jobs(self, q, body): return {"deleted": self.app.jobs.delete(statuses=body.get("statuses"))}

    @route("POST", r"/api/jobs/(\d+)/retry")
    def retry_job(self, q, body, jid): return {"job": self.app.jobs.retry(int(jid))}

    @route("POST", r"/api/jobs/(\d+)/cancel")
    def cancel(self, q, body, jid): return {"ok": self.app.jobs.cancel(int(jid))}


def make_handler(app: App, host: str, port: int):
    api = Api(app)
    allowed_hosts = {f"{host}:{port}", f"localhost:{port}", f"127.0.0.1:{port}"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "MusicAI"

        def log_message(self, fmt, *args): pass

        def _send(self, status: int, payload: bytes, content_type: str, extra: dict | None = None) -> None:
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items(): self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD": self.wfile.write(payload)

        def _json(self, status: int, data) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _guard(self) -> bool:
            """Schuetzt gegen DNS-Rebinding und fremde Webseiten (CSRF) bei lokalem Betrieb."""
            if self.headers.get("Host", "") not in allowed_hosts: self._json(403, {"error": "Host nicht erlaubt"}); return False
            origin = self.headers.get("Origin")
            # Hinter dem permanenten Access-Proxy bleibt Host lokal (127.0.0.1),
            # Origin ist aber die oeffentliche Adresse. Der Proxy liefert seinen
            # urspruenglichen Host in X-Forwarded-Host; diesen akzeptieren wir als
            # Same-Origin, ohne fremde Origins zu oeffnen.
            forwarded = {value.strip() for value in self.headers.get("X-Forwarded-Host", "").split(",") if value.strip()}
            trusted_origins = allowed_hosts | forwarded
            if self.command != "GET" and origin and urlparse(origin).netloc not in trusted_origins:
                log.warning("CSRF guard rejected %s from origin=%r host=%r forwarded=%r", self.path, origin, self.headers.get("Host"), sorted(forwarded))
                self._json(403, {"error": "Origin nicht erlaubt"}); return False
            return True

        def _media(self, path: Path) -> None:
            if not path.is_file(): self._json(404, {"error": "Datei nicht gefunden"}); return
            size = path.stat().st_size; start, end, status = 0, size - 1, 200
            match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if match and (match.group(1) or match.group(2)):
                if match.group(1): start = int(match.group(1))
                if match.group(2): end = min(int(match.group(2)), size - 1)
                if not match.group(1): start = max(0, size - int(match.group(2))); end = size - 1
                status = 206
            length = end - start + 1
            self.send_response(status); self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(length))
            if status == 206: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD": return
            try:
                with open(path, "rb") as handle:
                    handle.seek(start); remaining = length
                    while remaining > 0:
                        chunk = handle.read(min(1 << 20, remaining))
                        if not chunk: break
                        self.wfile.write(chunk); remaining -= len(chunk)
            except (ConnectionError, OSError): pass

        def _dispatch(self) -> None:
            try: self._dispatch_inner()
            except (ConnectionError, BrokenPipeError, TimeoutError): pass     # Browser hat die Verbindung geschlossen

        def _dispatch_inner(self) -> None:
            if not self._guard(): return
            parsed = urlparse(self.path); path = parsed.path
            if self.command in ("GET", "HEAD") and not path.startswith("/api/"): return self._static(path)
            body = {}
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                try: body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError: return self._json(400, {"error": "Ungueltiges JSON"})
            method = "GET" if self.command == "HEAD" else self.command
            for route_method, pattern, name in ROUTES:
                match = pattern.match(path)
                if match and route_method == method:
                    try:
                        result = getattr(api, name)(parse_qs(parsed.query), body, *match.groups())
                        if isinstance(result, tuple): return self._json(result[0], result[1])
                        return self._json(200, result)
                    except (ValueError, KeyError, FileNotFoundError) as error: return self._json(400, {"error": str(error.args[0] if error.args else error)})
                    except ConnectionError: raise
                    except Exception as error:  # noqa: BLE001
                        log.error("%s %s -> %s: %s", self.command, path, type(error).__name__, error, exc_info=True); return self._json(500, {"error": f"{type(error).__name__}: {error}"})
            self._json(404, {"error": "Unbekannter Endpunkt"})

        def _static(self, path: str) -> None:
            if (m := re.match(r"^/media/piece/(\d+)\.(wav|mp3)$", path)):
                folder = app.config.tracks / m.group(1)
                return self._media(folder / f"track.{m.group(2)}")
            if (m := re.match(r"^/media/source/(\d+)$", path)):      # Original eines registrierten Titels zum Probehoeren
                row = app.db.one("SELECT path FROM tracks WHERE id=?", (int(m.group(1)),))
                return self._media(Path(row["path"])) if row else self._json(404, {"error": "Titel nicht gefunden"})
            if path in ("/", "/index.html"): return self._media(STATIC / "index.html")
            file = (STATIC / path.lstrip("/")).resolve()
            if STATIC.resolve() in file.parents and file.is_file(): return self._media(file)
            self._json(404, {"error": "Nicht gefunden"})

        do_GET = do_POST = do_DELETE = do_HEAD = _dispatch

    return Handler


def serve(app: App, host: str | None = None, port: int | None = None) -> None:
    host = host or app.config.get("web", "host", "127.0.0.1"); port = int(port or app.config.get("web", "port", 8790))

    class QuietServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)): return      # Browser hat die Verbindung geschlossen
            super().handle_error(request, client_address)

    # Erst den Port reservieren, dann Hintergrundarbeit starten. Bei einem zweiten
    # Startversuch werden so keine Watcher/DB-Zugriffe erzeugt, bevor die klare
    # Meldung "Adresse bereits verwendet" kommt.
    server = QuietServer((host, port), make_handler(app, host, port)); server.daemon_threads = True
    from .watcher import Watcher
    app.watcher = Watcher(app); app.watcher.start()
    print(f"MusicAI WebUI: http://{host}:{port}", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nBeendet.")

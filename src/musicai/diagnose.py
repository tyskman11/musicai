"""Analyse fehlgeschlagener Jobs (nur Status 'error'): gruppiert gleiche Fehler, nennt Ursache und Gegenmassnahme, optional LLM-Erklaerung."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request

PATTERNS = [
    (r"bitsandbytes", "4/8-Bit-Quantisierung nicht verfuegbar",
     "Der konfigurierte Modellmodus braucht bitsandbytes. `python start.py setup --yes` erstellt/aktualisiert die eigene MusicAI-Umgebung; danach neu starten. Alternativ quantize: none und ein kleineres Modell waehlen."),
    (r"decoder_start_token_id", "MusicGen-Konfiguration unvollstaendig",
     "Die Transformers-/MusicGen-Konfiguration passt nicht zum Trainingspfad. MusicAI auf den aktuellen Stand bringen und den Job neu starten."),
    (r"out of memory|OutOfMemory|CUDA error|CUBLAS|CUDNN", "GPU-Speicher voll",
     "Zu wenig VRAM. ComfyUI/Whisper/XTTS der anderen Projekte beenden, kleinere Ausschnitte/Batches waehlen oder auf CPU ausweichen."),
    (r"No module named|ModuleNotFoundError|ImportError", "Python-Paket fehlt",
     "`python start.py setup` ausfuehren: installiert die fehlenden Pakete in die projekteigene .venv."),
    (r"ffmpeg|ffprobe|moov atom|Invalid data found|Invalid argument", "ffmpeg/Mediendatei",
     "ffmpeg fehlt oder die Datei ist beschaedigt/unvollstaendig kopiert. Datei neu kopieren bzw. mit einem Player pruefen, dann erneut starten."),
    (r"PermissionError|WinError 5\b|WinError 32|Zugriff verweigert|used by another process", "Zugriff verweigert / Datei gesperrt",
     "Die Datei ist durch ein anderes Programm oder den Virenscanner gesperrt (auf Netzlaufwerken haeufig). Spaeter erneut versuchen."),
    (r"FileNotFoundError|No such file|WinError 2\b|WinError 3\b|nicht gefunden|nicht lokal vorhanden", "Datei, Ordner oder Modell fehlt",
     "Die Quelle wurde verschoben/geloescht, liegt auf einem nicht erreichbaren Laufwerk, oder das Modell ist nicht heruntergeladen."),
    (r"Connection refused|WinError 10061|ConnectionError|URLError|timed out|Timeout|nicht erreichbar", "Dienst nicht erreichbar",
     "ComfyUI bzw. der LLM-Server laeuft nicht oder antwortet zu langsam. Im Terminal von start.py pruefen und den Dienst starten."),
    (r"Rechte|rights", "Rechte-Bestaetigung fehlt",
     "Fuer diesen Creator ist keine Rechte-Bestaetigung samt Grundlage gesetzt. Im Dashboard eintragen und erneut starten."),
    (r"database is locked|sqlite3", "Datenbank gesperrt",
     "Die SQLite-Datei war kurz gesperrt (Netzlaufwerk/parallele Zugriffe). Erneut versuchen."),
    (r"Abgebrochen|Interrupted|KeyboardInterrupt", "Vorzeitig beendet", "Der Prozess wurde beendet (Neustart oder Strg+C). Erneut starten."),
]


def _signature(log: str) -> str:
    lines = [line.strip() for line in (log or "").splitlines() if line.strip()]
    error_lines = [line for line in lines if "FEHLER" in line or "Error" in line or "Exception" in line]
    text = (error_lines[-1] if error_lines else (lines[-1] if lines else "(kein Log)"))
    text = re.sub(r"^\[\d\d:\d\d:\d\d\]\s*", "", text)
    text = re.sub(r"[A-Za-z]:[\\/][^\s'\"]+", "<pfad>", text)
    text = re.sub(r"\d+", "#", text)
    return text[:160]


def _classify(log: str) -> tuple[str, str]:
    for pattern, title, hint in PATTERNS:
        if re.search(pattern, log or "", re.IGNORECASE):
            return title, hint
    return "Unbekannter Fehler", "Keine bekannte Ursache erkannt. Die Fehlermeldung unten zeigt die Details; \"Mit lokalem LLM erklaeren\" liefert eine Einschaetzung."


def analyze(db, limit: int = 300) -> list[dict]:
    rows = db.query("SELECT id,kind,title,log,finished FROM jobs WHERE status='error' ORDER BY id DESC LIMIT ?", (limit,))
    groups: dict[str, dict] = {}
    for row in rows:
        log = row.get("log") or ""
        signature = _signature(log)
        title, hint = _classify(log)
        key = hashlib.sha1((title + "|" + signature).encode("utf-8")).hexdigest()[:8]
        group = groups.setdefault(key, {"id": key, "title": title, "hint": hint, "signature": signature, "count": 0,
                                        "kinds": [], "jobs": [], "trace": log[-1800:]})
        group["count"] += 1
        if row["kind"] not in group["kinds"]: group["kinds"].append(row["kind"])
        group["jobs"].append({"id": row["id"], "title": row["title"], "finished": row.get("finished")})
    return sorted(groups.values(), key=lambda g: -g["count"])


def explain(config, group: dict) -> str:
    """Kurze Erklaerung durch das lokale LLM (llama-server aus SeriesAI/start.py); ohne LLM eine klare Meldung."""
    base = (config.get("llm", "base_url", None) or "http://127.0.0.1:11435").rstrip("/")
    prompt = ("Du hilfst bei der Fehlersuche in einer lokalen Python-Anwendung (Windows, Netzlaufwerk, ffmpeg, PyTorch). "
              "Erklaere auf Deutsch in hoechstens 6 Saetzen die wahrscheinliche Ursache und nenne konkrete naechste Schritte.\n\n"
              f"Fehlerart: {group['title']}\nJobs: {', '.join(group['kinds'])}\nLog-Ende:\n{group['trace']}")
    body = json.dumps({"model": "local", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=120) as response:
            return json.loads(response.read())["choices"][0]["message"]["content"].strip()
    except Exception as error:  # noqa: BLE001 - LLM optional
        return f"Das lokale LLM ist nicht erreichbar ({type(error).__name__}). Es wird von SeriesAI/start.py gestartet (Port 11435)."

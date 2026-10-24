from __future__ import annotations

import os
import shutil
import string
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

def _prefer_mapped_nas_path(path: Path) -> Path:
    """Nimmt auf Windows den vorhandenen lokalen Laufwerksbuchstaben fuer UNC.

    SQLite arbeitet mit dem gemappten, lokalen NAS-Laufwerk stabiler als mit
    einem UNC-Pfad. Ist keine Zuordnung vorhanden (oder kein Windows), bleibt
    der urspruengliche Pfad unveraendert.
    """
    raw = str(path)
    if os.name != "nt" or not raw.startswith("\\\\"):
        return path
    for letter in string.ascii_uppercase:
        drive = f"{letter}:"
        buffer = ctypes.create_unicode_buffer(2048)
        length = ctypes.c_uint(len(buffer))
        # ERROR_NOT_CONNECTED (2250) ist fuer lokale Laufwerke normal.
        if ctypes.windll.mpr.WNetGetConnectionW(drive, buffer, ctypes.byref(length)) != 0:
            continue
        remote = buffer.value.rstrip("\\/")
        if raw.casefold().startswith((remote + "\\").casefold()):
            return Path(drive + "\\" + raw[len(remote):].lstrip("\\/"))
    return path


ROOT = _prefer_mapped_nas_path(Path(__file__).resolve().parents[2])


def _resolve(value: str | None) -> Path | None:
    if value is None: return None
    path = Path(value)
    # ``Path.resolve()`` verwandelt ein gemapptes B:\\ auf Windows wieder in
    # UNC. Das ist fuer SQLite gerade der problematische Pfad; deshalb bleibt
    # die lokale NAS-Zuordnung absichtlich erhalten.
    return _prefer_mapped_nas_path(path) if path.is_absolute() else ROOT / path


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or ROOT / "config.yaml"
        return cls((yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {})

    def get(self, section: str, key: str, default: Any = None) -> Any:
        value = (self.raw.get(section) or {}).get(key, default)
        return default if value is None and default is not None else value

    def path(self, section: str, key: str, default: str | None = None) -> Path | None:
        return _resolve(self.get(section, key, default))

    def _dir(self, section: str, key: str, default: str) -> Path:
        path = self.path(section, key, default); path.mkdir(parents=True, exist_ok=True); return path

    @property
    def data(self) -> Path: return self._dir("paths", "data", "data")
    @property
    def input(self) -> Path: return self._dir("paths", "input", "input")
    @property
    def models(self) -> Path: return self._dir("paths", "models", "models")
    @property
    def db_path(self) -> Path: return self.data / "musicai.db"
    @property
    def work(self) -> Path: return self._sub("work")
    @property
    def tracks(self) -> Path: return self._sub("tracks")
    @property
    def cache(self) -> Path: return self._dir("music", "cache", "runtime/cache/huggingface")

    @property
    def nas_root(self) -> Path:
        """Einzige erlaubte Quelle fuer verknuepfte Creator-Ordner.

        Bei einem UNC-Projektpfad ist ``ROOT.anchor`` die NAS-Freigabe. Eine
        optionale Konfiguration kann den Browser weiter auf einen Teilbereich
        dieser Freigabe begrenzen.
        """
        configured = self.path("paths", "nas_root", None)
        return configured or Path(ROOT.anchor)

    def _sub(self, name: str) -> Path:
        path = self.data / name; path.mkdir(parents=True, exist_ok=True); return path

    @property
    def ffmpeg(self) -> str:
        local = ROOT / "runtime" / "ffmpeg" / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")      # vom Starter ins Projekt geladen
        if local.exists(): return str(local)
        configured = self.get("paths", "ffmpeg", "ffmpeg")
        return shutil.which(configured) or configured

    @property
    def ffprobe(self) -> str:
        ffmpeg = Path(self.ffmpeg)
        sibling = ffmpeg.with_name("ffprobe" + ffmpeg.suffix)
        return str(sibling) if sibling.exists() else (shutil.which("ffprobe") or "ffprobe")

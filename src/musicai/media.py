"""ffmpeg/ffprobe-Zugriff: Dauer ermitteln, Audio als float32-Mono dekodieren (Streaming), WAV/MP3 schreiben."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .config import Config

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}      # Audiospur wird genutzt
MEDIA_EXT = AUDIO_EXT | VIDEO_EXT


def probe(config: Config, path: str | Path) -> dict:
    """Dauer und Audio-Eckdaten (leeres Dict, wenn keine Audiospur)."""
    out = subprocess.run([config.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    if out.returncode != 0: raise RuntimeError(f"ffprobe fehlgeschlagen: {out.stderr.strip()[:200]}")
    info = json.loads(out.stdout or "{}")
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not audio: raise RuntimeError("keine Audiospur")
    duration = float(info.get("format", {}).get("duration") or audio.get("duration") or 0)
    return {"duration": duration, "channels": int(audio.get("channels") or 0), "sample_rate": int(audio.get("sample_rate") or 0), "codec": audio.get("codec_name")}


def decode(config: Config, path: str | Path, sample_rate: int, start: float = 0.0, seconds: float | None = None, channels: int = 1) -> np.ndarray:
    """Audio -> float32 [-1,1]; Mono (n,) oder Stereo (n,2)."""
    cmd = [config.ffmpeg, "-v", "error", "-nostdin"]
    if start: cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if seconds: cmd += ["-t", f"{seconds:.3f}"]
    cmd += ["-vn", "-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW)
    if out.returncode != 0 and not out.stdout: raise RuntimeError(f"ffmpeg Dekodierung fehlgeschlagen: {out.stderr.decode(errors='replace')[:200]}")
    data = np.frombuffer(out.stdout, dtype="<f4")
    return data.reshape(-1, channels) if channels > 1 else data.copy()


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")


def to_mp3(config: Config, wav: Path, mp3: Path, bitrate: str = "192k") -> None:
    subprocess.run([config.ffmpeg, "-v", "error", "-y", "-i", str(wav), "-b:a", bitrate, str(mp3)], check=True, capture_output=True, creationflags=NO_WINDOW)

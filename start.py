"""MusicAI starten - legt bei Bedarf selbst eine virtuelle Umgebung an und installiert die Abhaengigkeiten.

    python start.py              # WebUI starten (Adresse wird im Terminal angezeigt, es wird nie ein Tab geoeffnet)
    python start.py doctor       # Umgebung pruefen
    python start.py setup        # nur Umgebung anlegen/aktualisieren (--core: ohne KI-Pakete, --yes: ohne Rueckfrage)
    python start.py <befehl>     # beliebiger CLI-Befehl (analyze, train, generate, ...)

Dieses Skript nutzt ausschliesslich die Standardbibliothek, laeuft also mit jedem Python 3.10+.
Reihenfolge der Interpreter: ./.venv, ../SeriesAI/.venv (teilt dessen Pakete) - nie ein globaler Python.
Gibt es keinen vollstaendigen, wird ./.venv ohne Rueckfrage erzeugt und aus requirements.txt (+ requirements-full.txt) befuellt;
ffmpeg wird bei Bedarf nach ./runtime/ffmpeg geladen (oder von ../ASMRAI/runtime/ffmpeg uebernommen). Caches/Temp liegen in ./runtime.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def acquire_server_lock():
    """Hält bei ``serve`` genau eine Instanz pro Projekt aktiv.

    Ein offen gehaltener Byte-Lock funktioniert auf Windows auch bei einem
    UNC-/SMB-Pfad. Die Sperre wird beim Prozessende automatisch freigegeben;
    die Datei selbst ist nur ein harmloser Marker.
    """
    lock_path = ROOT / "runtime" / "musicai.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    try:
        handle.seek(0)
        if not handle.read(1):
            handle.seek(0); handle.write("0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle

# Keine globalen Verzeichnisse: Caches und Temp-Dateien liegen im Projekt (werden an alle Unterprozesse vererbt).
_LOCAL = ROOT / "runtime" / "cache"
for _name, _sub in (("HF_HOME", "huggingface"), ("TORCH_HOME", "torch"), ("XDG_CACHE_HOME", "xdg"),
                    ("PIP_CACHE_DIR", "pip"), ("TMP", "tmp"), ("TEMP", "tmp"), ("TMPDIR", "tmp")):
    (_LOCAL / _sub).mkdir(parents=True, exist_ok=True)
    os.environ[_name] = str(_LOCAL / _sub)


# ---- portabler Python (python-build-standalone): liegt im Projekt bzw. im gemeinsamen runtime-Ordner, nie in einer globalen Installation ----
PORTABLE_PY_MINOR = "3.11"      # 3.10-3.12 haben fuer alle KI-Pakete (torch, coqui-tts ...) Wheels


def portable_python() -> Path | None:
    exe = "python.exe" if os.name == "nt" else "bin/python3"
    for base in (ROOT / "runtime" / "python", ROOT.parent.parent / "runtime" / "python"):   # projekteigen, sonst gemeinsam im Pool-Ordner
        if (base / exe).exists() and (base / ".install_complete").exists(): return base / exe   # Marker: nur vollstaendig entpackte Installationen
    return None


def ensure_portable_python() -> Path | None:
    """Gibt den portablen Python zurueck; fehlt er, wird er einmalig (mit SHA-256-Pruefung) geladen. None = offline/fehlgeschlagen."""
    import hashlib, json, tarfile, urllib.request
    found = portable_python()
    if found: return found
    platform_tag = "x86_64-pc-windows-msvc" if os.name == "nt" else ("aarch64-apple-darwin" if sys.platform == "darwin" else "x86_64-unknown-linux-gnu")
    target = ROOT.parent.parent / "runtime" / "python" if ROOT.parent.name == "projects" else ROOT / "runtime" / "python"
    try:
        request = urllib.request.Request("https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest", headers={"User-Agent": "portable-python"})
        with urllib.request.urlopen(request, timeout=30) as response: release = json.load(response)
        pattern = re.compile(rf"^cpython-{re.escape(PORTABLE_PY_MINOR)}\.\d+\+\d+-{platform_tag}-install_only\.tar\.gz$")
        asset = next(a for a in release["assets"] if pattern.match(a["name"]))
        digest = str(asset.get("digest", ""))
        if not digest.startswith("sha256:"): raise RuntimeError("Release ohne Pruefsumme")
        archive = ROOT / "runtime" / "downloads" / asset["name"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Python] Lade portables Python {PORTABLE_PY_MINOR} ({asset['size'] >> 20} MB) nach {target} ...", flush=True)
        sha = hashlib.sha256()
        with urllib.request.urlopen(asset["browser_download_url"], timeout=60) as response, archive.open("wb") as out:
            while chunk := response.read(1 << 20): out.write(chunk); sha.update(chunk)
        if sha.hexdigest() != digest[7:]: archive.unlink(missing_ok=True); raise RuntimeError("Pruefsumme stimmt nicht")
        staging = target.with_name("python.staging")
        shutil.rmtree(staging, ignore_errors=True)
        skipped = 0
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            for number, member in enumerate(members, 1):      # einzeln entpacken: auf Netzlaufwerken scheitert gelegentlich eine Datei (z. B. leere .lock-Dateien)
                if number % 250 == 0: print(f"[Python] Entpacke ... {number}/{len(members)} Dateien", flush=True)
                for attempt in range(3):
                    try:
                        bundle.extract(member, staging, filter="data") if hasattr(tarfile, "data_filter") else bundle.extract(member, staging)
                        break
                    except PermissionError:
                        if attempt == 2:
                            if not member.name.endswith(".lock"): skipped += 1
                            break
                        time.sleep(1)
        if skipped or not (staging / "python" / ("python.exe" if os.name == "nt" else "bin/python3")).exists():
            raise RuntimeError(f"Entpacken unvollstaendig ({skipped} Dateien nicht schreibbar; laeuft ein Virenscanner/anderer Prozess auf dem Ziel?)")
        archive.unlink(missing_ok=True)
        shutil.rmtree(target, ignore_errors=True)
        for attempt in range(5):   # Netzlaufwerke/Virenscanner halten frisch entpackte Ordner kurz fest
            try: (staging / "python").replace(target); break
            except PermissionError:
                if attempt == 4: shutil.move(str(staging / "python"), str(target))
                else: time.sleep(2)
        shutil.rmtree(staging, ignore_errors=True)
        if not portable_python():
            shutil.rmtree(target, ignore_errors=True); raise RuntimeError("python nach dem Verschieben nicht vorhanden")
        (target / ".install_complete").write_text("ok", encoding="utf-8")
        if not portable_python():
            shutil.rmtree(target, ignore_errors=True); raise RuntimeError("Marker konnte nicht geschrieben werden")
        print("[Python] Portabler Python bereit.", flush=True)
    except Exception as error:  # noqa: BLE001 - offline o. ae.: Aufrufer weicht auf den startenden Python aus
        print(f"[Python] Portabler Python nicht verfuegbar ({type(error).__name__}: {error}); verwende den startenden Python.", flush=True)
        return None
    return portable_python()
EXE = "Scripts/python.exe" if os.name == "nt" else "bin/python"
CORE = ("numpy", "scipy", "soundfile", "yaml")                  # Basis: Analyse, WebUI, Profile, Mischen der Profile
FULL = CORE + ("torch", "transformers")                          # KI: Training und Erzeugung
SUPPORTED = ((3, 10), (3, 11), (3, 12))                          # Versionen, fuer die alle KI-Pakete Wheels haben


def has_modules(python: str, modules: tuple[str, ...]) -> bool:
    code = "import importlib.util,sys;sys.exit(any(importlib.util.find_spec(m) is None for m in %r))" % (list(modules),)
    return subprocess.run([python, "-c", code], capture_output=True).returncode == 0


def candidates() -> list[Path]:
    return [ROOT / ".venv" / EXE, (ROOT / ".." / "SeriesAI" / ".venv" / EXE).resolve()]      # nie ein globaler Python mit globalen Paketen


def find_python(modules: tuple[str, ...] = CORE) -> str | None:
    for candidate in candidates():
        if candidate.exists() and has_modules(str(candidate), modules): return str(candidate)
    return None


def base_pythons() -> list[list[str]]:
    """Interpreter, aus denen ein venv gebaut werden kann: zuerst der portable Python im Projekt; global installierte nur als letzte Notloesung."""
    found: list[list[str]] = []
    portable = ensure_portable_python()
    if portable: found.append([str(portable)])
    else: found.append([sys.executable])                                                # offline: der startende Python
    return found


def working_base() -> list[str]:
    for cmd in base_pythons():
        try:
            out = subprocess.run([*cmd, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"], capture_output=True, text=True, timeout=30)
            if out.returncode == 0 and out.stdout.strip(): return cmd
        except (OSError, subprocess.SubprocessError): continue
    raise SystemExit("Kein Python verfuegbar: der portable Python konnte nicht geladen werden (Internet?) und der startende Python ist unbrauchbar.")


def setup(core_only: bool = False, assume_yes: bool = False) -> str:
    venv = ROOT / ".venv"; python = venv / EXE
    base = working_base()
    version = subprocess.run([*base, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"], capture_output=True, text=True).stdout.strip()
    supported = tuple(int(x) for x in version.split(".")) in SUPPORTED
    size = "ca. 200 MB" if core_only else "ca. 4 GB (inkl. PyTorch mit CUDA, transformers)"
    print(f"\nEs fehlt eine vollstaendige Python-Umgebung. MusicAI legt jetzt {venv} an (Python {version}) und installiert Pakete: {size}.")
    if not supported and not core_only:
        print(f"WARNUNG: Python {version} wird von torch nicht durchgehend unterstuetzt. Empfohlen: Python 3.10-3.12.")
    if not python.exists(): subprocess.check_call([*base, "-m", "venv", str(venv)])
    run = lambda *args: subprocess.check_call([str(python), "-m", "pip", *args])
    run("install", "--upgrade", "pip")
    run("install", "-r", str(ROOT / "requirements.txt"))
    if not core_only: run("install", "-r", str(ROOT / "requirements-full.txt"))
    print("Umgebung fertig.\n")
    return str(python)


FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"      # enthaelt ffmpeg und ffprobe


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds); h, rest = divmod(seconds, 3600); m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def download(url: str, dest: Path) -> None:
    """Download mit Fortschritt, Geschwindigkeit und ETA (Ausgabe etwa alle 2 s)."""
    with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length", 0)); done = 0; started = last = time.time()
        while True:
            chunk = response.read(1024 * 256)
            if not chunk: break
            out.write(chunk); done += len(chunk)
            if time.time() - last >= 2:
                speed = done / max(time.time() - started, 1e-6); last = time.time()
                eta = fmt_duration((total - done) / speed) if total and speed > 0 else "?"
                print(f"  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB ({done / total * 100 if total else 0:.0f} %)  {speed / 1e6:.1f} MB/s  ETA {eta}", flush=True)


def ensure_ffmpeg() -> None:
    """ffmpeg/ffprobe muessen im Projekt liegen (runtime/ffmpeg/bin). Quelle: Nachbarprojekt ASMRAI (Kopie) oder Download (Windows)."""
    target = ROOT / "runtime" / "ffmpeg"; exe = target / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if exe.exists(): return
    if os.name != "nt":
        if shutil.which("ffmpeg"): return
        raise SystemExit("ffmpeg fehlt. Bitte ffmpeg installieren oder nach runtime/ffmpeg/bin legen.")
    neighbour = (ROOT / ".." / "ASMRAI" / "runtime" / "ffmpeg").resolve()
    if (neighbour / "bin" / "ffmpeg.exe").exists():
        print(f"ffmpeg wird aus {neighbour} nach {target} kopiert ...", flush=True)
        shutil.copytree(neighbour, target, dirs_exist_ok=True); return
    print(f"ffmpeg wird nach {target} geladen ({FFMPEG_URL}) ...", flush=True)
    archive = _LOCAL / "tmp" / "ffmpeg.zip"
    download(FFMPEG_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        top = zf.namelist()[0].split("/")[0]
        zf.extractall(_LOCAL / "tmp")
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(_LOCAL / "tmp" / top), str(target))
    archive.unlink(missing_ok=True)
    print("ffmpeg bereit.\n", flush=True)


def web_address() -> str:
    text = (ROOT / "config.yaml").read_text(encoding="utf-8") if (ROOT / "config.yaml").exists() else ""
    host = re.search(r'^\s*host:\s*"?([\w.\-]+)"?', text, re.M); port = re.search(r"^\s*port:\s*(\d+)", text, re.M)
    return f"http://{host.group(1) if host else '127.0.0.1'}:{port.group(1) if port else '8790'}"


def banner() -> None:
    """Direkte Adressen im Terminal. Der Starter oeffnet nie selbst einen Browser- oder Terminal-Tab."""
    lines = [f"MusicAI WebUI : {web_address()}/", f"Eingang       : {ROOT / 'input'}  (ein Unterordner = ein Creator)", f"Logs          : {ROOT / 'logs'}", "Beenden       : Strg+C"]
    width = max(len(line) for line in lines) + 4
    print("\n" + "=" * width + "\n" + "\n".join(f"  {line}" for line in lines) + "\n" + "=" * width + "\n", flush=True)


def main() -> int:
    args = sys.argv[1:] or ["serve"]
    flags = {a for a in args if a in ("--core", "--yes")}
    args = [a for a in args if a not in flags]
    if args[0] == "setup":
        setup(core_only="--core" in flags, assume_yes="--yes" in flags); return 0
    ensure_ffmpeg()
    python = find_python()
    if python is None: python = setup(core_only="--core" in flags, assume_yes="--yes" in flags)
    elif args[0] != "doctor" and not has_modules(python, FULL):
        print("Hinweis: torch/transformers fehlen in diesem Python - Training und Erzeugung gehen erst nach 'python start.py setup'.", flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONWARNINGS": "ignore"}
    if args[0] == "serve": banner()
    try:
        return subprocess.call([python, "-m", "musicai", *args], env={**env, "MUSICAI_START_CHILD": os.environ.get("MUSICAI_START_CHILD", "")}, cwd=ROOT)
    except KeyboardInterrupt:
        return 0


RESTART_CODE = 75   # Neustart-Button im Dashboard: start.py beendet sich mit diesem Code, der aeussere Teil startet neu


def supervise() -> int:
    """Aeusserer Teil im Terminal: startet start.py als Kindprozess und neu, wenn das Dashboard einen Neustart anfordert."""
    if os.environ.get("MUSICAI_START_CHILD") == "1": return main()
    is_server = not sys.argv[1:] or sys.argv[1] == "serve"
    lock = acquire_server_lock() if is_server else None
    if is_server and lock is None:
        print("MusicAI laeuft bereits. Das bestehende Dashboard ist unter http://127.0.0.1:8790 erreichbar.", flush=True)
        return 2
    env = {**os.environ, "MUSICAI_START_CHILD": "1"}
    try:
        while True:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
            try: code = child.wait()
            except KeyboardInterrupt:      # das Kind bekommt Strg+C ebenfalls und raeumt auf
                try: return child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill(); return 130
            if code != RESTART_CODE: return code
            print("\n[Neustart] MusicAI wird neu gestartet ...\n", flush=True)
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(supervise())

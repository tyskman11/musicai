# MusicAI

MusicAI is a local system that learns the **style** of one or more creators whose material you are authorized to use, then generates **new, original music** in one style or a weighted mix of styles. Everything runs offline on your own computer.

> **Principle:** Train only on material you own or are licensed or contracted to use. Every creator requires a recorded rights confirmation before training is enabled. The project trains a style adapter, not songs: it does not copy or output source recordings. Check the rights attached to your training material before publishing generated results.

## Start

```text
python start.py            # starts the Web UI on http://127.0.0.1:8790
python start.py doctor     # checks the environment
python start.py setup      # creates or updates the environment
python start.py version    # shows the installed version and change history
```

The change history is in [CHANGELOG.md](CHANGELOG.md). Diagnostic events are written to `logs/diagnostics.log` alongside the regular logs.

`start.py` uses only the Python standard library to bootstrap the project. It searches `.venv` first and then the sibling `../SeriesAI/.venv`; it never uses a global Python installation. If necessary, it creates a portable Python 3.11 environment and downloads FFmpeg into `runtime/`. Caches, temporary files, and the base model remain inside the project runtime directory.

## Workflow

1. Create one subfolder per creator in `input/` and add audio or video files. Video files are analysed through their audio track.
2. Run the resumable analysis job to calculate tempo, key, brightness, dynamics, onset density, bass share, and energy curves, then create a creator style profile.
3. Record the rights basis and train the style. MusicAI fine-tunes a MusicGen decoder through a LoRA adapter and saves checkpoints after every epoch.
4. Mix one or more styles with normalized weights, describe the desired result, and generate audio. Longer tracks are created through continuations with short crossfades.

CLI examples:

```text
python start.py analyze <creator>
python start.py rights <creator> "own recordings"
python start.py train <creator>
python start.py generate --style alice=70 --style bob=30 --prompt "soft piano" --seconds 90
```

## How it works

| Stage | Implementation |
|---|---|
| Features | FFmpeg and NumPy STFT analysis for tempo, key, spectral features, dynamics, and energy |
| Profile | Duration-weighted creator averages; mixed styles use weighted averages |
| Training | Configurable MusicGen model, EnCodec excerpts, and rank-16 LoRA adapters on decoder attention and feed-forward layers |
| Mixing | Linear combination of adapter deltas without runtime overhead |
| Generation | MusicGen generation with classifier-free guidance and continuation prompts |

## Limits

- `musicgen-small` produces mono 32 kHz music and is best treated as a sketching model; it does not generate lyric vocals.
- A 6 GB GPU is generally not sufficient for practical `medium` or `large` model training.
- Small training sets mainly teach timbre, instrumentation, and tempo/key tendencies rather than complete song structure.
- Feature extraction is an estimate and may be less reliable for ambient music, dense mixes, or frequent meter changes.
- Style mixing is linear, so very different styles can produce softer or less distinct results.
- Training and generation share local GPU capacity with related projects such as ComfyUI, Whisper, and XTTS.

## Data and security

- Source files are never modified. Removing a creator removes database records only; adapters require an explicit separate action.
- The server binds to `127.0.0.1` and validates `Host` and `Origin` headers.
- Paths in `config.yaml` are project-relative, so the project can be moved as a folder.

## Layout

```text
config.yaml                    paths, model, training, generation, and ports
start.py                       launcher
input/<creator>/               source music (never modified)
models/<creator>/adapter/      trained style adapter
data/                          local database and generated tracks
src/musicai/                   analysis, profile, pipeline, LoRA, generation, application, Web UI, CLI, and jobs
src/musicai/static/            single-page Web UI without a build step
tests/                         pytest suite
```

# MusicAI

Lokales System, das aus der Musik eines oder mehrerer **Creators, an deren Material du die Rechte hast**, einen **Stil** lernt
und daraus **neue, eigene Musik** erzeugt – im Stil eines Creators oder als **Mischung mehrerer Stile**. Alles läuft offline auf dem eigenen PC.

> **Grundsatz:** Trainiert wird nur mit Material, an dem du die Rechte hast (eigene Aufnahmen, Lizenz, Vertrag). Pro Creator muss die
> **Rechte-Bestätigung samt Grundlage** gesetzt sein, sonst ist das Training gesperrt. Gelernt wird ein *Stil* (LoRA-Adapter), keine Titel:
> Es werden keine Aufnahmen kopiert oder ausgegeben. Erzeugte Stücke sind Neuschöpfungen eines KI-Modells – beachte trotzdem die Lizenz
> deines Trainingsmaterials, bevor du Ergebnisse veröffentlichst.

## Start

```text
python start.py            # WebUI auf http://127.0.0.1:8790 (Adresse steht im Terminal; es wird nie selbst ein Tab geöffnet)
python start.py doctor     # Umgebung prüfen
python start.py setup      # Umgebung anlegen/aktualisieren (--core: ohne KI-Pakete)
python start.py version    # installierte Projektversion und Versionsverlauf
```

Der Versionsverlauf liegt in [CHANGELOG.md](CHANGELOG.md). Fuer die Fehlersuche gibt es neben `musicai.log` und `errors.log` das strukturierte Ereignisprotokoll `logs\diagnostics.log`.

`start.py` nutzt nur die Standardbibliothek und sucht einen Interpreter: zuerst `.venv`, dann `..\SeriesAI\.venv` (dort sind torch und
transformers schon installiert) – nie einen globalen Python. Gibt es keinen, wird `.venv` angelegt, und zwar aus einem **portablen Python 3.11**
(python-build-standalone, SHA-256-geprüft), der – wie bei den Schwesterprojekten – gemeinsam unter `<Pool>untime\python` liegt (sonst `runtime\python` im Projekt).
`start.bat` startet `start.py` direkt mit diesem Python. ffmpeg wird aus `..\ASMRAIuntimefmpeg` übernommen oder nach `runtime/ffmpeg` geladen.
Alle Caches, Temp-Dateien und das Basismodell liegen in `runtime\cache` – nichts in globalen Ordnern.

## Ordnerstruktur (Pool)

```text
tyskman11.duckdns.orgi\            Hauptordner = Pool-Server (server\, web\, client\, _template\, runtime\python)
    projects\                        die Generatoren: jeder Ordner ist ein Pool-Projekt
        MusicAI\  ASMRAI\  SeriesAI\  00 - DO NOT LIST\ (ComfyUI)
```

`project.js` + `worker.js` binden MusicAI an den Pool an (verteilte Transkription von Audio-Abschnitten, Eingang `pool\input`). Alle Verweise auf Nachbarprojekte
sind relativ (`..\SeriesAI`, `..\ASMRAI`), das Projekt läuft deshalb nach einem Verschieben des ganzen `projects`-Ordners unverändert weiter.

## Ablauf

1. **Eingang:** Lege in `input\` je Creator einen Unterordner an und kopiere Audio-/Videodateien hinein (mp3, wav, flac, m4a, mp4 …; bei Videos wird die
   Tonspur genutzt). Neue Ordner/Dateien werden automatisch erkannt (Übersicht → „Jetzt prüfen“ erzwingt es).
2. **Analysieren** (Job, fortsetzbar): je Titel Tempo (BPM), Tonart, Helligkeit, Dynamik, Onset-Dichte, Bassanteil, Energieverlauf → **Stil-Profil** des Creators.
3. **Rechte bestätigen** (Grundlage eintragen) und **Stil trainieren**: LoRA-Feintuning des MusicGen-Decoders auf Ausschnitten der Titel.
   Jeder Ausschnitt bekommt eine automatische englische Beschreibung aus seinen Merkmalen („slow, 72 bpm, A minor, dark and warm, gentle, flowing“).
   Nach jeder Epoche wird gespeichert; ein Abbruch behält den letzten Stand. Ergebnis: `models\<creator>\adapter\lora.pt` (wenige MB).
4. **Stil-Mix & Erzeugen:** Regler je Stil → Gewichte werden normiert → die Adapter werden gewichtet in die Basisgewichte eingerechnet
   (`W += Stärke · Σ wᵢ · (α/r) · Bᵢ·Aᵢ`) → MusicGen erzeugt mit deiner Beschreibung (+ optional Tempo/Tonart/Klangfarbe aus dem gemischten Profil).
   Längere Stücke entstehen aus Fortsetzungen (Kontext 8 s, 2 s Überblendung). Ergebnis als WAV und MP3 unter `data\tracks\<id>\`.

Per CLI: `python start.py analyze <creator>`, `rights <creator> "eigene Aufnahmen"`, `train <creator>`,
`generate --style alice=70 --style bob=30 --prompt "soft piano" --seconds 90`.

## Wie es funktioniert

| Stufe | Technik |
|---|---|
| Merkmale | ffmpeg → numpy-STFT: Tempo per Autokorrelation des Spektralflusses, Tonart per Krumhansl-Profile auf Chroma, Schwerpunkt/Flachheit/Bänder/Dynamik |
| Profil | dauergewichteter Mittelwert je Creator; Mischen = gewichteter Mittelwert (Tempo, Chroma, Tonartverteilung …) |
| Training | `facebook/musicgen-large` (3,3 Mrd.; in `config.yaml` einstellbar); EnCodec-Codes der Ausschnitte (32 kHz, 4 Codebooks, Delay-Pattern), Cross-Entropy je Codebook, nur LoRA-Matrizen (Rang 16) auf Attention und Feed-Forward des Decoders, fp16-Autocast, Gradient-Checkpointing |
| Mischen | lineare Kombination der Adapter-Deltas (keine Laufzeitkosten, beliebig viele Stile) |
| Erzeugen | MusicGen `generate` mit Classifier-Free-Guidance; Verlängerung per Audio-Prompt der letzten Sekunden |

## Grenzen (ehrlich)

* **MusicGen-small** erzeugt Mono-Musik in 32 kHz und ist deutlich schwächer als aktuelle Cloud-Modelle; **kein Gesang mit Text**, Klang eher „Skizze“.
  Mit 6 GB VRAM sind `medium`/`large` nicht realistisch (Modell in `config.yaml` änderbar, wenn mehr VRAM da ist).
* Stil-Training mit wenigen Titeln lernt eher **Klangfarbe, Tempo-/Tonartverteilung und Instrumentierung** als Melodien oder Songstruktur.
  Mehr als ~1–2 Stunden Material pro Creator bringt gerade bei kleinem Modell zunehmend weniger.
* Tempo/Tonart/Klassen sind Signal-Schätzungen (kein trainierter Klassifikator); bei Mixen, Ambient oder Metrumwechseln können sie danebenliegen.
* Das **Mischen** zweier LoRAs ist eine lineare Mittelung: das funktioniert gut für ähnliche Stile, bei sehr gegensätzlichen kann das Ergebnis unscharf werden – Stärke und Gewichte ausprobieren.
* Der erste Lauf lädt das Basismodell (ca. 2,3 GB) nach `runtime\cache\huggingface` (`music.allow_download: false` verhindert das).
* Training und Erzeugung teilen sich die GPU mit ComfyUI/Whisper/XTTS der Nachbarprojekte – nicht gleichzeitig laufen lassen.

## Daten & Sicherheit

* Quelldateien werden **nie** verändert; „Entfernen“ löscht nur Datenbank-Einträge (der Adapter nur auf ausdrückliche Rückfrage).
* Der Server bindet an `127.0.0.1`, prüft `Host` (DNS-Rebinding) und `Origin` (CSRF).
* Alle Pfade sind projektrelativ (`config.yaml`); das Projekt lässt sich verschieben, gespeicherte absolute Pfade werden beim Start umgeschrieben.

## Struktur

```text
config.yaml            Pfade, Modell, Training, Erzeugung, Ports
start.py               Starter
input\<creator>\       deine Musik (nie verändert)
models\<creator>\adapter\   trainierter Stil (lora.pt + adapter.json)
data\                  musicai.db, tracks\<id>\ (erzeugte Stücke)
src\musicai\           analysis, profile, pipeline, lora, musgen/train/generate, app, web, cli, watcher, jobs
src\musicai\static\    WebUI (Single-Page, kein Build-Schritt)
tests\                 pytest (python -m pytest)
```

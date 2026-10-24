"""Musik erzeugen: Stil(e) waehlen und mischen -> LoRA-Adapter gewichtet in MusicGen einrechnen -> Segmente erzeugen und verbinden."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

from . import media
from .config import Config
from .db import Database
from .jobs import Cancelled, JobContext
from .musicgen import load_base, pick_device, require_ai
from .profile import get_profile, mix_profiles, normalize_weights
from .train import adapter_dir

glog = logging.getLogger("musicai.generation")


def resolve_styles(db: Database, config: Config, weights: dict[str, float]) -> dict:
    """Stil-Gewichte (slug -> Gewicht) -> normierte Gewichte, geladene Adapter, gemischtes Profil, Warnungen."""
    norm = normalize_weights(weights); adapters, profiles, warnings = [], [], []
    for slug, weight in norm.items():
        creator = db.one("SELECT * FROM creators WHERE slug=?", (slug,))
        if not creator: raise KeyError(f"Stil nicht gefunden: {slug}")
        profile = get_profile(db, creator["id"])
        if profile: profiles.append((profile, weight))
        else: warnings.append(f"{slug}: kein Profil (Analyse fehlt)")
        folder = adapter_dir(config, slug)
        if (folder / "lora.pt").exists():
            meta = json.loads((folder / "adapter.json").read_text(encoding="utf-8")) if (folder / "adapter.json").exists() else {}
            trained_on, current = meta.get("base_model"), str(config.get("music", "base_model"))
            if trained_on and trained_on != current:
                raise ValueError(f"Der Stil-Adapter von '{slug}' wurde fuer {trained_on} trainiert, eingestellt ist {current} - Adapter und Modell muessen zusammenpassen (neu trainieren oder base_model zurueckstellen).")
            adapters.append((slug, weight, folder))
        else: warnings.append(f"{slug}: kein trainierter Adapter - nur das Profil steuert diesen Anteil")
    return {"weights": norm, "adapters": adapters, "profile": mix_profiles(profiles) if profiles else None, "warnings": warnings}


def build_prompt(user_prompt: str, profile: dict | None, use_profile: bool) -> str:
    parts = [p for p in (user_prompt.strip(), profile["caption"] if (profile and use_profile) else "") if p]
    return ", ".join(parts) or "instrumental music"


def crossfade_join(acc: np.ndarray, new: np.ndarray, overlap: int, fade: int) -> np.ndarray:
    """Haengt new an acc an. Die ersten `overlap` Samples von new rekonstruieren das Ende von acc (Fortsetzungs-Kontext) und werden ueberblendet."""
    overlap = min(overlap, len(acc), len(new)); fade = max(1, min(fade, overlap))
    head = acc[: len(acc) - overlap]
    old_tail, new_head = acc[len(acc) - overlap: len(acc) - overlap + fade], new[:fade]
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    return np.concatenate([head, old_tail * (1 - ramp) + new_head * ramp, new[fade:]])


def generate_audio(ctx: JobContext, model: Any, processor: Any, prompt: str, seconds: float, settings: dict, device: str, seed: int | None) -> np.ndarray:
    import torch
    sr = int(model.config.audio_encoder.sampling_rate); frame_rate = 50
    window = float(settings["max_seconds"]); overlap = float(settings["continue_overlap_seconds"]); fade = float(settings["crossfade_seconds"])
    if seed is not None: torch.manual_seed(seed)
    common = dict(do_sample=True, guidance_scale=float(settings["guidance_scale"]), temperature=float(settings["temperature"]), top_k=int(settings["top_k"]))
    audio = np.zeros(0, dtype=np.float32); segment = 0
    while len(audio) < seconds * sr:
        ctx.check()
        remaining = seconds - len(audio) / sr
        if segment == 0:
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
            new_tokens = int(min(window, remaining) * frame_rate)
        else:
            context = audio[-int(overlap * sr):]
            inputs = processor(text=[prompt], audio=context, sampling_rate=sr, padding=True, return_tensors="pt").to(device)
            new_tokens = int(min(window - overlap, remaining + fade) * frame_rate)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            out = model.generate(**inputs, max_new_tokens=max(new_tokens, 64), **common)
        chunk = out[0, 0].float().cpu().numpy()
        audio = chunk if segment == 0 else crossfade_join(audio, chunk, int(overlap * sr), int(fade * sr))
        segment += 1; ctx.progress(min(len(audio) / (seconds * sr), 0.98), f"Segment {segment}: {len(audio) / sr:.0f} / {seconds:.0f} s")
    return audio[: int(seconds * sr)]


def fade_edges(audio: np.ndarray, sr: int, seconds: float = 1.5) -> np.ndarray:
    n = min(int(seconds * sr), len(audio) // 4)
    if n > 1:
        audio = audio.copy(); audio[:n] *= np.linspace(0, 1, n, dtype=np.float32); audio[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
    return audio


def render_piece(ctx: JobContext, db: Database, config: Config, piece_id: int) -> dict:
    piece = db.one("SELECT * FROM pieces WHERE id=?", (piece_id,))
    if not piece: raise KeyError("Stueck nicht gefunden")
    params = json.loads(piece["params"]); require_ai()
    from . import lora
    db.execute("UPDATE pieces SET status='running', updated=? WHERE id=?", (time.time(), piece_id))
    try:
        styles = resolve_styles(db, config, params.get("styles") or {})
        for warning in styles["warnings"]: ctx.log(warning, logging.WARNING)
        prompt = build_prompt(params.get("prompt", ""), styles["profile"], bool(params.get("use_profile", True)))
        ctx.log(f"Prompt: {prompt}")
        device = pick_device(config); model, processor = load_base(config, device)
        merged_weights = {slug: w for slug, w, _ in styles["adapters"]}
        if styles["adapters"]:
            total = sum(merged_weights.values())      # nur Stile mit Adapter teilen sich die Adapter-Staerke
            adapters = [(lora.load_file(folder), w / total) for _, w, folder in styles["adapters"]]
            apply = lora.attach if getattr(model, "is_quantized_base", False) else lora.merge      # quantisierte Gewichte lassen sich nicht veraendern
            changed = apply(model.decoder, adapters, strength=float(params.get("strength", 1.0)))
            ctx.log("Stil-Mix: " + ", ".join(f"{slug} {w / total:.0%}" for slug, w in merged_weights.items()) + f" ({changed} Schichten angepasst, Staerke {params.get('strength', 1.0)})")
        else: ctx.log("Kein trainierter Adapter ausgewaehlt - es wird nur mit dem Basismodell und der Beschreibung erzeugt.", logging.WARNING)
        model.eval(); model.config.use_cache = True
        settings = {k: config.get("generation", k) for k in ("max_seconds", "guidance_scale", "temperature", "top_k", "continue_overlap_seconds", "crossfade_seconds")}
        for key in ("guidance_scale", "temperature"):
            if params.get(key): settings[key] = float(params[key])
        seconds = float(params.get("seconds", 30)); sr = int(model.config.audio_encoder.sampling_rate)
        audio = fade_edges(generate_audio(ctx, model, processor, prompt, seconds, settings, device, params.get("seed")), sr)
        peak = float(np.abs(audio).max()) or 1.0
        audio = audio * min(1.0, 0.95 / peak)
        folder = config.tracks / str(piece_id); wav = folder / "track.wav"
        media.write_wav(wav, audio, sr)
        try: media.to_mp3(config, wav, folder / "track.mp3")
        except Exception as error: ctx.log(f"MP3-Export uebersprungen: {error}", logging.WARNING)      # noqa: BLE001
        db.execute("UPDATE pieces SET status='done', audio_path=?, error=NULL, updated=? WHERE id=?", (str(wav), time.time(), piece_id))
        ctx.log(f"Fertig: {len(audio) / sr:.0f} s -> {wav}")
        return {"audio": str(wav)}
    except Cancelled:
        db.execute("UPDATE pieces SET status='canceled', updated=? WHERE id=?", (time.time(), piece_id)); raise
    except Exception as error:
        db.execute("UPDATE pieces SET status='error', error=?, updated=? WHERE id=?", (str(error)[:500], time.time(), piece_id))
        raise

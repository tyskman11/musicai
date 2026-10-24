"""Stil-Training: LoRA-Feintuning des MusicGen-Decoders auf den Titeln eines Creators (nur mit bestaetigten Rechten)."""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import analysis, media
from .config import Config
from .db import Database
from .jobs import Cancelled, JobContext
from .musicgen import check_load_environment, delay_labels, encode_audio, load_base, pick_device, require_ai
from .profile import caption

tlog = logging.getLogger("musicai.training")


def adapter_dir(config: Config, slug: str) -> Path:
    return config.models / slug / "adapter"


def collect_clips(ctx: JobContext, db: Database, config: Config, creator_id: int) -> list[tuple[np.ndarray, str]]:
    """Ausschnitte (Audio 32 kHz mono + Caption) gleichmaessig ueber alle analysierten Titel verteilt."""
    sr = int(config.get("music", "sample_rate", 32000)); clip = float(config.get("training", "clip_seconds", 20))
    per_track = int(config.get("training", "clips_per_track", 6)); max_clips = int(config.get("training", "max_clips", 400))
    rows = db.query("SELECT id,title,path,duration FROM tracks WHERE creator_id=? AND features IS NOT NULL AND (duration IS NULL OR duration>=?) ORDER BY id", (creator_id, clip + 2))
    if not rows: raise ValueError(f"Keine analysierten Titel mit mindestens {clip + 2:.0f} s - zuerst die Analyse starten.")
    rng = random.Random(7); clips: list[tuple[np.ndarray, str]] = []
    budget = max(1, min(per_track, max_clips // len(rows) or 1))
    for i, row in enumerate(rows, 1):
        ctx.check(); ctx.progress(0.1 * i / len(rows), f"Ausschnitte {i}/{len(rows)}: {row['title']}" if i % 5 == 1 else None)
        duration = float(row["duration"] or 0)
        if duration < clip + 2: continue
        starts = np.linspace(min(5.0, (duration - clip) / 2), max(duration - clip - 5.0, 0.0), budget) if duration > clip + 12 else [0.0]
        for start in starts:
            try:
                audio = media.decode(config, row["path"], sr, start=float(start) + rng.uniform(-1.5, 1.5) * (duration > clip + 12), seconds=clip)
                if len(audio) < sr * clip * 0.95 or float(np.abs(audio).max()) < 1e-3: continue
                clips.append((audio[: int(sr * clip)], caption(analysis.analyze(audio, sr))))
            except Exception as error:  # noqa: BLE001 - ein defekter Ausschnitt stoppt das Training nicht
                ctx.log(f"  Ausschnitt uebersprungen ({row['title']}): {error}", logging.WARNING)
    rng.shuffle(clips)
    return clips[:max_clips]


def cache_key(db: Database, config: Config, creator_id: int) -> str:
    """Aendert sich, sobald Titel, Analyse oder Ausschnitt-Einstellungen anders sind - dann wird neu gesammelt."""
    import hashlib
    tracks = db.query("SELECT id,size,mtime FROM tracks WHERE creator_id=? AND features IS NOT NULL ORDER BY id", (creator_id,))
    parts = [[config.get("training", k) for k in ("clip_seconds", "clips_per_track", "max_clips")], [(t["id"], t["size"], t["mtime"]) for t in tracks]]
    return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()


def save_cache(path: Path, key: str, codes: list, captions: list[str]) -> None:
    import torch
    torch.save({"key": key, "codes": torch.stack(codes).to(torch.int16), "captions": captions}, path)


def load_cache(path: Path, key: str) -> tuple[list, list[str]] | None:
    if not path.exists(): return None
    import torch
    try: data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - beschaedigter Zwischenspeicher wird einfach neu erzeugt
        return None
    return ([c.long() for c in data["codes"]], list(data["captions"])) if data.get("key") == key else None


def train_loop(ctx: JobContext, model: Any, processor: Any, codes: list[Any], captions: list[str], settings: dict, device: str,
               checkpoint: Any = None) -> dict:
    """Trainiert die LoRA-Parameter des (bereits injizierten) Modells. codes: Liste von Tensoren (K, T)."""
    import torch
    from . import lora
    epochs, batch, accum = int(settings["epochs"]), int(settings["batch_size"]), int(settings["grad_accum"])
    params = lora.trainable_parameters(model.decoder)
    optimizer = torch.optim.AdamW(params, lr=float(settings["learning_rate"]), weight_decay=0.01)
    steps_total = max(1, (epochs * len(codes)) // (batch * accum)); step = 0
    use_amp = bool(settings.get("fp16", True)) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pad = model.config.decoder.pad_token_id
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(2, steps_total // 10)) * max(0.05, 1 - s / steps_total))
    model.train(); model.text_encoder.eval(); model.audio_encoder.eval()
    started, losses, micro = time.monotonic(), [], 0
    for epoch in range(epochs):
        order = list(range(len(codes))); random.Random(epoch).shuffle(order)
        for i in range(0, len(order) - batch + 1, batch):
            ctx.check()
            idx = order[i:i + batch]
            labels = delay_labels(torch.stack([codes[j] for j in idx]).to(device), pad)
            text = processor.tokenizer([captions[j] for j in idx], padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=text.input_ids, attention_mask=text.attention_mask, labels=labels).loss
            scaler.scale(loss / accum).backward(); losses.append(float(loss.detach())); micro += 1
            if micro % accum == 0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                recent = float(np.mean(losses[-accum * 4:]))
                eta = (time.monotonic() - started) / step * (steps_total - step)
                eta_minutes = max(0, round(eta / 60))
                eta_text = f"{eta_minutes} min" if eta_minutes < 60 else f"{eta_minutes // 60} h {eta_minutes % 60} min"
                ctx.progress(0.15 + 0.8 * min(step / steps_total, 1.0), f"Epoche {epoch + 1}/{epochs}  Schritt {step}/{steps_total}  Loss {recent:.3f}  ETA {eta_text}")
        if checkpoint: checkpoint(step, float(np.mean(losses[-max(1, len(codes)):])))
    return {"steps": step, "loss": float(np.mean(losses[-max(1, len(codes)):])) if losses else 0.0, "seconds": time.monotonic() - started}


def train_creator(ctx: JobContext, db: Database, config: Config, creator_id: int) -> dict:
    creator = db.one("SELECT * FROM creators WHERE id=?", (creator_id,))
    if not creator: raise KeyError("Creator nicht gefunden")
    if not creator["rights"]: raise ValueError(f"Fuer '{creator['name']}' ist die Rechte-Bestaetigung nicht gesetzt - Training gesperrt.")
    require_ai()
    import torch
    from . import lora
    settings = {k: config.get("training", k) for k in ("epochs", "batch_size", "grad_accum", "learning_rate", "lora_rank", "lora_alpha", "lora_dropout", "fp16", "gradient_checkpointing")}
    cache_file = config.work / f"train_{creator['slug']}.pt"; key = cache_key(db, config, creator_id)
    cached = load_cache(cache_file, key)
    device = pick_device(config)
    # Muss vor collect_clips passieren: sonst kostet ein fehlendes GPU-Paket
    # bei grossen Bibliotheken viele Minuten, bevor der Job ohnehin scheitert.
    check_load_environment(config, device)
    if cached:
        codes, captions = cached
        ctx.log(f"Zwischenspeicher gefunden: {len(codes)} fertig kodierte Ausschnitte (Sammeln + Kodieren entfaellt); Basismodell laden ...")
        model, processor = load_base(config, device)
    else:
        ctx.log(f"Stil-Training '{creator['name']}': Ausschnitte sammeln ...")
        clips = collect_clips(ctx, db, config, creator_id)
        if len(clips) < 8: raise ValueError(f"Nur {len(clips)} Ausschnitte - fuer ein Stil-Training werden mindestens 8 (besser einige Dutzend) gebraucht.")
        ctx.log(f"{len(clips)} Ausschnitte a {config.get('training', 'clip_seconds', 20)} s; Basismodell laden ({config.get('music', 'base_model')}) ...")
        model, processor = load_base(config, device)
        codes = []
        for i in range(0, len(clips), 4):
            ctx.check()
            wave = torch.from_numpy(np.stack([c[0] for c in clips[i:i + 4]]))[:, None, :]
            codes.extend(encode_audio(model, wave, device).cpu())
            ctx.progress(0.1 + 0.05 * i / len(clips))
        captions = [c[1] for c in clips]; del clips
        save_cache(cache_file, key, codes, captions); ctx.log("Kodierte Ausschnitte zwischengespeichert (ein erneutes Training startet dann sofort).")
    if device == "cpu": ctx.log("Keine GPU gefunden - Training auf der CPU ist sehr langsam.", logging.WARNING)
    ctx.log("Beispiel-Beschreibungen: " + " | ".join(captions[:3]))
    lora.freeze(model)      # auch Text-/Audio-Encoder: sonst entstehen unnoetige Gradienten (VRAM)
    names = lora.inject(model.decoder, int(settings["lora_rank"]), float(settings["lora_alpha"]), float(settings["lora_dropout"]))
    ctx.log(f"LoRA: {len(names)} Schichten, {sum(p.numel() for p in lora.trainable_parameters(model.decoder)) / 1e6:.1f} M trainierbare Parameter")
    if settings["gradient_checkpointing"]:
        try:
            model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.decoder.config.use_cache = False
        except Exception as error:  # noqa: BLE001 - optional
            ctx.log(f"Gradient-Checkpointing nicht verfuegbar: {error}", logging.WARNING)
    folder = adapter_dir(config, creator["slug"])
    meta = lambda extra=None: {"creator": creator["slug"], "base_model": config.get("music", "base_model"), "clips": len(codes), "epochs": settings["epochs"], "rank": settings["lora_rank"], **(extra or {})}
    save_checkpoint = lambda step, loss: lora.save(model.decoder, folder, meta({"steps": step, "loss": loss, "partial": True}))
    try:
        result = train_loop(ctx, model, processor, codes, captions, settings, device, checkpoint=save_checkpoint)
    except Cancelled:
        if folder.joinpath("lora.pt").exists(): ctx.log("Abgebrochen - der letzte Epochen-Stand bleibt gespeichert.")
        raise
    lora.save(model.decoder, folder, meta({"steps": result["steps"], "loss": result["loss"], "partial": False, "trained": time.time()}))
    db.execute("INSERT INTO adapters(creator_id,path,base_model,clips,epochs,steps,loss,seconds,trained) VALUES(?,?,?,?,?,?,?,?,?) "
               "ON CONFLICT(creator_id) DO UPDATE SET path=excluded.path, base_model=excluded.base_model, clips=excluded.clips, epochs=excluded.epochs, "
               "steps=excluded.steps, loss=excluded.loss, seconds=excluded.seconds, trained=excluded.trained",
               (creator_id, str(folder), str(config.get("music", "base_model")), len(codes), settings["epochs"], result["steps"], result["loss"], result["seconds"], time.time()))
    ctx.log(f"Fertig: {result['steps']} Schritte, Loss {result['loss']:.3f}, {result['seconds'] / 60:.1f} min -> {folder}")
    del model
    if device.startswith("cuda"): torch.cuda.empty_cache()
    return result

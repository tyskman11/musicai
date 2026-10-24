"""Stil-Profil je Creator (aus den analysierten Titeln), Beschreibungstexte (Captions) und das Mischen mehrerer Stile."""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from .analysis import NOTES
from .db import Database, dumps


def tempo_word(bpm: float) -> str:
    if bpm <= 0: return ""
    return "very slow" if bpm < 70 else "slow" if bpm < 95 else "mid-tempo" if bpm < 120 else "upbeat" if bpm < 145 else "fast"


def brightness_word(centroid_hz: float) -> str:
    return "dark and warm" if centroid_hz < 1400 else "balanced" if centroid_hz < 2400 else "bright" if centroid_hz < 3600 else "very bright and airy"


def energy_word(dynamic_range_db: float, onset_rate: float) -> str:
    if onset_rate < 1.2: return "sparse, calm"
    if onset_rate < 3.0: return "gentle, flowing"
    return "dense, energetic" if dynamic_range_db < 25 else "dynamic, energetic"


def caption(feat: dict) -> str:
    """Englische Kurzbeschreibung eines Titels/Ausschnitts - das ist der Text, mit dem MusicGen trainiert und gesteuert wird."""
    parts = []
    if feat.get("bpm"): parts.append(f"{tempo_word(feat['bpm'])}, {round(feat['bpm'])} bpm")
    if feat.get("key_confidence", 0) >= 0.25: parts.append(f"{feat['tonic']} {feat['mode']}")
    parts.append(brightness_word(feat.get("centroid_hz", 2000)))
    parts.append(energy_word(feat.get("dynamic_range_db", 20), feat.get("onset_rate", 2)))
    bands = feat.get("bands") or {}
    if bands.get("low", 0) > 0.55: parts.append("bass-heavy")
    return ", ".join(parts)


def _wavg(items: list[tuple[float, float]]) -> float:
    total = sum(w for _, w in items)
    return sum(v * w for v, w in items) / total if total > 0 else 0.0


def build_profile(db: Database, creator_id: int) -> dict:
    rows = [r for r in db.query("SELECT id,title,duration,features FROM tracks WHERE creator_id=? AND features IS NOT NULL", (creator_id,))]
    feats = [(r, json.loads(r["features"])) for r in rows]
    if not feats: raise ValueError("Noch keine analysierten Titel - zuerst die Analyse starten.")
    weights = [max(f["duration"], 1.0) for _, f in feats]
    pairs = lambda key: [(f[key], w) for (_, f), w in zip(feats, weights)]
    tempos = [(f["bpm"], w * max(f["tempo_strength"], 0.05)) for (_, f), w in zip(feats, weights) if f["bpm"] > 0]
    keys: dict[str, float] = {}
    for (_, f), w in zip(feats, weights):
        if f["key_confidence"] >= 0.2: keys[f"{f['tonic']} {f['mode']}"] = keys.get(f"{f['tonic']} {f['mode']}", 0.0) + w * f["key_confidence"]
    key_total = sum(keys.values()) or 1.0
    chroma = np.sum([np.array(f["chroma"]) * w for (_, f), w in zip(feats, weights)], axis=0); chroma = chroma / chroma.sum()
    curve = np.array([np.interp(np.linspace(0, 1, 12), np.linspace(0, 1, len(f["energy_curve"])), f["energy_curve"]) - np.mean(f["energy_curve"])
                      for _, f in feats if len(f["energy_curve"]) >= 2])
    bpm_values = [f["bpm"] for _, f in feats if f["bpm"] > 0]
    profile: dict[str, Any] = {
        "tracks": len(feats), "hours": round(sum(f["duration"] for _, f in feats) / 3600, 2),
        "bpm": round(_wavg(tempos), 1), "bpm_min": round(float(np.percentile(bpm_values, 10)), 1) if bpm_values else 0, "bpm_max": round(float(np.percentile(bpm_values, 90)), 1) if bpm_values else 0,
        "keys": {k: round(v / key_total, 3) for k, v in sorted(keys.items(), key=lambda kv: -kv[1])[:8]},
        "chroma": [round(float(x), 4) for x in chroma],
        "loudness_db": round(_wavg(pairs("loudness_db")), 2), "dynamic_range_db": round(_wavg(pairs("dynamic_range_db")), 2),
        "centroid_hz": round(_wavg(pairs("centroid_hz")), 1), "onset_rate": round(_wavg(pairs("onset_rate")), 2), "flatness": round(_wavg(pairs("flatness")), 4),
        "bands": {b: round(_wavg([(f["bands"][b], w) for (_, f), w in zip(feats, weights)]), 4) for b in ("low", "mid", "high")},
        "energy_arc": [round(float(x), 2) for x in (curve.mean(axis=0) if len(curve) else [])],
    }
    profile["top_key"] = next(iter(profile["keys"]), "")
    profile["caption"] = describe(profile)
    db.execute("INSERT INTO profiles(creator_id,data,updated) VALUES(?,?,?) ON CONFLICT(creator_id) DO UPDATE SET data=excluded.data, updated=excluded.updated",
               (creator_id, dumps(profile), time.time()))
    return profile


def get_profile(db: Database, creator_id: int) -> dict | None:
    row = db.one("SELECT data FROM profiles WHERE creator_id=?", (creator_id,))
    return json.loads(row["data"]) if row else None


def describe(profile: dict) -> str:
    """Beschreibung eines (auch gemischten) Profils im Caption-Format."""
    top = profile.get("top_key", "")
    feat = {"bpm": profile.get("bpm", 0), "key_confidence": 1.0 if top else 0.0, "tonic": top.split(" ")[0] if top else "", "mode": top.split(" ")[1] if top else "",
            "centroid_hz": profile.get("centroid_hz", 2000), "dynamic_range_db": profile.get("dynamic_range_db", 20), "onset_rate": profile.get("onset_rate", 2),
            "bands": profile.get("bands", {})}
    return caption(feat)


def mix_profiles(profiles: list[tuple[dict, float]]) -> dict:
    """Gewichtetes Mischen mehrerer Stil-Profile (Gewichte werden normiert). Das Ergebnis hat dieselbe Form wie ein Profil."""
    profiles = [(p, w) for p, w in profiles if w > 0 and p]
    if not profiles: raise ValueError("Keine Stile mit Gewicht > 0 gewaehlt.")
    total = sum(w for _, w in profiles); norm = [(p, w / total) for p, w in profiles]
    avg = lambda key: round(sum(p.get(key, 0) * w for p, w in norm), 3)
    keys: dict[str, float] = {}
    for p, w in norm:
        for k, v in (p.get("keys") or {}).items(): keys[k] = keys.get(k, 0.0) + v * w
    chroma = np.sum([np.array(p["chroma"]) * w for p, w in norm], axis=0)
    mixed = {"tracks": sum(p.get("tracks", 0) for p, _ in norm), "bpm": avg("bpm"), "bpm_min": avg("bpm_min"), "bpm_max": avg("bpm_max"),
             "keys": {k: round(v, 3) for k, v in sorted(keys.items(), key=lambda kv: -kv[1])[:8]}, "chroma": [round(float(x), 4) for x in chroma / chroma.sum()],
             "loudness_db": avg("loudness_db"), "dynamic_range_db": avg("dynamic_range_db"), "centroid_hz": avg("centroid_hz"), "onset_rate": avg("onset_rate"),
             "bands": {b: round(sum(p.get("bands", {}).get(b, 0) * w for p, w in norm), 4) for b in ("low", "mid", "high")}}
    mixed["top_key"] = next(iter(mixed["keys"]), ""); mixed["caption"] = describe(mixed)
    return mixed


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {k: float(v) for k, v in weights.items() if float(v) > 0}
    total = sum(clean.values())
    if total <= 0: raise ValueError("Mindestens ein Stil braucht ein Gewicht > 0.")
    return {k: v / total for k, v in clean.items()}


__all__ = ["NOTES", "build_profile", "get_profile", "mix_profiles", "normalize_weights", "caption", "describe"]

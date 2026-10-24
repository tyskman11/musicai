import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from musicai import analysis, pipeline
from musicai.app import App
from musicai.config import Config
from musicai.db import Database, dumps
from musicai.generate import build_prompt, crossfade_join, resolve_styles
from musicai.profile import build_profile, caption, mix_profiles, normalize_weights

SR = 22050


def make_config(tmp_path: Path) -> Config:
    return Config({"paths": {"data": str(tmp_path / "data"), "input": str(tmp_path / "input"), "models": str(tmp_path / "models")},
                   "music": {"cache": str(tmp_path / "cache")}, "inbox": {"settle_seconds": 0}})


def click_track(bpm: float, seconds: float = 24.0, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    audio = 0.15 * np.sin(2 * np.pi * freq * t)
    beat = 60.0 / bpm
    for k in range(int(seconds / beat)):
        i = int(k * beat * SR); n = int(0.04 * SR)
        audio[i:i + n] += 0.8 * np.random.default_rng(k).standard_normal(len(audio[i:i + n])) * np.hanning(len(audio[i:i + n]))[: len(audio[i:i + n])]
    return audio.astype(np.float32)


def test_key_from_krumhansl_profiles():
    for tonic in (0, 7, 9):
        assert analysis.estimate_key(np.roll(analysis.MAJOR, tonic))[:2] == (analysis.NOTES[tonic], "major")
        assert analysis.estimate_key(np.roll(analysis.MINOR, tonic))[:2] == (analysis.NOTES[tonic], "minor")


@pytest.mark.parametrize("bpm", [90, 120, 140])
def test_tempo_estimate(bpm):
    feat = analysis.analyze(click_track(bpm), SR)
    assert abs(feat["bpm"] - bpm) <= 4, feat["bpm"]
    assert feat["duration"] == pytest.approx(24.0, abs=0.1) and len(feat["chroma"]) == 12


def test_too_short_rejected():
    with pytest.raises(ValueError): analysis.analyze(np.zeros(SR * 2, dtype=np.float32), SR)


def insert_track(db, creator_id, title, bpm, tonic="A", mode="minor", centroid=1800.0, duration=200.0):
    chroma = np.roll(analysis.MINOR if mode == "minor" else analysis.MAJOR, analysis.NOTES.index(tonic)); chroma = (chroma / chroma.sum()).tolist()
    feat = {"duration": duration, "bpm": bpm, "tempo_strength": 0.6, "onset_rate": 2.0, "tonic": tonic, "mode": mode, "key_confidence": 0.8, "chroma": chroma,
            "loudness_db": -20.0, "dynamic_range_db": 18.0, "centroid_hz": centroid, "flatness": 0.1, "bands": {"low": 0.2, "mid": 0.7, "high": 0.1}, "energy_curve": [-20, -18, -22, -19]}
    db.execute("INSERT INTO tracks(creator_id,path,title,duration,features) VALUES(?,?,?,?,?)", (creator_id, f"/x/{title}.mp3", title, duration, dumps(feat)))


def two_creators(db):
    ids = {}
    for slug, bpm, tonic, mode, centroid in (("slow-dark", 70.0, "A", "minor", 1200.0), ("fast-bright", 140.0, "C", "major", 3200.0)):
        cid = db.execute("INSERT INTO creators(slug,name,source_dir,created) VALUES(?,?,?,0)", (slug, slug, "/x")); ids[slug] = cid
        for i in range(3): insert_track(db, cid, f"{slug}{i}", bpm + i, tonic, mode, centroid)
        build_profile(db, cid)
    return ids


def test_profile_and_caption(tmp_path):
    db = Database(tmp_path / "t.db"); ids = two_creators(db)
    p = json.loads(db.one("SELECT data FROM profiles WHERE creator_id=?", (ids["slow-dark"],))["data"])
    assert 69 <= p["bpm"] <= 73 and p["top_key"] == "A minor" and "slow" in p["caption"] and "dark" in p["caption"] and "A minor" in p["caption"]
    assert "upbeat" in caption({"bpm": 128, "key_confidence": 0, "centroid_hz": 2000, "dynamic_range_db": 20, "onset_rate": 2})


def test_mix_profiles_interpolates(tmp_path):
    db = Database(tmp_path / "t.db"); ids = two_creators(db)
    a = json.loads(db.one("SELECT data FROM profiles WHERE creator_id=?", (ids["slow-dark"],))["data"]); b = json.loads(db.one("SELECT data FROM profiles WHERE creator_id=?", (ids["fast-bright"],))["data"])
    half = mix_profiles([(a, 1), (b, 1)]); assert (a["bpm"] + b["bpm"]) / 2 == pytest.approx(half["bpm"], abs=0.2)
    mostly_a = mix_profiles([(a, 3), (b, 1)]); assert a["bpm"] < mostly_a["bpm"] < half["bpm"]
    assert mix_profiles([(a, 1), (b, 0)])["bpm"] == pytest.approx(a["bpm"], abs=0.01) and sum(half["chroma"]) == pytest.approx(1.0, abs=0.01)
    with pytest.raises(ValueError): mix_profiles([(a, 0)])
    assert normalize_weights({"x": 3, "y": 1, "z": 0}) == {"x": 0.75, "y": 0.25}
    with pytest.raises(ValueError): normalize_weights({"x": 0})


def test_inbox_sync_and_scan(tmp_path):
    config = make_config(tmp_path); db = Database(config.db_path)
    (config.input / "Alice").mkdir(); (config.input / "Alice" / "song one [dQw4w9WgXcQ].mp3").write_bytes(b"x"); (config.input / "Alice" / "notes.txt").write_text("no")
    (config.input / "_ignored").mkdir(); (config.input / "Bob").mkdir()
    created = pipeline.sync_inbox(db, config); assert sorted(c["slug"] for c in created) == ["alice", "bob"]
    assert pipeline.sync_inbox(db, config) == []
    alice = db.one("SELECT * FROM creators WHERE slug='alice'")
    assert pipeline.scan_creator(db, alice) == (1, 1) and pipeline.scan_creator(db, alice) == (0, 1)
    assert db.one("SELECT title FROM tracks")["title"] == "song one"


def test_creator_folder_is_limited_to_nas_root(tmp_path):
    config = make_config(tmp_path)
    config.raw["paths"]["nas_root"] = str(tmp_path)
    (tmp_path / "library" / "Artist").mkdir(parents=True)
    app = App(config)
    browser = app.nas_folders("library")
    assert browser["folders"] == [{"name": "Artist", "path": str(Path("library") / "Artist")}]
    creator = app.add_creator("Artist", str(tmp_path / "library" / "Artist"))
    assert creator["source_dir"] == str(tmp_path / "library" / "Artist")
    with pytest.raises(ValueError, match="NAS"):
        app.add_creator("Outside", str(tmp_path.parent))
    with pytest.raises(ValueError, match="Ungueltiger"):
        app.nas_folders("../")


def test_rights_gate_blocks_training(tmp_path):
    app = App(make_config(tmp_path)); cid = app.db.execute("INSERT INTO creators(slug,name,source_dir,created) VALUES('c','C','/x',0)")
    insert_track(app.db, cid, "t", 100.0)
    with pytest.raises(ValueError, match="Rechte"): app.start_train(cid)
    with pytest.raises(ValueError, match="Grundlage|Rechte"): app.set_rights(cid, True, "  ")
    app.set_rights(cid, True, "eigene Aufnahmen")
    assert app.db.one("SELECT rights FROM creators WHERE id=?", (cid,))["rights"] == 1
    job = app.start_train(cid); assert isinstance(job, int)       # Job wird eingereiht (schlaegt ohne Adapterdaten spaeter selbst fehl)
    app.set_rights(cid, False)
    with pytest.raises(ValueError): app.start_train(cid)


def test_create_piece_and_resolve_styles(tmp_path):
    app = App(make_config(tmp_path)); ids = two_creators(app.db)
    styles = resolve_styles(app.db, app.config, {"slow-dark": 3, "fast-bright": 1})
    assert styles["weights"] == {"slow-dark": 0.75, "fast-bright": 0.25} and styles["adapters"] == [] and len(styles["warnings"]) == 2
    assert 70 < styles["profile"]["bpm"] < 105
    with pytest.raises(KeyError): resolve_styles(app.db, app.config, {"nope": 1})
    piece = app.create_piece({"styles": {"slow-dark": 50, "fast-bright": 50}, "prompt": "piano", "seconds": 4})
    row = app.db.one("SELECT * FROM pieces WHERE id=?", (piece["id"],)); params = json.loads(row["params"])
    assert params["seconds"] == 5.0 and params["styles"] == {"slow-dark": 0.5, "fast-bright": 0.5}      # Mindestlaenge 5 s, normierte Gewichte
    assert build_prompt("piano", styles["profile"], True).startswith("piano, ") and build_prompt("", None, True) == "instrumental music"


def test_crossfade_join_length_and_continuity():
    acc = np.ones(1000, dtype=np.float32); new = np.ones(800, dtype=np.float32)
    out = crossfade_join(acc, new, overlap=300, fade=100)
    assert len(out) == len(acc) - 300 + len(new) and np.allclose(out, 1.0)


def test_app_analyze_real_audio(tmp_path):
    config = make_config(tmp_path)
    if not (shutil.which(config.ffmpeg) or Path(config.ffmpeg).exists()): pytest.skip("ffmpeg nicht verfuegbar")
    import soundfile as sf
    folder = config.input / "Tester"; folder.mkdir(parents=True)
    for i, bpm in enumerate((100, 110)): sf.write(str(folder / f"track{i}.wav"), click_track(bpm, 30), SR)
    app = App(config); cid = app.db.one("SELECT id FROM creators WHERE slug='tester'")["id"]
    from musicai.jobs import JobContext
    ctx = JobContext(app.jobs, app.db.execute("INSERT INTO jobs(kind,title,params,status,created) VALUES('analyze','t','{}','running',0)"), "analyze", "t")
    result = pipeline.analyze_creator(ctx, app.db, config, cid)
    assert result["analyzed"] == 2 and result["profile"]
    assert 95 <= json.loads(app.db.one("SELECT data FROM profiles WHERE creator_id=?", (cid,))["data"])["bpm"] <= 115


def test_adapter_for_other_base_model_is_rejected(tmp_path):
    config = make_config(tmp_path); config.raw["music"]["base_model"] = "facebook/musicgen-large"
    db = Database(config.db_path); ids = two_creators(db)
    folder = config.models / "slow-dark" / "adapter"; folder.mkdir(parents=True)
    (folder / "lora.pt").write_bytes(b"x"); (folder / "adapter.json").write_text(json.dumps({"base_model": "facebook/musicgen-small"}), encoding="utf-8")
    with pytest.raises(ValueError, match="musicgen-small.*musicgen-large"): resolve_styles(db, config, {"slow-dark": 1})
    (folder / "adapter.json").write_text(json.dumps({"base_model": "facebook/musicgen-large"}), encoding="utf-8")
    assert resolve_styles(db, config, {"slow-dark": 1})["adapters"][0][0] == "slow-dark"


def test_failures_retry_only_failed(tmp_path):
    app = App(make_config(tmp_path)); cid = app.db.execute("INSERT INTO creators(slug,name,source_dir,created) VALUES('c','C','/x',0)")
    insert_track(app.db, cid, "good", 100.0)
    app.db.execute("INSERT INTO tracks(creator_id,path,title,error) VALUES(?,?,?,?)", (cid, "/x/bad.mp3", "bad", "kaputt"))
    app.db.execute("INSERT INTO pieces(name,params,status,error,created,updated) VALUES('p','{}','error','boom',0,0)")
    app.db.execute("INSERT INTO pieces(name,params,status,created,updated) VALUES('ok','{}','done',0,0)")
    fail = app.failures()
    assert [t["title"] for t in fail["tracks"]] == ["bad"] and [p["name"] for p in fail["pieces"]] == ["p"] and fail["total"] == 2
    res = app.retry_failures(["tracks"])
    assert res["count"] == 1 and app.db.one("SELECT params FROM jobs WHERE id=?", (res["started"]["tracks"][0],))["params"].count('"only_failed":true') == 1
    app.retry_failures(["pieces"])
    assert app.db.one("SELECT status FROM pieces WHERE name='p'")["status"] == "queued" and app.db.one("SELECT status FROM pieces WHERE name='ok'")["status"] == "done"

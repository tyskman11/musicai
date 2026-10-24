"""Musikalische Merkmale aus Audio (nur numpy): Tempo, Tonart, Klangfarbe, Dynamik, Energieverlauf.

Das sind Schaetzungen aus dem Signal (kein trainiertes Modell) - sie dienen als Stil-Profil, als Trainings-Beschreibung
(Caption je Ausschnitt) und zum Steuern der Erzeugung."""
from __future__ import annotations

import numpy as np

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Kessler-Tonartprofile
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
N_FFT, HOP = 2048, 512


def stft_mag(audio: np.ndarray) -> np.ndarray:
    """Betragsspektrogramm (Frames x Bins), Hann-Fenster."""
    if len(audio) < N_FFT: audio = np.pad(audio, (0, N_FFT - len(audio)))
    window = np.hanning(N_FFT).astype(np.float32)
    count = 1 + (len(audio) - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(count)[:, None]
    return np.abs(np.fft.rfft(audio[index] * window, axis=1)).astype(np.float32)


def chroma_from_mag(mag: np.ndarray, sr: int) -> np.ndarray:
    """12-dimensionaler Tonklassen-Vektor je Frame (65 Hz - 2,1 kHz)."""
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    use = (freqs >= 65) & (freqs <= 2100)
    pitch = np.round(12 * np.log2(freqs[use] / 440.0) + 69).astype(int) % 12
    chroma = np.zeros((mag.shape[0], 12), dtype=np.float32)
    for pc in range(12): chroma[:, pc] = mag[:, use][:, pitch == pc].sum(axis=1)
    return chroma


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    """(Grundton, 'major'|'minor', Konfidenz 0..1) per Korrelation mit den Krumhansl-Profilen."""
    best = (-2.0, 0, "major"); scores = []
    for mode, profile in (("major", MAJOR), ("minor", MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(chroma_mean, np.roll(profile, tonic))[0, 1]) if chroma_mean.std() > 0 else 0.0
            scores.append(r)
            if r > best[0]: best = (r, tonic, mode)
    scores.sort(reverse=True)
    margin = scores[0] - scores[1] if len(scores) > 1 else 0.0
    return NOTES[best[1]], best[2], float(np.clip(best[0] * 0.6 + margin * 2.0, 0.0, 1.0))


def estimate_tempo(mag: np.ndarray, sr: int) -> tuple[float, float, float]:
    """(BPM, Staerke 0..1, Onsets je Sekunde) per Autokorrelation der Spektralfluss-Huellkurve (Bereich 60-190 BPM)."""
    frame_rate = sr / HOP
    logmag = np.log1p(mag * 10.0)
    flux = np.maximum(0.0, np.diff(logmag, axis=0)).sum(axis=1)
    if len(flux) < int(frame_rate * 4) or flux.std() < 1e-9: return 0.0, 0.0, 0.0
    flux = flux - np.convolve(flux, np.ones(int(frame_rate)) / int(frame_rate), mode="same")      # lokalen Mittelwert abziehen
    flux = np.maximum(flux, 0.0)
    n = len(flux); size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(flux, size); ac = np.fft.irfft(spec * np.conj(spec))[:n]
    ac /= ac[0] + 1e-12
    lags = np.arange(n); bpm = 60.0 * frame_rate / np.maximum(lags, 1)
    valid = (bpm >= 60) & (bpm <= 190)
    if not valid.any(): return 0.0, 0.0, 0.0
    prior = np.exp(-0.5 * (np.log2(bpm / 110.0) / 0.75) ** 2)                                  # sanfte Vorliebe fuer 80-150 BPM (Oktavfehler)
    score = np.where(valid, ac * prior, -1.0)
    lag = int(np.argmax(score))
    lo, hi = max(lag - 1, 1), min(lag + 2, n)                                                  # Parabel-Verfeinerung
    if 0 < lag < n - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]; denom = a - 2 * b + c
        lag_f = lag + (0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0)
    else: lag_f = float(lag)
    peaks = float((flux > flux.mean() + flux.std()).sum()) / (n / frame_rate)
    return float(60.0 * frame_rate / lag_f), float(np.clip(ac[lag], 0.0, 1.0)), peaks


def analyze(audio: np.ndarray, sr: int, block_seconds: float = 10.0) -> dict:
    """Merkmale eines Titels. audio: mono float32."""
    if len(audio) < sr * 5: raise ValueError("Titel kuerzer als 5 Sekunden")
    mag = stft_mag(audio)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    power = mag ** 2
    frame_db = 10 * np.log10(power.sum(axis=1) / N_FFT + 1e-12)
    active = frame_db > (np.percentile(frame_db, 95) - 50)                                    # Stille/Rauschboden ausnehmen
    centroid = (mag * freqs).sum(axis=1) / (mag.sum(axis=1) + 1e-9)
    total = power.sum() + 1e-12
    bands = {"low": float(power[:, freqs < 200].sum() / total), "mid": float(power[:, (freqs >= 200) & (freqs < 4000)].sum() / total),
             "high": float(power[:, freqs >= 4000].sum() / total)}
    chroma = chroma_from_mag(mag, sr)
    chroma_mean = chroma[active].mean(axis=0) if active.any() else chroma.mean(axis=0)
    chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-9)
    tonic, mode, key_conf = estimate_key(chroma_mean)
    bpm, tempo_strength, onset_rate = estimate_tempo(mag, sr)
    frames_per_block = max(1, int(block_seconds * sr / HOP))
    curve = [float(np.mean(frame_db[i:i + frames_per_block])) for i in range(0, len(frame_db), frames_per_block)]
    flat = np.exp(np.mean(np.log(mag + 1e-9), axis=1)) / (mag.mean(axis=1) + 1e-9)             # Spektrale Flachheit (Rauschanteil)
    return {
        "duration": len(audio) / sr, "bpm": round(bpm, 1), "tempo_strength": round(tempo_strength, 3), "onset_rate": round(onset_rate, 2),
        "tonic": tonic, "mode": mode, "key_confidence": round(key_conf, 3), "chroma": [round(float(x), 4) for x in chroma_mean],
        "loudness_db": round(float(np.mean(frame_db[active])) if active.any() else float(frame_db.mean()), 2),
        "dynamic_range_db": round(float(np.percentile(frame_db[active], 95) - np.percentile(frame_db[active], 10)) if active.any() else 0.0, 2),
        "centroid_hz": round(float(np.mean(centroid[active])) if active.any() else float(centroid.mean()), 1),
        "flatness": round(float(np.mean(flat[active])) if active.any() else float(flat.mean()), 4),
        "bands": {k: round(v, 4) for k, v in bands.items()}, "energy_curve": [round(c, 2) for c in curve], "block_seconds": block_seconds,
    }

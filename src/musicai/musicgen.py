"""MusicGen-Anbindung (transformers): Basismodell laden, Audio -> Codes, Delay-Pattern fuer das Training. Schwere Imports erst bei Gebrauch."""
from __future__ import annotations

import os
import importlib.util
from typing import Any

from .config import Config


def require_ai() -> None:
    missing = []
    for name in ("torch", "transformers"):
        try: __import__(name)
        except ImportError: missing.append(name)
    if missing: raise RuntimeError(f"KI-Pakete fehlen: {', '.join(missing)} - 'python start.py setup' installiert sie.")


def pick_device(config: Config) -> str:
    import torch
    wanted = str(config.get("music", "device", "auto"))
    if wanted != "auto": return wanted
    return "cuda" if torch.cuda.is_available() else "cpu"


def quant_mode(config: Config) -> str:
    """none | 8bit | 4bit. auto: 4bit bei large/medium auf der GPU (die fp32-Gewichte von large passen nicht in 6 GB VRAM)."""
    import torch
    mode = str(config.get("music", "quantize", "auto")).lower()
    if mode != "auto": return mode if mode in ("none", "8bit", "4bit") else "none"
    big = any(w in str(config.get("music", "base_model", "")).lower() for w in ("large", "medium"))
    return "4bit" if big and pick_device(config).startswith("cuda") else "none"


def check_load_environment(config: Config, device: str | None = None) -> None:
    """Prueft die Voraussetzungen *vor* dem aufwendigen Sammeln/Kodieren von Clips.

    Ein fehlendes bitsandbytes wurde bisher erst nach bis zu mehreren Minuten
    Clip-Sammlung sichtbar. Diese Pruefung laedt kein Modell und keinen Audio-Cache;
    nur die vorhandene Torch-Installation wird fuer die GPU-Erkennung abgefragt.
    """
    require_ai()
    device = device or pick_device(config)
    mode = quant_mode(config)
    if mode != "none" and not device.startswith("cuda"):
        raise RuntimeError("Quantisierung braucht eine NVIDIA-GPU. music.quantize: none setzen oder ein kleineres Modell waehlen.")
    if mode != "none" and importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError("bitsandbytes fehlt fuer die konfigurierte 4/8-Bit-Quantisierung. 'python start.py setup --yes' richtet eine eigene MusicAI-Umgebung mit diesem Paket ein.")


def load_base(config: Config, device: str | None = None) -> tuple[Any, Any]:
    """(Modell, Processor). Cache liegt im Projekt (runtime/cache/huggingface).
    Ohne Quantisierung: fp32 (Training mit autocast). Mit Quantisierung (large): Decoder in 4/8 Bit direkt auf der GPU, Rest fp16/fp32 - LoRA bleibt fp32."""
    check_load_environment(config, device)
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    name = str(config.get("music", "base_model", "facebook/musicgen-small")); cache = str(config.cache)
    os.environ.setdefault("HF_HOME", cache)
    local_only = not bool(config.get("music", "allow_download", True))
    device = device or pick_device(config); mode = quant_mode(config)
    processor = AutoProcessor.from_pretrained(name, cache_dir=cache, local_files_only=local_only)
    kwargs = dict(cache_dir=cache, local_files_only=local_only)
    if mode == "none":
        model = MusicgenForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float32, **kwargs).to(device)
    else:
        try: import bitsandbytes  # noqa: F401
        except ImportError as error: raise RuntimeError("bitsandbytes fehlt (4/8-Bit-Quantisierung fuer musicgen-large) - 'python start.py setup' installiert es.") from error
        if not device.startswith("cuda"): raise RuntimeError("Quantisierung braucht eine NVIDIA-GPU (music.quantize: none setzen oder ein kleineres Modell waehlen).")
        from transformers import BitsAndBytesConfig
        skip = ["text_encoder", "audio_encoder", "enc_to_dec_proj", "lm_heads", "embed_tokens"]
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=skip)             if mode == "4bit" else BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        model = MusicgenForConditionalGeneration.from_pretrained(name, quantization_config=quant, device_map={"": device}, torch_dtype=torch.float16, **kwargs)
        for part in (model.text_encoder, model.audio_encoder, model.enc_to_dec_proj): part.float()      # T5/EnCodec in fp16 neigen zu Ueberlaeufen; sind klein
    model.is_quantized_base = mode != "none"
    fix_config(model)
    return model, processor


def fix_config(model: Any) -> None:
    """Die Original-Konfiguration von musicgen-small setzt decoder_start_token_id nicht (nur bos/pad = 2048); fuer das Training mit labels wird es gebraucht."""
    decoder = model.config.decoder
    if getattr(decoder, "decoder_start_token_id", None) is None: decoder.decoder_start_token_id = decoder.bos_token_id
    if getattr(decoder, "pad_token_id", None) is None: decoder.pad_token_id = decoder.bos_token_id


def delay_labels(codes: Any, pad_token_id: int) -> Any:
    """Codes (bsz, K, T) -> Trainingslabels (bsz, T+K-1, K) im Delay-Pattern von MusicGen (Codebook k um k Schritte verzoegert, Rest = Pad)."""
    import torch
    bsz, num_codebooks, steps = codes.shape
    out = torch.full((bsz, num_codebooks, steps + num_codebooks - 1), pad_token_id, dtype=torch.long, device=codes.device)
    for k in range(num_codebooks): out[:, k, k:k + steps] = codes[:, k]
    return out.transpose(1, 2).contiguous()


def encode_audio(model: Any, audio: Any, device: str) -> Any:
    """audio: Tensor (bsz, 1, samples) -> Codes (bsz, K, T) via EnCodec (ohne Gradienten)."""
    import torch
    with torch.no_grad():
        out = model.audio_encoder.encode(audio.to(device), padding_mask=torch.ones_like(audio, dtype=torch.long, device=device))
    return out.audio_codes[0]

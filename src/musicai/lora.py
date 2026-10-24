"""Eigene, abhaengigkeitsfreie LoRA-Implementierung (nur torch) fuer den MusicGen-Decoder.

* inject()  - ersetzt ausgewaehlte nn.Linear-Schichten durch LoRA-Schichten (nur A/B werden trainiert)
* save()/load() - nur die Adapter-Gewichte (wenige MB) liegen in models/<creator>/adapter/
* merge()   - addiert gewichtete Adapter-Deltas direkt in die Basisgewichte: W += staerke * w_i * (alpha/r) * B_i @ A_i
              -> beliebig viele Stile mischen, ohne Laufzeit-Mehraufwand
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base, self.rank, self.scale = base, rank, alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))          # B startet bei 0 -> das Modell beginnt exakt wie das Basismodell
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.dropout(x).to(self.A.dtype) @ self.A.t()) @ self.B.t()
        return self.base(x) + (delta * self.scale).to(x.dtype)


class MixedLoRALinear(nn.Module):
    """Gewichteter Adapter-Mix ohne Veraendern der Basisgewichte (noetig bei 4/8-Bit-Quantisierung): alle Adapter sind entlang des Rangs gestapelt,
    die Mischgewichte stecken schon in B -> nur zwei Matrixprodukte, egal wie viele Stile gemischt werden."""
    def __init__(self, base: nn.Linear, a_cat: torch.Tensor, b_cat: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("A", a_cat.to(base.weight.device), persistent=False)
        self.register_buffer("B", b_cat.to(base.weight.device), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + ((x.to(self.A.dtype) @ self.A.t()) @ self.B.t()).to(x.dtype)


def target_names(root: nn.Module) -> list[str]:
    return [name for name, module in root.named_modules() if isinstance(module, nn.Linear) and name.split(".")[-1] in TARGET_SUFFIXES]


def _parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    *path, leaf = name.split(".")
    for part in path: root = getattr(root, part)
    return root, leaf


def freeze(model: nn.Module) -> None:
    """Friert das gesamte Modell ein (auch Text-/Audio-Encoder) - danach werden nur neu erzeugte LoRA-Parameter trainiert."""
    for p in model.parameters(): p.requires_grad_(False)


def inject(root: nn.Module, rank: int = 16, alpha: float = 32, dropout: float = 0.05) -> list[str]:
    """Friert alle Parameter von root ein und ersetzt die Ziel-Schichten durch LoRA-Schichten. -> Namen der ersetzten Schichten"""
    for p in root.parameters(): p.requires_grad_(False)
    names = target_names(root)
    for name in names:
        parent, leaf = _parent(root, name)
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), rank, alpha, dropout))
    return names


def lora_modules(root: nn.Module) -> dict[str, LoRALinear]:
    return {name: m for name, m in root.named_modules() if isinstance(m, LoRALinear)}


def trainable_parameters(root: nn.Module) -> list[nn.Parameter]:
    return [p for m in lora_modules(root).values() for p in (m.A, m.B)]


def save(root: nn.Module, folder: Path, meta: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    mods = lora_modules(root)
    if not mods: raise ValueError("Keine LoRA-Schichten im Modell.")
    first = next(iter(mods.values()))
    state = {name: {"A": m.A.detach().cpu().half(), "B": m.B.detach().cpu().half()} for name, m in mods.items()}
    torch.save({"rank": first.rank, "scale": first.scale, "weights": state}, folder / "lora.pt")
    (folder / "adapter.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder / "lora.pt"


def load_file(folder: Path) -> dict:
    return torch.load(folder / "lora.pt", map_location="cpu", weights_only=True)


@torch.no_grad()
def merge(root: nn.Module, adapters: list[tuple[dict, float]], strength: float = 1.0) -> int:
    """Addiert gewichtete Adapter in die Basisgewichte (root = Modell OHNE injizierte LoRA-Schichten). Gewichte muessen vorher normiert sein.
    -> Anzahl veraenderter Schichten"""
    modules = dict(root.named_modules()); changed = 0
    for adapter, weight in adapters:
        for name, ab in adapter["weights"].items():
            module = modules.get(name)
            if not isinstance(module, nn.Linear): continue
            delta = (ab["B"].float() @ ab["A"].float()) * adapter["scale"] * weight * strength
            module.weight.add_(delta.to(module.weight.device, module.weight.dtype)); changed += 1
    return changed


@torch.no_grad()
def attach(root: nn.Module, adapters: list[tuple[dict, float]], strength: float = 1.0) -> int:
    """Wie merge(), aber als Zusatzschicht statt in die Gewichte: funktioniert auch mit quantisierter Basis. Gewichte vorher normieren. -> Anzahl Schichten"""
    modules = dict(root.named_modules()); per_layer: dict[str, list] = {}
    for adapter, weight in adapters:
        for name, ab in adapter["weights"].items():
            per_layer.setdefault(name, []).append((ab["A"].float(), ab["B"].float() * (adapter["scale"] * weight * strength)))
    changed = 0
    for name, parts in per_layer.items():
        base = modules.get(name)
        if not isinstance(base, nn.Linear): continue
        parent, leaf = _parent(root, name)
        setattr(parent, leaf, MixedLoRALinear(base, torch.cat([a for a, _ in parts], dim=0), torch.cat([b for _, b in parts], dim=1))); changed += 1
    return changed

"""ML-Pfad mit einem winzigen, zufaellig initialisierten MusicGen (kein Download): LoRA-Training, Speichern, Mischen, Erzeugen."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from musicai import lora
from musicai.musicgen import delay_labels, fix_config
from musicai.train import train_loop


class StubTokenizer:
    def __call__(self, texts, padding=True, return_tensors="pt"):
        ids = torch.tensor([[(ord(c) % 90) + 1 for c in (t + " " * 12)[:12]] for t in texts])
        class Out:
            input_ids = ids; attention_mask = torch.ones_like(ids)
            def to(self, device): return self
        return Out()


class StubProcessor:
    tokenizer = StubTokenizer()


class Ctx:
    def __init__(self): self.messages = []
    def check(self): pass
    def log(self, message, level=None): self.messages.append(message)
    def progress(self, value, message=None):
        if message: self.messages.append(message)


def tiny_model():
    from transformers import EncodecConfig, MusicgenConfig, MusicgenDecoderConfig, MusicgenForConditionalGeneration, T5Config
    text = T5Config(vocab_size=100, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=4)
    audio = EncodecConfig(sampling_rate=32000, target_bandwidths=[1.2], upsampling_ratios=[8, 5, 4, 4], num_filters=8, hidden_size=16,
                          codebook_size=64, codebook_dim=16, num_residual_layers=1, audio_channels=1, chunk_length_s=None, overlap=None)
    decoder = MusicgenDecoderConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, ffn_dim=64, num_codebooks=4,
                                    pad_token_id=64, bos_token_id=64, max_position_embeddings=256)      # wie musicgen-small: kein decoder_start_token_id
    config = MusicgenConfig.from_sub_models_config(text, audio, decoder)
    torch.manual_seed(0)
    model = MusicgenForConditionalGeneration(config).eval()
    assert model.config.decoder.decoder_start_token_id is None
    fix_config(model)
    assert model.config.decoder.decoder_start_token_id == 64
    return model


def test_delay_labels_pattern():
    codes = torch.arange(1, 9).reshape(1, 2, 4)           # K=2, T=4
    out = delay_labels(codes, pad_token_id=0)             # (1, T+K-1, K)
    assert out.shape == (1, 5, 2)
    assert out[0, :, 0].tolist() == [1, 2, 3, 4, 0]       # Codebook 0 unverzoegert
    assert out[0, :, 1].tolist() == [0, 5, 6, 7, 8]       # Codebook 1 um einen Schritt verzoegert


def test_lora_starts_identical_and_train_reduces_loss(tmp_path):
    model = tiny_model()
    pattern = torch.stack([(torch.arange(30) * (k + 1) + k) % 8 for k in range(4)])      # lernbare Struktur (kein Rauschen), je Codebook anders
    labels_codes = [pattern.clone() for _ in range(4)]
    captions = ["clip"] * 4
    tok = StubTokenizer(); text = tok(captions[:1]); labels = delay_labels(labels_codes[0][None], 64)
    base_logits = model(input_ids=text.input_ids, attention_mask=text.attention_mask, labels=labels).logits
    lora.freeze(model); names = lora.inject(model.decoder, rank=4, alpha=8, dropout=0.0)
    assert names and all(n.split(".")[-1] in lora.TARGET_SUFFIXES for n in names)
    after = model(input_ids=text.input_ids, attention_mask=text.attention_mask, labels=labels).logits
    assert torch.allclose(base_logits, after, atol=1e-5), "B=0 -> Start muss dem Basismodell entsprechen"
    trainable = {id(p) for p in lora.trainable_parameters(model.decoder)}
    assert {p.requires_grad for p in model.parameters() if id(p) not in trainable} == {False}, "Basisgewichte muessen eingefroren sein"
    model.eval()
    with torch.no_grad(): before = float(model(input_ids=text.input_ids, attention_mask=text.attention_mask, labels=labels).loss)
    settings = {"epochs": 10, "batch_size": 1, "grad_accum": 1, "learning_rate": 1e-2, "fp16": False}
    ctx = Ctx(); result = train_loop(ctx, model, StubProcessor(), labels_codes, captions, settings, "cpu")
    assert result["steps"] == 40 and result["loss"] == result["loss"]
    assert any(float(m.B.abs().max()) > 0 for m in lora.lora_modules(model.decoder).values()), "LoRA-B wurde nie aktualisiert"
    model.eval()
    with torch.no_grad(): trained = model(input_ids=text.input_ids, attention_mask=text.attention_mask, labels=labels)
    assert float(trained.loss) < before - 0.05, (before, float(trained.loss))      # Loss sinkt messbar (winziges Zufallsmodell, nur 40 Schritte)
    a = trained.logits
    folder = tmp_path / "adapter"; lora.save(model.decoder, folder, {"x": 1})
    assert (folder / "lora.pt").exists() and (folder / "adapter.json").exists()

    # Merge in ein frisches Basismodell == Modell mit LoRA-Schichten
    fresh = tiny_model(); lora.merge(fresh.decoder, [(lora.load_file(folder), 1.0)])
    with torch.no_grad():
        b = fresh(input_ids=text.input_ids, attention_mask=text.attention_mask, labels=labels).logits
    assert torch.allclose(a, b, atol=5e-3), float((a - b).abs().max())      # Adapter liegen in fp16 gespeichert

    # Mischen: Gewicht 0.5 liegt zwischen Basis und vollem Adapter
    half = tiny_model(); lora.merge(half.decoder, [(lora.load_file(folder), 0.5)])
    with torch.no_grad():
        c = half(input_ids=text.input_ids, attention_mask=text.attention_mask, labels=labels).logits
    assert not torch.allclose(c, a, atol=1e-3) and not torch.allclose(c, base_logits, atol=1e-3)


def test_generate_runs_with_merged_model():
    model = tiny_model(); tok = StubTokenizer(); text = tok(["calm"])
    out = model.generate(input_ids=text.input_ids, attention_mask=text.attention_mask, max_new_tokens=40, do_sample=True, guidance_scale=3.0)
    assert out.ndim == 3 and out.shape[1] == 1 and out.shape[2] > 0


def test_attach_equals_merge(tmp_path):
    """Adapter-Mix als Zusatzschicht (quantisierte Basis) == Einrechnen in die Gewichte; Mischgewichte stimmen."""
    model = tiny_model(); lora.freeze(model); lora.inject(model.decoder, rank=4, alpha=8, dropout=0.0)
    for m in lora.lora_modules(model.decoder).values(): torch.nn.init.normal_(m.B, std=0.05)      # B != 0, sonst waere der Test leer
    one = tmp_path / "a"; two = tmp_path / "b"; lora.save(model.decoder, one, {}); 
    for m in lora.lora_modules(model.decoder).values(): torch.nn.init.normal_(m.B, std=0.05)
    lora.save(model.decoder, two, {})
    adapters = [(lora.load_file(one), 0.7), (lora.load_file(two), 0.3)]
    text = StubTokenizer()(["x"]); labels = delay_labels(torch.randint(0, 64, (1, 4, 30)), 64)
    merged, attached = tiny_model(), tiny_model()
    assert lora.merge(merged.decoder, adapters, strength=0.8) == lora.attach(attached.decoder, adapters, strength=0.8) > 0
    with torch.no_grad():
        a = merged(input_ids=text.input_ids, attention_mask=text.attention_mask, labels=labels).logits
        b = attached(input_ids=text.input_ids, attention_mask=text.attention_mask, labels=labels).logits
        base = tiny_model()(input_ids=text.input_ids, attention_mask=text.attention_mask, labels=labels).logits
    assert torch.allclose(a, b, atol=1e-3), float((a - b).abs().max())
    assert not torch.allclose(a, base, atol=1e-3)

"""FreyaTTS high-level synthesis pipeline.

Usage:
    from freyatts import FreyaTTS
    tts = FreyaTTS.from_pretrained("freyavoice/freya-tts", device="cuda")
    wav = tts.synthesize("Merhaba, size nasıl yardımcı olabilirim?")
    tts.save_wav(wav, "output.wav")
"""

import json
import math
import os
import re

import numpy as np
import torch

from .model import LEYLA_SEED, FreyaDiT
from .vae import load_audio_vae

FILL_ID = 0
UNK_ID = 1

SAMPLE_RATE = 48000

# minimum voiced fraction (pyin) below which a clause is re-sampled
VOICED_FRAC_MIN = 0.06

# English/foreign words the character vocabulary cannot pronounce as written;
# transliterated to Turkish phonetics. SFT already teaches common Turkish
# acronyms natively, so acronyms are left untouched.
BRAND = {
    "platinum": "platinyum",
    "mastercard": "mastırkard",
    "visa": "viza",
    "signature": "siğnetçır",
    "american express": "amerikan ekspres",
    "qr kod": "kju ar kod",
    "online": "onlayn",
    "mobil": "mobil",
}

_UNITS = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
_SCALES = [(10 ** 9, "milyar"), (10 ** 6, "milyon"), (10 ** 3, "bin")]


def spell_number(n):
    """Spell a non-negative integer in Turkish (0 -> 'sıfır')."""
    if n == 0:
        return "sıfır"
    parts = []
    for scale, name in _SCALES:
        if n >= scale:
            head = n // scale
            n = n % scale
            # Turkish drops the leading 'bir' before 'bin'
            if not (scale == 1000 and head == 1):
                parts.append(spell_number(head))
            parts.append(name)
    if n >= 100:
        if n // 100 > 1:
            parts.append(_UNITS[n // 100])
        parts.append("yüz")
        n = n % 100
    if n >= 10:
        parts.append(_TENS[n // 10])
        n = n % 10
    if n > 0:
        parts.append(_UNITS[n])
    return " ".join(parts)


def expand_digits(text):
    """Rewrite digit runs in their spoken Turkish form.

    A digit string is orthographically short but phonetically long, and the
    duration head sizes the utterance from the character sequence, so numbers
    left as digits come out truncated. Spelling them out before synthesis is
    part of the input contract. Clock times read as hour then minute, long
    account-style strings digit by digit, everything else as an integer.
    Multi-digit strings with a leading zero are read digit by digit as well,
    so the zeros are not lost.
    """
    def spoken(match):
        s = match.group(0)
        if ":" in s:
            hour, minute = s.split(":")
            out = spell_number(int(hour))
            if int(minute):
                out += " " + spell_number(int(minute))
            return out
        if "." not in s and (len(s) >= 6 or (len(s) > 1 and s[0] == "0")):
            return " ".join(spell_number(int(ch)) for ch in s)
        return spell_number(int(s.replace(".", "")))

    return re.sub(r"\d+:\d+|\d+(?:\.\d+)*", spoken, text)


def normalize(text):
    """Light text normalization: transliterate foreign words, spell out digit
    runs, collapse punctuation."""
    t = text
    for k in sorted(BRAND, key=len, reverse=True):
        t = re.sub(r"(?i)\b" + re.escape(k) + r"\b", BRAND[k], t)

    t = expand_digits(t)
    t = t.replace("...", ", ").replace("…", ", ").replace(" — ", ", ").replace(" - ", ", ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


class FreyaTTS:
    """Text-to-speech pipeline around FreyaDiT and the frozen VoxCPM2 AudioVAE.

    A fixed noise seed gives one deterministic voice. Long inputs are split
    at clause boundaries and synthesized per clause with the same seed, then
    concatenated with short gaps.
    """

    def __init__(self, model, vae, char_to_id, device="cuda", seed=LEYLA_SEED, t_floor=8, max_words=11):
        self.model = model
        self.vae = vae
        self.char_to_id = char_to_id
        self.device = device
        self.seed = LEYLA_SEED if seed is None else seed
        self.t_floor = t_floor
        self.max_words = max_words
        self.sample_rate = SAMPLE_RATE

    @classmethod
    def from_pretrained(cls, model_id_or_path: str = "freyavoice/freya-tts", device: str = "cuda") -> "FreyaTTS":
        """Load FreyaTTS from a Hugging Face repo id or a local directory.

        Expects `config.json` and `model.safetensors` in the repo/directory.
        The AudioVAE is fetched separately from openbmb/VoxCPM2.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        if os.path.isdir(model_id_or_path):
            config_path = os.path.join(model_id_or_path, "config.json")
            weights_path = os.path.join(model_id_or_path, "model.safetensors")
        else:
            config_path = hf_hub_download(model_id_or_path, "config.json")
            weights_path = hf_hub_download(model_id_or_path, "model.safetensors")

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        model = FreyaDiT(
            vocab=cfg["vocab"],
            d=cfg["d"],
            depth=cfg["depth"],
            heads=cfg["heads"],
            ff=cfg["ff"],
        )
        model.load_state_dict(load_file(weights_path), strict=True)
        model = model.to(device).eval()

        vae = load_audio_vae(device)

        vocab_path = os.path.join(os.path.dirname(__file__), "char_vocab.json")
        with open(vocab_path, encoding="utf-8") as f:
            char_to_id = json.load(f)

        return cls(model, vae, char_to_id, device=device)

    def synthesize(self, text: str, steps: int = 32, seed: int = LEYLA_SEED) -> np.ndarray:
        """Synthesize `text` and return a float32 waveform at 48 kHz.

        Args:
            text: Input text (Turkish).
            steps: Euler ODE steps for the flow-matching sampler.
            seed: Noise seed, which selects the speaker — the model has no speaker
                conditioning, so x0 *is* the voice. The default (LEYLA_SEED) gives the
                canonical Leyla voice deterministically; other seeds give other people,
                and some are not Leyla at all.
        """
        wav, _, _ = self._synth(text, steps=steps, seed=seed)
        return wav

    def save_wav(self, wav: np.ndarray, path: str):
        """Write a waveform returned by `synthesize` to a 48 kHz wav file."""
        import soundfile as sf

        sf.write(path, wav, self.sample_rate)

    # ---- internals ----

    def _ids(self, text):
        return torch.tensor([[self.char_to_id.get(ch, UNK_ID) for ch in text]], device=self.device)

    @torch.no_grad()
    def _synth_one(self, text, steps=32, seed=None):
        seed = self.seed if seed is None else seed
        ids = self._ids(text)
        cmask = torch.ones_like(ids, dtype=torch.bool)

        # duration head: masked mean over text features -> log frame count
        te = self.model.text_encode(ids)
        pooled = (te * cmask[..., None].float()).sum(1) / (cmask.sum(1, keepdim=True) + 1e-6)
        T = int(round(math.exp(float(self.model.dur(pooled).squeeze(-1)))))
        # floor keeps short inputs from collapsing, cap keeps runaways bounded
        T = max(self.t_floor, ids.shape[1] + 4, min(300, T))

        # fixed seed = fixed voice
        latents = self.model.sample(ids, T, steps=steps, cmask=cmask, seed=seed)

        wav = self.vae.decode(latents.transpose(1, 2).float()).squeeze().float().cpu().numpy()
        return wav

    def _voiced_ok(self, wav):
        # a near-zero voiced fraction means the sample collapsed to noise/silence
        try:
            import librosa

            y = librosa.resample(wav.astype(np.float32), orig_sr=self.sample_rate, target_sr=16000)
            f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=16000)
            voiced_frac = float(np.mean(~np.isnan(f0)))
            return voiced_frac >= VOICED_FRAC_MIN
        except Exception:
            return True

    def _clauses(self, text):
        """Split text at punctuation into chunks of at most `max_words` words."""
        parts = re.split(r"(?<=[\.\?\!,:;])\s+", text)
        out = []
        cur = ""
        for p in parts:
            if len((cur + " " + p).split()) <= self.max_words:
                cur = (cur + " " + p).strip()
            else:
                if cur:
                    out.append(cur)
                cur = p
        if cur:
            out.append(cur)

        # hard-split any clause that is still too long
        final = []
        for c in out:
            words = c.split()
            if len(words) <= self.max_words + 4:
                final.append(c)
            else:
                for i in range(0, len(words), self.max_words):
                    final.append(" ".join(words[i : i + self.max_words]))
        return [c for c in final if c.strip()]

    def _synth(self, text, steps=32, seed=None, do_norm=True, do_chunk=True):
        seed = self.seed if seed is None else seed
        t = normalize(text) if do_norm else text
        # very short inputs rely on the duration floor

        if do_chunk and len(t.split()) > self.max_words:
            chunks = self._clauses(t)
        else:
            chunks = [t]

        wavs = []
        gap = np.zeros(int(0.12 * self.sample_rate), dtype=np.float32)
        for c in chunks:
            w = self._synth_one(c, steps=steps, seed=seed)
            if not self._voiced_ok(w):
                # unvoiced collapse: retry with seeds near the voice seed
                for offset in (1, 2, 3):
                    w2 = self._synth_one(c, steps=steps, seed=seed + offset)
                    if self._voiced_ok(w2):
                        w = w2
                        break
            wavs.append(w.astype(np.float32))
            wavs.append(gap)

        if len(wavs) > 1:
            wav = np.concatenate(wavs[:-1])
        else:
            wav = wavs[0]
        return wav, t, chunks

---
language:
  - tr
license: apache-2.0
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - tts
  - turkish
  - flow-matching
  - diffusion-transformer
  - speech-synthesis
library_name: freyatts
---

# FreyaTTS-small

FreyaTTS-small is a 183M-parameter Turkish text-to-speech model, and the open-source member of the FreyaTTS family. It is tokenizer-free at the character level (92 symbols, no phonemizer or G2P) and generates speech with a non-autoregressive conditional flow-matching DiT in the frozen [AudioVAE2](https://huggingface.co/openbmb/VoxCPM2) latent space (25 Hz, 64-dim latents, 16 kHz encode / 48 kHz decode). Output is 48 kHz mono.

- **Repository:** https://github.com/freyavoiceai/FreyaTTS
- **Eval set:** https://huggingface.co/datasets/freyavoice/freya-tr-eval
- **License:** Apache-2.0

## The FreyaTTS family

**FreyaTTS-small** (this model) is our compact, Apache-2.0, self-hostable model, released in full — weights, inference code, and training pipeline.

**FreyaTTS-large** is our production model, serving Turkish voice agents at [FreyaVoice](https://app.freyavoice.ai) with higher naturalness and expressivity. It is available commercially rather than as open weights — for access, contact us at **dev@freyavoice.ai**.

The model id `freyavoice/freya-tts` predates this naming and is unchanged; it refers to FreyaTTS-small.

## Usage

```python
from freyatts import FreyaTTS

tts = FreyaTTS.from_pretrained("freyavoice/freya-tts", device="cuda")
wav = tts.synthesize("Merhaba, size nasıl yardımcı olabilirim?")   # np.float32, 48 kHz
tts.save_wav(wav, "output.wav")
```

Clone [freyavoiceai/FreyaTTS](https://github.com/freyavoiceai/FreyaTTS) and install `requirements.txt`.

## Model details

- **Architecture:** conditional flow-matching diffusion transformer, non-autoregressive, 32-step Euler ODE, no CFG
- **Parameters:** 183.2M
- **Input:** character-level Turkish, 92-symbol vocabulary
- **Latent space:** frozen AudioVAE2 (Apache-2.0, [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)), 64-dim at 25 Hz, decodes to 48 kHz
- **Training:** from scratch on Turkish speech; pretraining followed by SFT stage 1/2 (voice lock, short-utterance coverage)
- **Voice:** single target speaker, no cloning

## Evaluation

On [Freya-TR-Eval](https://huggingface.co/datasets/freyavoice/freya-tr-eval): **WER 8.0% / CER 3.0%**, ranking 3rd of 7 among open sub-1B Turkish TTS models, ahead of XTTS-v2 (11.1% WER) and F5-TTS (24.3% WER).

## Speed

- RTX 4090: RTF 0.10-0.11, TTFT ~0.5 s, 1.5 GB VRAM, 9.4 audio-s/s at concurrency 4
- About 3.2x faster RTF and 3.7x less VRAM than the 2B VoxCPM2
- Apple M3 laptop CPU: RTF 0.70 (fp32); ~0.12 end to end via Core ML on Apple silicon

## Citation

FreyaTTS-small is described in our technical report, [arXiv:2607.09530](https://arxiv.org/abs/2607.09530):

```bibtex
@misc{pamuk2026freyattscompacttokenizerfreeflowmatching,
      title={FreyaTTS: A Compact Tokenizer-Free Flow-Matching Transformer for Turkish-First Speech Synthesis}, 
      author={Ahmet Erdem Pamuk and Ömer Yentür and Ahmet Tunga Bayrak and Yavuz Alp Sencer Öztürk and Mustafa Yavuz},
      year={2026},
      eprint={2607.09530},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.09530}, 
}
```

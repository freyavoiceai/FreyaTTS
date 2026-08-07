
#!/usr/bin/env python3
"""TTS speed benchmark for a single GPU.

Metrics per system: load time, parameter count, peak VRAM, per-bucket
latency / TTFT / RTF (median of --runs after --warmup), and throughput at
concurrency C (C worker processes, each with its own engine, splitting a
fixed utterance queue).

TTFT definition:
- native streaming systems report time to first streamed chunk;
- systems without a streaming API report full-utterance latency as TTFT.

Usage:
  python3 eval/speed.py --system freyatts --out results/freyatts_speed.json
  python3 eval/speed.py --system freyatts --concurrency 1,2,4,8
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------- system adapters -----------------------------
# Each adapter exposes:
#   synth(text)       -> (waveform, sample_rate)
#   synth_ttft(text)  -> (waveform, sample_rate, seconds_to_first_audio)
#   params            -> int or None
#   streaming         -> "native" | "none"

class FreyaTTSAdapter:
    # FreyaTTS exposes synthesis through a non-streaming API here.
    # Therefore TTFT is reported as full-utterance latency rather than
    # performing an additional synthesis just to estimate first-clause time.
    streaming = "none"

    def __init__(self, args):
        from freyatts import FreyaTTS

        self.tts = FreyaTTS.from_pretrained(args.model, device="cuda")
        self.params = sum(p.numel() for p in self.tts.model.parameters())

    def synth(self, text):
        wav = self.tts.synthesize(text)
        return wav, 48000

    def synth_ttft(self, text):
        t0 = time.time()
        wav = self.tts.synthesize(text)
        elapsed = time.time() - t0
        return wav, 48000, elapsed


class XTTSAdapter:
    """Example third-party adapter: Coqui XTTS-v2, native streaming TTFT.

    Requires the `TTS` package and a reference clip (--ref). Kept here as a
    template for benchmarking other systems.
    """

    streaming = "native"

    def __init__(self, args):
        from TTS.api import TTS as CoquiTTS

        if not args.ref:
            raise SystemExit("xtts needs --ref pointing to a reference wav")

        self.t = CoquiTTS(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        ).to("cuda")

        self.ref = args.ref
        self.model = self.t.synthesizer.tts_model

        latents = self.model.get_conditioning_latents(
            audio_path=[args.ref]
        )
        self.gpt_cond = latents[0]
        self.spk_emb = latents[1]

        self.params = sum(p.numel() for p in self.model.parameters())

    def synth(self, text):
        import numpy as np

        wav = self.t.tts(
            text=text,
            speaker_wav=self.ref,
            language="tr",
        )
        return np.asarray(wav, dtype="float32"), 24000

    def synth_ttft(self, text):
        import torch

        t0 = time.time()
        first = None
        parts = []

        for chunk in self.model.inference_stream(
            text,
            "tr",
            self.gpt_cond,
            self.spk_emb,
        ):
            if first is None:
                first = time.time() - t0
            parts.append(chunk.detach().cpu())

        wav = torch.cat(parts).numpy().astype("float32")
        return wav, 24000, first


ADAPTERS = {
    "freyatts": FreyaTTSAdapter,
    "xtts": XTTSAdapter,
}


# ----------------------------- measurement ---------------------------------

def vram_peak():
    import torch

    return round(
        torch.cuda.max_memory_allocated() / 2 ** 30,
        2,
    )


def run_single(args):
    import torch

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)

    t0 = time.time()
    engine = ADAPTERS[args.system](args)
    load_s = time.time() - t0

    torch.cuda.reset_peak_memory_stats()

    out = {
        "system": args.system,
        "gpu": torch.cuda.get_device_name(0),
        "load_s": round(load_s, 1),
        "params": engine.params,
        "ttft_mode": engine.streaming,
        "buckets": {},
    }

    for bucket, sentences in prompts.items():
        latencies = []
        ttfts = []
        rtfs = []

        for sentence in sentences:
            for _ in range(args.warmup):
                engine.synth(sentence)

            runs_lat = []
            runs_ttft = []
            runs_rtf = []

            for _ in range(args.runs):
                t1 = time.time()

                wav, sr, first = engine.synth_ttft(sentence)

                wall = time.time() - t1
                duration = len(wav) / sr

                runs_lat.append(wall)
                runs_ttft.append(first)
                runs_rtf.append(
                    wall / max(1e-6, duration)
                )

            latencies.append(statistics.median(runs_lat))
            ttfts.append(statistics.median(runs_ttft))
            rtfs.append(statistics.median(runs_rtf))

        out["buckets"][bucket] = {
            "latency_s": round(
                statistics.median(latencies),
                3,
            ),
            "ttft_s": round(
                statistics.median(ttfts),
                3,
            ),
            "rtf": round(
                statistics.median(rtfs),
                3,
            ),
        }

        print(
            bucket,
            out["buckets"][bucket],
            flush=True,
        )

    out["vram_gb_peak"] = vram_peak()
    return out


def run_worker(args):
    """--worker mode: synthesize the given queue slice, print wall + audio seconds."""

    engine = ADAPTERS[args.system](args)

    with open(args.queue_file, encoding="utf-8") as f:
        texts = json.load(f)[args.worker_lo:args.worker_hi]

    if texts:
        engine.synth(texts[0])

    t0 = time.time()
    audio = 0.0

    for text in texts:
        wav, sr = engine.synth(text)
        audio += len(wav) / sr

    print(
        json.dumps(
            {
                "wall": time.time() - t0,
                "audio": audio,
            }
        )
    )


def run_concurrency(args):
    """Throughput sweep: C independent worker processes split a fixed queue."""

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)

    queue = (
        prompts["short"]
        + prompts["medium"]
        + prompts["long"]
    ) * 3

    queue_file = os.path.join(
        HERE,
        "results",
        f"_queue_{args.system}.json",
    )

    os.makedirs(
        os.path.dirname(queue_file),
        exist_ok=True,
    )

    with open(queue_file, "w", encoding="utf-8") as f:
        json.dump(
            queue,
            f,
            ensure_ascii=False,
        )

    passthru = []

    for flag, value in (
        ("--model", args.model),
        ("--ref", args.ref),
    ):
        if value:
            passthru.append(flag)
            passthru.append(value)

    sweep = {}

    for concurrency in [
        int(c)
        for c in args.concurrency.split(",")
    ]:
        per_worker = math.ceil(
            len(queue) / concurrency
        )

        procs = []
        t0 = time.time()

        for i in range(concurrency):
            cmd = [
                sys.executable,
                __file__,
                "--system",
                args.system,
                "--worker",
                "--queue-file",
                queue_file,
                "--worker-lo",
                str(i * per_worker),
                "--worker-hi",
                str(min(
                    len(queue),
                    (i + 1) * per_worker,
                )),
            ]

            cmd += passthru

            procs.append(
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )

        audio = 0.0

        for proc in procs:
            stdout, stderr = proc.communicate()

            lines = [
                ln
                for ln in stdout.strip().splitlines()
                if ln.strip().startswith("{")
            ]

            if not lines:
                print(
                    "WORKER FAILED:",
                    (stderr or "")[-400:],
                    flush=True,
                )
                continue

            audio += json.loads(
                lines[-1]
            )["audio"]

        wall = time.time() - t0

        sweep[concurrency] = {
            "wall_s": round(wall, 1),
            "utt_per_min": round(
                len(queue) / wall * 60,
                1,
            ),
            "audio_s_per_s": round(
                audio / wall,
                2,
            ),
        }

        print(
            "C =",
            concurrency,
            sweep[concurrency],
            flush=True,
        )

    return sweep


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--system",
        default="freyatts",
        choices=sorted(ADAPTERS),
        help="which system to benchmark",
    )

    ap.add_argument(
        "--model",
        default="freyavoice/freya-tts",
        help="FreyaTTS model id or local directory",
    )

    ap.add_argument(
        "--prompts",
        default=os.path.join(
            HERE,
            "prompts_tr.json",
        ),
        help="JSON with short/medium/long prompt buckets",
    )

    ap.add_argument(
        "--runs",
        type=int,
        default=10,
        help="timed runs per sentence",
    )

    ap.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="untimed warmup runs per sentence",
    )

    ap.add_argument(
        "--concurrency",
        default="",
        help="comma-separated worker counts for the throughput sweep, e.g. 1,2,4,8",
    )

    ap.add_argument(
        "--ref",
        default="",
        help="reference wav for cloning systems (e.g. xtts)",
    )

    ap.add_argument(
        "--out",
        default="",
        help="write results JSON here",
    )

    ap.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    ap.add_argument(
        "--queue-file",
        default="",
        help=argparse.SUPPRESS,
    )

    ap.add_argument(
        "--worker-lo",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )

    ap.add_argument(
        "--worker-hi",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )

    args = ap.parse_args()

    if args.worker:
        run_worker(args)
        return

    out = run_single(args)

    if args.concurrency:
        out["concurrency"] = run_concurrency(args)

    path = args.out or os.path.join(
        HERE,
        "results",
        f"{args.system}_speed.json",
    )

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            out,
            f,
            indent=1,
            ensure_ascii=False,
        )

    print("wrote", path)


if __name__ == "__main__":
    main()

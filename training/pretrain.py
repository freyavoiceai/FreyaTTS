#!/usr/bin/env python3
"""From-scratch FreyaTTS pretraining on precomputed AudioVAE latents.

Masked flow-matching loss plus a small duration-prediction loss, bf16 mixed
precision via Accelerate (single or multi GPU). Hyperparameters live in
configs/pretrain.yaml; any CLI flag overrides the config.
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from freyatts.model import FreyaDiT

FILL_ID = 0
UNK_ID = 1


def parse_args():
    # resolve --config first so YAML values become the argparse defaults,
    # then explicit CLI flags override both
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--data", default="data/latents",
                        help="directory of latent shards from precompute_latents.py")
    parser.add_argument("--vocab_json", default="freyatts/char_vocab.json",
                        help="character vocabulary (char -> id)")
    parser.add_argument("--out", default="checkpoints/pretrain",
                        help="checkpoint output directory")
    parser.add_argument("--d_model", type=int, default=768, help="transformer width")
    parser.add_argument("--depth", type=int, default=22, help="number of DiT blocks")
    parser.add_argument("--heads", type=int, default=12, help="attention heads")
    parser.add_argument("--ff", type=int, default=2048, help="feed-forward width")
    parser.add_argument("--steps", type=int, default=150000, help="total optimizer steps")
    parser.add_argument("--batch_size", type=int, default=64, help="per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="peak learning rate")
    parser.add_argument("--warmup", type=int, default=2000, help="linear warmup steps")
    parser.add_argument("--lambda_dur", type=float, default=0.1,
                        help="weight of the duration loss")
    parser.add_argument("--max_frames", type=int, default=500,
                        help="drop clips longer than this many latent frames")
    parser.add_argument("--num_workers", type=int, default=3, help="dataloader workers")
    parser.add_argument("--save_every", type=int, default=10000, help="checkpoint interval")
    parser.add_argument("--log_every", type=int, default=50, help="logging interval")
    parser.add_argument("--resume", default="", help="Accelerate state directory to resume from")
    parser.add_argument("--wandb", action="store_true",
                        help="log to Weights & Biases (project \"freyatts\")")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    if known.config:
        with open(known.config) as f:
            parser.set_defaults(**yaml.safe_load(f))
    return parser.parse_args()


def lr_at(step, warmup, total, base):
    """Linear warmup then cosine decay to 5% of the peak."""
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 * base + 0.95 * base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


class LatentDataset(Dataset):
    """Map-style dataset over precomputed latent shards, held in RAM as fp16."""

    def __init__(self, data_dir, char_to_id, max_frames):
        self.items = []
        for path in sorted(glob.glob(os.path.join(data_dir, "*.pt"))):
            for entry in torch.load(path, weights_only=False):
                latent = entry["latent"]
                n_frames = latent.shape[0]
                if n_frames < 8 or n_frames > max_frames:
                    continue
                text = str(entry.get("text", ""))[:300]
                ids = [char_to_id.get(ch, UNK_ID) for ch in text]
                if len(ids) < 1 or len(ids) > 250:
                    continue
                self.items.append((latent.half(), torch.tensor(ids, dtype=torch.long)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    """Pad latents and text ids to the batch maximum, with boolean masks."""
    max_t = max(latent.shape[0] for latent, _ in batch)
    max_l = max(ids.shape[0] for _, ids in batch)
    batch_size = len(batch)
    feat = batch[0][0].shape[1]

    latents = torch.zeros(batch_size, max_t, feat)
    text = torch.full((batch_size, max_l), FILL_ID, dtype=torch.long)
    frame_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    char_mask = torch.zeros(batch_size, max_l, dtype=torch.bool)
    log_frames = torch.zeros(batch_size)

    for i, (latent, ids) in enumerate(batch):
        t = latent.shape[0]
        l = ids.shape[0]
        latents[i, :t] = latent.float()
        frame_mask[i, :t] = True
        text[i, :l] = ids
        char_mask[i, :l] = True
        log_frames[i] = math.log(t)

    return dict(lat=latents, text=text, fmask=frame_mask, cmask=char_mask, logT=log_frames)


def save_model(model, path, cfg):
    torch.save(dict(state_dict=model.state_dict(), cfg=cfg), path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16")
    device = accelerator.device

    with open(args.vocab_json) as f:
        char_to_id = json.load(f)
    vocab = len(char_to_id)

    model = FreyaDiT(vocab=vocab, feat=64, d=args.d_model, depth=args.depth,
                     heads=args.heads, ff=args.ff, fill_id=FILL_ID)
    model_cfg = dict(vocab=vocab, d=args.d_model, depth=args.depth,
                     heads=args.heads, ff=args.ff, arch="xattn")
    n_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        print(f"[init] FreyaDiT {n_params / 1e6:.1f}M params, vocab={vocab}", flush=True)
        os.makedirs(args.out, exist_ok=True)

    dataset = LatentDataset(args.data, char_to_id, args.max_frames)
    if accelerator.is_main_process:
        print(f"[data] {len(dataset)} clips", flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, collate_fn=collate,
                        persistent_workers=(args.num_workers > 0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    use_wandb = False
    if args.wandb and accelerator.is_main_process:
        try:
            import wandb
            wandb.init(project="freyatts", name=os.path.basename(args.out),
                       config=vars(args) | {"params_M": round(n_params / 1e6, 1)})
            use_wandb = True
        except Exception as e:
            print(f"[wandb] disabled: {str(e)[:60]}", flush=True)

    start_step = 0
    if args.resume and os.path.isdir(args.resume):
        accelerator.load_state(args.resume)
        step_file = os.path.join(args.resume, "step.txt")
        if os.path.exists(step_file):
            with open(step_file) as f:
                start_step = int(f.read())

    model.train()
    step = start_step
    t0 = time.time()
    running = {"cfm": 0.0, "dur": 0.0, "n": 0}
    it = iter(loader)

    while step < args.steps:
        with accelerator.accumulate(model):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: v.to(device) for k, v in batch.items()}

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, args.warmup, args.steps, args.lr)

            unwrapped = accelerator.unwrap_model(model)
            with accelerator.autocast():
                cfm = unwrapped.cfm_loss(batch["lat"], batch["text"], batch["fmask"], batch["cmask"])
                dur, _ = unwrapped.dur_loss(batch["text"], batch["logT"], batch["cmask"])
                loss = cfm + args.lambda_dur * dur

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running["cfm"] += float(cfm)
        running["dur"] += float(dur)
        running["n"] += 1

        if accelerator.sync_gradients:
            step += 1
            if step % args.log_every == 0 and accelerator.is_main_process:
                n = max(1, running["n"])
                samples_per_s = running["n"] * args.batch_size * args.grad_accum * accelerator.num_processes / (time.time() - t0)
                avg_cfm = running["cfm"] / n
                avg_dur = running["dur"] / n
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"step {step:6d}/{args.steps}  cfm {avg_cfm:.4f}  dur {avg_dur:.4f}  "
                      f"lr {lr_now:.2e}  {samples_per_s:.0f} samp/s", flush=True)
                if use_wandb:
                    import wandb
                    wandb.log({"loss/cfm": avg_cfm, "loss/dur": avg_dur,
                               "loss/total": avg_cfm + args.lambda_dur * avg_dur,
                               "lr": lr_now, "samp_per_s": samples_per_s}, step=step)
                running = {"cfm": 0.0, "dur": 0.0, "n": 0}
                t0 = time.time()

            if step % args.save_every == 0 and step > start_step:
                ckpt = os.path.join(args.out, f"step{step}")
                accelerator.save_state(ckpt)
                if accelerator.is_main_process:
                    with open(os.path.join(ckpt, "step.txt"), "w") as f:
                        f.write(str(step))
                    save_model(accelerator.unwrap_model(model),
                               os.path.join(ckpt, "model.pt"), model_cfg)
                    print(f"[ckpt] {ckpt}", flush=True)

    # save_state() must run on every process so each process can save
    # its own accelerator state, including RNG state.
    ckpt = os.path.join(args.out, "final")
    accelerator.save_state(ckpt)

    if accelerator.is_main_process:
        with open(os.path.join(ckpt, "step.txt"), "w") as f:
            f.write(str(step))
        save_model(accelerator.unwrap_model(model), os.path.join(ckpt, "model.pt"), model_cfg)
        print(f"[done] {step}", flush=True)


if __name__ == "__main__":
    main()
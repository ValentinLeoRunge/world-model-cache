"""Train the VQ-VAE tokenizer on Memory-Maze frames.

Loss = MSE(input, recon) + commitment loss (the vq_loss returned by the model;
the codebook itself is updated by EMA, not by this optimizer).

Artifacts, relative to --out_dir:
  checkpoints/tokenizer/step_XXXXXX.pt   every --ckpt_every steps (+ latest.pt)
  outputs/tokenizer/step_XXXXXX.png      every --viz_every steps, on a fixed
                                         held-out batch from the eval split

Smoke test (blueprint #3):
  python -m src.train_tokenizer --episodes_limit 20 --max_steps 200 --batch_size 32
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.episode_dataset import EpisodeFrameDataset
from src.models.vqvae import VQVAE
from src.utils.viz import save_recon_grid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data")
    p.add_argument("--split", default="train0")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--max_steps", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--frame_stride", type=int, default=4)
    p.add_argument("--shuffle_buffer", type=int, default=8)
    p.add_argument("--episodes_limit", type=int, default=None,
                   help="cap #episodes (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--viz_every", type=int, default=1000)
    p.add_argument("--viz_n", type=int, default=8)
    p.add_argument("--resume", default=None,
                   help="checkpoint path, or 'latest' to use <out_dir>/checkpoints/tokenizer/latest.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def infinite(loader):
    """Yield batches forever; each epoch respawns workers -> fresh shuffle."""
    while True:
        for batch in loader:
            yield batch


def fixed_viz_batch(args):
    """A held-out batch from the eval split, grabbed once and reused every viz."""
    ds = EpisodeFrameDataset(
        args.data_root, split="eval", frame_stride=args.frame_stride,
        shuffle_buffer=4, episodes_limit=4, seed=args.seed,
    )
    return next(iter(DataLoader(ds, batch_size=args.viz_n, num_workers=0)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True  # inputs are always 64x64

    ckpt_dir = os.path.join(args.out_dir, "checkpoints", "tokenizer")
    viz_dir = os.path.join(args.out_dir, "outputs", "tokenizer")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    train_ds = EpisodeFrameDataset(
        args.data_root, split=args.split, frame_stride=args.frame_stride,
        shuffle_buffer=args.shuffle_buffer, episodes_limit=args.episodes_limit,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        drop_last=True, pin_memory=(args.device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    viz_batch = fixed_viz_batch(args).to(args.device)

    model = VQVAE().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        path = (os.path.join(ckpt_dir, "latest.pt")
                if args.resume == "latest" else args.resume)
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optim"])
        start_step = ckpt["step"]
        print(f"resumed from {path} at step {start_step}")

    def save_ckpt(step):
        blob = {"step": step, "model": model.state_dict(),
                "optim": opt.state_dict(), "args": vars(args)}
        torch.save(blob, os.path.join(ckpt_dir, f"step_{step:06d}.pt"))
        torch.save(blob, os.path.join(ckpt_dir, "latest.pt"))

    def save_viz(step, recon_loss):
        model.eval()
        with torch.no_grad():
            recon, _, _, _ = model(viz_batch)
        psnr = -10.0 * math.log10(max(recon_loss, 1e-10))
        save_recon_grid(
            viz_batch, recon, os.path.join(viz_dir, f"step_{step:06d}.png"),
            n=args.viz_n, title=f"step {step} | recon MSE {recon_loss:.4f} | PSNR {psnr:.1f}",
        )
        model.train()

    model.train()
    batches = infinite(train_loader)
    pbar = tqdm(range(start_step + 1, args.max_steps + 1), initial=start_step,
                total=args.max_steps)
    for step in pbar:
        x = next(batches).to(args.device, non_blocking=True)

        recon, vq_loss, idx, perplexity = model(x)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + vq_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        if step % args.log_every == 0:
            active = int(idx.unique().numel())
            pbar.set_postfix(recon=f"{recon_loss.item():.4f}",
                             vq=f"{vq_loss.item():.4f}",
                             perp=f"{perplexity.item():.1f}",
                             active=f"{active}/256")

        if step % args.ckpt_every == 0 or step == args.max_steps:
            save_ckpt(step)
        if step % args.viz_every == 0 or step == args.max_steps:
            save_viz(step, recon_loss.item())

    print(f"done at step {args.max_steps}; checkpoints in {ckpt_dir}, grids in {viz_dir}")


if __name__ == "__main__":
    main()

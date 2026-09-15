"""Evaluate a trained VQ-VAE tokenizer checkpoint on the eval split.

Metrics (blueprint): mean MSE, PSNR, codebook perplexity, active-code fraction.
Also saves original-vs-recon grids -- the qualitative sign-off is the real bar.

  python -m src.eval_tokenizer --ckpt checkpoints/tokenizer/latest.pt
Smoke test (blueprint #4): add --episodes_limit 10 --num_frames 200

`evaluate()` is importable so the training loop can call it in-process.
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.episode_dataset import EpisodeFrameDataset
from src.models.vqvae import VQVAE
from src.utils.viz import save_recon_grid


@torch.no_grad()
def evaluate(model, data_root="data", num_frames=2000, batch_size=256,
             frame_stride=4, episodes_limit=None, seed=0, device="cpu",
             grid_dir=None, num_grids=2, viz_n=8):
    """Return {frames, mse, psnr, perplexity, active_frac}. Does not mutate model."""
    model.eval()
    ds = EpisodeFrameDataset(
        data_root, split="eval", frame_stride=frame_stride,
        shuffle_buffer=8, episodes_limit=episodes_limit, seed=seed,
    )
    # num_workers=0 -> dataset uses its fixed seed -> deterministic frame order
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)

    k = model.vq.k
    counts = torch.zeros(k, device=device)
    se_sum, n_pix, seen, grids = 0.0, 0, 0, 0

    for x in loader:
        if seen >= num_frames:
            break
        x = x.to(device)
        recon, _, idx, _ = model(x)
        se_sum += F.mse_loss(recon, x, reduction="sum").item()
        n_pix += x.numel()
        counts += torch.bincount(idx.flatten(), minlength=k)
        seen += x.shape[0]

        if grid_dir and grids < num_grids:
            save_recon_grid(
                x, recon, os.path.join(grid_dir, f"eval_{grids:02d}.png"),
                n=viz_n, title=f"eval grid {grids}",
            )
            grids += 1

    mse = se_sum / max(n_pix, 1)
    p = counts / counts.sum().clamp(min=1)
    return {
        "frames": seen,
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, 1e-10)),
        "perplexity": torch.exp(-(p * (p + 1e-10).log()).sum()).item(),
        "active_frac": (counts > 0).float().mean().item(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/tokenizer/latest.pt")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--num_frames", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--frame_stride", type=int, default=4)
    p.add_argument("--episodes_limit", type=int, default=None)
    p.add_argument("--num_grids", type=int, default=2)
    p.add_argument("--viz_n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = VQVAE().to(args.device)
    model.load_state_dict(ckpt["model"])

    grid_dir = os.path.join(args.out_dir, "outputs", "tokenizer", "eval")
    os.makedirs(grid_dir, exist_ok=True)

    m = evaluate(
        model, args.data_root, num_frames=args.num_frames,
        batch_size=args.batch_size, frame_stride=args.frame_stride,
        episodes_limit=args.episodes_limit, seed=args.seed, device=args.device,
        grid_dir=grid_dir, num_grids=args.num_grids, viz_n=args.viz_n,
    )

    print(f"ckpt {args.ckpt} (step {ckpt.get('step', '?')})")
    print(f"  frames evaluated : {m['frames']}")
    print(f"  mean MSE         : {m['mse']:.5f}")
    print(f"  PSNR             : {m['psnr']:.2f} dB")
    print(f"  perplexity       : {m['perplexity']:.1f} / {model.vq.k}")
    print(f"  active codes     : {m['active_frac'] * 100:.1f}%")
    print(f"  grids            : {grid_dir}")


if __name__ == "__main__":
    main()

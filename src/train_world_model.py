"""Train the world model on data/tokens/.

Loss = masked next-token cross-entropy over frame-token positions (computed by
WorldModel.compute_loss). Purely token-space training -- no images, no VQ-VAE
gradients. The frozen tokenizer is loaded only to decode predicted tokens back
to pixels for the viz grids.

Scheduled sampling (exposure-bias fix), v2 -- real compounding rollout: pure
teacher-forcing never shows the model its own mistakes, so real autoregressive
rollout collapses (see eval_world_model.py's revisit accuracy: flat at chance
for every N bucket before this fix). A first attempt corrupted each
frame-token slot independently at random -- cheap, but the wrong *shape* of
noise: independent single-slot corruption never produces the fully
self-generated, error-compounding context a real rollout does, so it barely
moved the needle (measured: a per-step decay trace showed accuracy falling
off smoothly from k=1 self-generated step to a floor around k=8-10, well
before the window even fills with self-generated content -- the first fix
only nudged the first couple of points on that curve).

This version instead genuinely rolls the trailing --ss_max_k steps of the
training window forward autoregressively, each step conditioned on the
*previous self-generated* steps (real compounding, matching what happens at
inference), not on a single clean pass. To afford this every step, the
per-frame generation itself is approximated as one parallel forward pass
(WorldModel.predict_frame_tokens) instead of generate_last_frame's real
16-call token-by-token loop -- justified by the same diagnostic trace, which
showed within-frame autoregression is *not* the bottleneck (k=1 rollout
accuracy already matches the fully-parallel teacher-forced ceiling almost
exactly). So: k (sampled each step, capped by a linear ramp 0 -> --ss_max_k
over --ss_ramp_steps) trailing steps are generated one at a time under
no_grad, each reading the *previous* iterations' self-generated output as
context; the resulting partially-self-generated window is then scored by one
real graded forward+backward pass against the true targets regardless
(WorldModel.compute_loss's target_tokens). Needs `--ss_max_k` extra true
steps of context loaded up front to bootstrap the earliest self-generated
step's own H-step lookback -- see the WindowDataset call below.

Artifacts, relative to --out_dir:
  checkpoints/worldmodel/step_XXXXXX.pt   every --ckpt_every steps (+ latest.pt)
  outputs/worldmodel/step_XXXXXX.png      every --viz_every steps: a fixed
                                          held-out window's true last frame vs.
                                          the model's predicted one, both
                                          decoded through the frozen tokenizer
                                          (so "true" here means "tokenizer's
                                          own reconstruction of the true
                                          tokens" -- isolates world-model error
                                          from tokenizer blur, same reasoning
                                          as the notebook's 3rd/4th rows)

Smoke test (blueprint #3):
  python -m src.train_world_model --episodes_limit 20 --max_steps 200
"""

import argparse
import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.window_dataset import WindowDataset
from src.models.vqvae import VQVAE
from src.models.world_model import WorldModel
from src.utils.viz import save_recon_grid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/tokens")
    p.add_argument("--split", default="train0")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/latest.pt")
    p.add_argument("--history", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=60_000,
                   help="bumped from the original 30k soft budget: the loss was still "
                        "declining (not plateaued) at 30k in the pre-fix baseline run")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shuffle_buffer", type=int, default=8)
    p.add_argument("--episodes_limit", type=int, default=None,
                   help="cap #episodes (smoke tests)")
    p.add_argument("--ss_max_k", type=int, default=10,
                   help="scheduled-sampling cap: max trailing window steps self-generated "
                        "via real compounding rollout, at full ramp. 0 disables. Diagnostic "
                        "trace showed the accuracy floor starts around k=8-10 self-generated "
                        "steps, so 10 covers the observed collapse range.")
    p.add_argument("--ss_ramp_steps", type=int, default=None,
                   help="steps to linearly ramp the self-generation cap 0 -> ss_max_k "
                        "(default: max_steps // 2, then held at ss_max_k)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--viz_every", type=int, default=1000)
    p.add_argument("--viz_n", type=int, default=8)
    p.add_argument("--resume", default=None,
                   help="checkpoint path, or 'latest' for <out_dir>/checkpoints/worldmodel/latest.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def infinite(loader):
    """Yield batches forever; each epoch respawns workers -> fresh shuffle."""
    while True:
        for batch in loader:
            yield batch


def fixed_viz_batch(args):
    """A held-out window batch from the eval split, grabbed once and reused every viz."""
    ds = WindowDataset(
        args.data_root, split="eval", window=args.history + 1,
        shuffle_buffer=4, episodes_limit=4, seed=args.seed,
    )
    return next(iter(DataLoader(ds, batch_size=args.viz_n, num_workers=0)))


def main():
    args = parse_args()
    if args.ss_ramp_steps is None:
        args.ss_ramp_steps = args.max_steps // 2
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    ckpt_dir = os.path.join(args.out_dir, "checkpoints", "worldmodel")
    viz_dir = os.path.join(args.out_dir, "outputs", "worldmodel")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    H = args.history
    K_MAX = args.ss_max_k
    # +K_MAX extra leading true steps: bootstraps the earliest self-generated
    # step's own H-step lookback (see module docstring)
    train_ds = WindowDataset(
        args.data_root, split=args.split, window=H + 1 + K_MAX,
        shuffle_buffer=args.shuffle_buffer, episodes_limit=args.episodes_limit,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        drop_last=True, pin_memory=(args.device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    viz_tokens, viz_actions = fixed_viz_batch(args)
    viz_tokens, viz_actions = viz_tokens.to(args.device), viz_actions.to(args.device)

    model = WorldModel(history=args.history).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # frozen tokenizer: decode-only, used for viz grids, never trained here
    tok_ckpt = torch.load(args.tokenizer_ckpt, map_location=args.device, weights_only=False)
    tokenizer = VQVAE().to(args.device)
    tokenizer.load_state_dict(tok_ckpt["model"])
    tokenizer.eval()

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

    def save_viz(step, loss_value):
        model.eval()
        predicted = model.generate_last_frame(viz_tokens, viz_actions)  # (B,16)
        with torch.no_grad():
            true_imgs = tokenizer.decode_indices(viz_tokens[:, -1, :])
            pred_imgs = tokenizer.decode_indices(predicted)
        token_acc = (predicted == viz_tokens[:, -1, :]).float().mean().item()
        save_recon_grid(
            true_imgs, pred_imgs, os.path.join(viz_dir, f"step_{step:06d}.png"),
            n=args.viz_n,
            title=f"step {step} | loss {loss_value:.3f} | token-acc {token_acc:.2f}",
        )
        model.train()

    model.train()
    batches = infinite(train_loader)
    pbar = tqdm(range(start_step + 1, args.max_steps + 1), initial=start_step,
                total=args.max_steps)
    for step in pbar:
        tok_ext, act_ext = next(batches)                        # (B, H+1+K_MAX, 16/action_dim)
        tok_ext = tok_ext.to(args.device, non_blocking=True)
        act_ext = act_ext.to(args.device, non_blocking=True)

        k_cap = int(round(K_MAX * min(1.0, step / max(args.ss_ramp_steps, 1))))
        k = random.randint(0, k_cap) if k_cap > 0 else 0

        if k > 0:
            with torch.no_grad():
                roll_ext = tok_ext.clone()
                for idx in range(K_MAX + H + 1 - k, K_MAX + H + 1):
                    pred_window = model.predict_frame_tokens(
                        roll_ext[:, idx - H: idx + 1], act_ext[:, idx - H: idx + 1]
                    )
                    roll_ext[:, idx] = pred_window[:, -1, :]
            input_tokens = roll_ext[:, K_MAX:]
        else:
            input_tokens = tok_ext[:, K_MAX:]

        target_tokens = tok_ext[:, K_MAX:]
        actions_final = act_ext[:, K_MAX:]
        loss = model.compute_loss(input_tokens, actions_final, target_tokens=target_tokens)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        if step % args.log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", k=f"{k}/{k_cap}")

        if step % args.ckpt_every == 0 or step == args.max_steps:
            save_ckpt(step)
        if step % args.viz_every == 0 or step == args.max_steps:
            save_viz(step, loss.item())

    print(f"done at step {args.max_steps}; checkpoints in {ckpt_dir}, grids in {viz_dir}")


if __name__ == "__main__":
    main()

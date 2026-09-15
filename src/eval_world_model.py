"""Evaluate a trained world-model checkpoint: accuracy vs. revisit gap N.

The deliverable is the accuracy-vs-N plot -- per-token prediction accuracy at
a genuine revisit event, as a function of how long ago (N steps) the agent
last saw that spot. The model only has H steps of context, so N > H should
show a visible drop (WORLDMODEL.md Evaluation): those tokens are structurally
outside the window, not just "hard to recall."

This requires real autoregressive rollout, not teacher-forcing. An earlier
version of this script built each event's window as `tokens_all[t2-H:t2+1]`
straight from the *true recorded* array -- i.e. teacher-forced -- so the H
steps before every t2 were always the correct ones regardless of N, and the
plot came out flat no matter how good or bad the model's actual memory was
(caught by comparing against notebooks/inspect_world_model.ipynb's genuine
rollout, which does show the collapse). Fixed here: one continuous
autoregressive rollout per episode, seeded with the true first H+1 steps,
generating every later frame from its own prior output (actions stay
ground-truth throughout -- observed/given, never generated). Batched across
episodes so it's one rollout pass total, not one per revisit event -- a
single eval episode alone can contain thousands of revisit events (any
close-then-away-then-close triple counts), so a separate O(N) rollout per
event would be infeasible.

Note this also captures *self*-consistency, not just recall of the ground
truth: for N <= H, the agent's own prior (self-generated) visit to that spot
is still inside the attention window, so the model can stay consistent with
whatever it already committed to. For N > H that prior visit -- true or
self-generated -- is structurally outside the window, so nothing but a
generic prior is available. That is the exact failure WORLDMODEL.md names.

Also reports plain teacher-forced next-token accuracy on random (non-revisit)
windows as a training-health sanity number, the way PSNR served the tokenizer
-- this one is intentionally teacher-forced (one-step prediction quality),
unrelated to the revisit/rollout question above.

  python -m src.eval_world_model --ckpt checkpoints/worldmodel/latest.pt
Smoke test (blueprint #5): add --episodes_limit 10
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.window_dataset import WindowDataset
from src.models.vqvae import VQVAE
from src.models.world_model import WorldModel
from src.utils.revisits import find_revisit_events
from src.utils.viz import save_recon_grid


def _load_episodes(data_root, split, episodes_limit):
    """-> tokens (E,T,16) int64, actions (E,T,6) float32, positions: list of
    E arrays (T,2), stems: list of E filenames (for reporting)."""
    paths = sorted(glob.glob(os.path.join(data_root, split, "*.npz")))
    if episodes_limit is not None:
        paths = paths[:episodes_limit]

    tokens, actions, positions, stems = [], [], [], []
    for path in paths:
        with np.load(path) as d:
            tokens.append(d["tokens"].astype(np.int64))
            actions.append(d["action"].astype(np.float32))
            positions.append(d["agent_pos"])
        stems.append(os.path.basename(path))
    return np.stack(tokens), np.stack(actions), positions, stems


@torch.no_grad()
def rollout_all_episodes(model, tokens, actions, device):
    """Batched full-episode autoregressive rollout (see module docstring):
    seed every episode's first H+1 frames with the true recorded tokens, then
    generate every later frame from the buffer's own prior output, one
    timestep at a time, all E episodes advanced together as one batch.

    tokens, actions: (E,T,16)/(E,T,6) numpy, same T for every episode
    (guaranteed by the fixed-length Memory Maze recordings).
    -> roll (E,T,16) numpy: true for t<=H, self-generated for t>H.
    """
    H = model.history
    _, t_episode, _ = tokens.shape
    roll = torch.from_numpy(tokens).clone().to(device)
    act = torch.from_numpy(actions).to(device)

    for t in tqdm(range(H + 1, t_episode), desc="rollout"):
        roll[:, t] = model.generate_last_frame(roll[:, t - H: t + 1], act[:, t - H: t + 1])

    return roll.cpu().numpy()


def score_revisits(roll, tokens, positions, history, num_examples=8):
    """roll, tokens: (E,T,16). positions: list of E (T,2) arrays.

    Scores every revisit event whose t2 is past the true seed region (t2 <=
    history would still be inside the true-token seed, not a fair memory
    test). -> events: list of (N, per_token_acc); examples: num_examples
    (true, pred, N) triples spread across the N range, for the qualitative
    grid.
    """
    events = []
    for e in range(roll.shape[0]):
        for t1, t2, n in find_revisit_events(positions[e]):
            if t2 <= history:
                continue
            acc = float((roll[e, t2] == tokens[e, t2]).mean())
            events.append((n, acc, tokens[e, t2], roll[e, t2]))

    events.sort(key=lambda ev: ev[0])  # by N, so the examples below span the range
    idx = np.linspace(0, len(events) - 1, num=min(num_examples, len(events)), dtype=int)
    examples = [(events[i][2], events[i][3], events[i][0]) for i in idx]

    return [(n, acc) for n, acc, _, _ in events], examples


@torch.no_grad()
def teacher_forced_accuracy(model, data_root, split, num_windows, batch_size, seed, device):
    """Fast sanity metric: teacher-forced next-token accuracy on random windows."""
    ds = WindowDataset(data_root, split=split, window=model.window,
                        shuffle_buffer=8, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)

    correct, total, seen = 0, 0, 0
    for tokens, actions in loader:
        if seen >= num_windows:
            break
        tokens, actions = tokens.to(device), actions.to(device)
        pred, tgt, mask = model.predict_targets(tokens, actions)
        correct += (pred[mask] == tgt[mask]).sum().item()
        total += mask.sum().item()
        seen += tokens.shape[0]

    return correct / max(total, 1), seen


def bucket_events(events, history):
    """N <= H, H < N <= 2H, 2H < N <= 4H, N > 4H -> (labels, mean_acc, counts)."""
    H = history
    labels = [f"N<={H}", f"{H}<N<={2*H}", f"{2*H}<N<={4*H}", f"N>{4*H}"]
    ns = np.array([n for n, _ in events])
    accs = np.array([a for _, a in events])
    idx = np.digitize(ns, [H, 2 * H, 4 * H], right=True)  # 0,1,2,3 matching labels
    means = [accs[idx == i].mean() if (idx == i).any() else float("nan") for i in range(4)]
    counts = [int((idx == i).sum()) for i in range(4)]
    return labels, means, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/worldmodel/latest.pt")
    p.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/latest.pt")
    p.add_argument("--data_root", default="data/tokens")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--episodes_limit", type=int, default=50)
    p.add_argument("--num_windows", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_grids", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    history = ckpt["args"]["history"]
    model = WorldModel(history=history).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokens, actions, positions, stems = _load_episodes(args.data_root, "eval", args.episodes_limit)
    print(f"rolling out {len(stems)} episodes ({tokens.shape[1]} steps each) ...")
    roll = rollout_all_episodes(model, tokens, actions, args.device)
    events, examples = score_revisits(roll, tokens, positions, history, args.num_grids)
    labels, bucket_acc, bucket_count = bucket_events(events, history)
    tf_acc, tf_seen = teacher_forced_accuracy(
        model, args.data_root, "eval", args.num_windows, args.batch_size, args.seed, args.device
    )

    print(f"ckpt {args.ckpt} (step {ckpt.get('step', '?')}, H={history})")
    print(f"  revisit events scored : {len(events)}")
    for label, acc, count in zip(labels, bucket_acc, bucket_count):
        print(f"  {label:12s}: acc={acc:.3f}  (n={count})")
    print(f"  teacher-forced acc (random windows, n={tf_seen}): {tf_acc:.3f}")

    out_dir = os.path.join(args.out_dir, "outputs", "worldmodel")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, [0 if np.isnan(a) else a for a in bucket_acc])
    for bar, acc, count in zip(bars, bucket_acc, bucket_count):
        if not np.isnan(acc):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"n={count}",
                     ha="center", fontsize=8)
    ax.axhline(1 / 256, color="gray", linestyle="--", linewidth=1, label="chance (1/256)")
    ax.set_ylabel("per-token accuracy at revisit")
    ax.set_ylim(0, 1)
    ax.set_title(f"revisit accuracy vs. gap N (H={history})")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "accuracy_vs_N.png")
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)
    print(f"  plot : {plot_path}")

    if examples:
        tok_ckpt = torch.load(args.tokenizer_ckpt, map_location=args.device, weights_only=False)
        tokenizer = VQVAE().to(args.device)
        tokenizer.load_state_dict(tok_ckpt["model"])
        tokenizer.eval()

        trues = torch.tensor(np.stack([e[0] for e in examples]), device=args.device)
        preds = torch.tensor(np.stack([e[1] for e in examples]), device=args.device)
        with torch.no_grad():
            true_imgs = tokenizer.decode_indices(trues)
            pred_imgs = tokenizer.decode_indices(preds)
        ns_shown = ",".join(str(e[2]) for e in examples)
        grid_path = os.path.join(out_dir, "eval_revisit_examples.png")
        save_recon_grid(true_imgs, pred_imgs, grid_path, n=len(examples),
                         title=f"revisit examples, N={ns_shown}")
        print(f"  grid : {grid_path}")


if __name__ == "__main__":
    main()

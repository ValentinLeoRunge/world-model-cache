import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datasets.world_model_dataset import WorldModelDataset
from src.models.tokenizer import Tokenizer
from src.models.world_model import ACTION_DIM, COND_DIM, WorldModel
from src.utils.viz import save_recon_grid
import random
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/tokens")
    p.add_argument("--split", default="train0")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--history", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=60_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ss_max_k", type=int, default=16)
    p.add_argument("--ss_ramp_steps", type=int, default=None)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/latest.pt")
    p.add_argument("--viz_every", type=int, default=1000)
    p.add_argument("--viz_n", type=int, default=8)
    p.add_argument("--run_name", default="worldmodel")
    p.add_argument("--cache_size", type=int, default=0)
    p.add_argument("--base_ckpt", default=None)
    p.add_argument("--cache_dropout", type=float, default=0.25)
    p.add_argument("--p_other", type=float, default=0.02)  # keep-prob of windows whose target has no useful snapshot
    p.add_argument("--ar_prob", type=float, default=0.0)   # share of rollouts generated token by token, like inference

    return p.parse_args()

def infinite(loader):
    while True:
        for batch in loader:
            yield batch

def to_cond(actions):
    cond = torch.zeros(*actions.shape[:2], COND_DIM, device=actions.device)
    cond[:, :, :ACTION_DIM] = actions
    return cond

def fixed_viz_batch(args):
    ds = WorldModelDataset(args.data_root, split="eval", window=args.history + 1,
                           shuffle_buffer=4, episodes_limit=4)
    return next(iter(DataLoader(ds, batch_size=args.viz_n, num_workers=0)))

def main():
    args = parse_args()

    if args.ss_ramp_steps is None:
        args.ss_ramp_steps = args.max_steps // 2
    random.seed(0)
    torch.manual_seed(0)
    ar_rng = random.Random(1)  # own stream, so k / cache-dropout draws don't depend on ar_prob

    H, K_MAX = args.history, args.ss_max_k

    ckpt_dir = os.path.join(args.out_dir, "checkpoints",args.run_name)
    viz_dir = os.path.join(args.out_dir, "outputs",args.run_name)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # cache runs keep only ~45 windows per episode, so mix more episodes per batch (32 vs 8)
    train_ds = WorldModelDataset(args.data_root, split=args.split, window=H+1+K_MAX, cache_size=args.cache_size, history=H,
                                 shuffle_buffer=32 if args.cache_size else 8, p_other=args.p_other)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=4, drop_last=True, pin_memory=(args.device=="cuda"), persistent_workers=True)

    model = WorldModel(history=H, cache_size=args.cache_size, vocab_size=256, d_model=256).to(args.device)
    if args.base_ckpt:
        model.load_state_dict(torch.load(args.base_ckpt, map_location=args.device)["model"])
        for blk in model.blocks:
            nn.init.zeros_(blk.cross_attention.proj.weight)
            nn.init.zeros_(blk.cross_attention.proj.bias)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # frozen tokenizer: decode-only, turns predicted tokens back into pixels for the grids
    tokenizer = Tokenizer().to(args.device)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location=args.device, weights_only=False)["model"])
    tokenizer.eval()

    viz_tokens, viz_actions = fixed_viz_batch(args)
    viz_tokens = viz_tokens.to(args.device)
    viz_cond = to_cond(viz_actions.to(args.device))

    def save_viz(step, loss_value):
        model.eval()
        predicted = model.generate_last_frame(viz_tokens, viz_cond)      # (B,16)
        with torch.no_grad():
            true_imgs = tokenizer.decode_indices(viz_tokens[:, -1, :])
            pred_imgs = tokenizer.decode_indices(predicted)
        acc = (predicted == viz_tokens[:, -1, :]).float().mean().item()
        save_recon_grid(true_imgs, pred_imgs,
                        os.path.join(viz_dir, f"step_{step:06d}.png"), n=args.viz_n,
                        title=f"step {step} | loss {loss_value:.3f} | token-acc {acc:.2f}")
        model.train()

    model.train()
    batches = infinite(train_loader)
    pbar = tqdm(range(1, args.max_steps + 1))

    for step in pbar:
        batch = [x.to(args.device, non_blocking=True) for x in next(batches)]
        if args.cache_size:
            tok_ext, act_ext, ctok, ccond, cmask, useful = batch
        else:
            (tok_ext, act_ext), ctok, ccond, cmask = batch, None, None, None

        cond_ext = to_cond(act_ext)

        # make autoregressive training occur increasingly over time
        k_cap = round(K_MAX * min(1.0, step/args.ss_ramp_steps))
        k = random.randint(0, k_cap)

        if ctok is not None and random.random() < args.cache_dropout:
            ct = cc = cm = None
        else:
            ct, cc, cm = ctok, ccond, cmask

        # cache view for the frame at tok_ext index idx (the dataset yields one per idx >= H)
        def cache_at(idx):
            return (None, None, None) if ct is None else (ct[:, idx-H], cc[:, idx-H], cm[:, idx-H])

        if k > 0:
            with torch.no_grad():
                roll = tok_ext.clone()
                ar = ar_rng.random() < args.ar_prob
                for idx in range(K_MAX + H + 1 - k, K_MAX + H +1):
                    if ar:  # each token from the model's own earlier tokens of the frame, as at inference
                        roll[:, idx] = model.generate_last_frame(roll[:, idx-H:idx+1], cond_ext[:, idx-H:idx+1], *cache_at(idx))
                    else:   # one pass: each token sees the TRUE earlier tokens of its frame (cheaper, cleaner)
                        pred = model.predict_frame_tokens(roll[:, idx-H:idx+1],cond_ext[:, idx-H:idx+1], *cache_at(idx))
                        roll[:,idx] = pred[:,-1,:]
            input_tokens = roll[:,K_MAX:]
        else:
            input_tokens = tok_ext[:, K_MAX:]

        loss = model.compute_loss(input_tokens, cond_ext[:,K_MAX:], *cache_at(K_MAX + H), target_tokens=tok_ext[:, K_MAX:], last_step_only=bool(args.cache_size))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            extra = {}
            if args.cache_size:
                # measured on the rows whose target has a useful snapshot only:
                # gain = loss(no cache) - loss(cache)       -> does it use the cache at all
                # loc  = loss(shuffled offsets) - loss(cache) -> does it use *where* each snapshot is
                u = useful
                if u.any():
                    ct_u, cc_u, cm_u = ctok[u, -1], ccond[u, -1], cmask[u, -1]
                    # permute offsets among each row's valid entries (valid ones are a prefix)
                    score = torch.rand(cm_u.shape, device=cm_u.device)
                    score[~cm_u] = 2 + torch.arange(cm_u.shape[1], device=cm_u.device).expand_as(cm_u)[~cm_u].float()
                    cc_shuf = torch.gather(cc_u, 1, score.argsort(1)[..., None].expand_as(cc_u))
                    with torch.no_grad():
                        loss_u = lambda *c: model.compute_loss(input_tokens[u], cond_ext[u, K_MAX:], *c,
                                                               target_tokens=tok_ext[u, K_MAX:], last_step_only=True)
                        l_on, l_off, l_shuf = loss_u(ct_u, cc_u, cm_u), loss_u(), loss_u(ct_u, cc_shuf, cm_u)
                    extra["gain"] = f"{(l_off - l_on).item():+.3f}"
                    extra["loc"] = f"{(l_shuf - l_on).item():+.3f}"
                    extra["useful"] = f"{u.float().mean().item():.2f}"
            pbar.set_postfix(loss=f"{loss.item():.4f}", k=f"{k}/{k_cap}", **extra)

        if step % args.ckpt_every == 0 or step == args.max_steps:
            blob = {"step": step, "model": model.state_dict(),
                    "optim": opt.state_dict(), "args": vars(args)}
            torch.save(blob, os.path.join(ckpt_dir, f"step_{step:06d}.pt"))
            torch.save(blob, os.path.join(ckpt_dir, "latest.pt"))

        if step % args.viz_every == 0 or step == args.max_steps:
            save_viz(step, loss.item())

if __name__ == "__main__":
    main()
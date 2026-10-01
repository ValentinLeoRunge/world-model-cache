import argparse
import glob
import os

import numpy as np
import torch
from tqdm import tqdm

from src.models.tokenizer import Tokenizer


@torch.no_grad()
def tokenize_episode(model, image, batch_size, device):
    """image: (T, 64, 64, 3) uint8  ->  (T, 16) uint8 code indices."""
    out = []
    for i in range(0, len(image), batch_size):
        chunk = image[i:i + batch_size]
        x = torch.from_numpy(chunk).to(device).float().div_(255.0)
        x = x.permute(0, 3, 1, 2).contiguous()            # (b,3,64,64)
        out.append(model.tokenize(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/tokenizer/latest.pt")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_root", default="data/tokens")
    p.add_argument("--splits", default="train0,eval")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--frame_stride", type=int, default=1,
                   help="1 = tokenize every frame (default); >1 also strides the copied fields")
    p.add_argument("--limit", type=int, default=None, help="cap #episodes per split (smoke test)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = Tokenizer().to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt} (step {ckpt.get('step', '?')}) on {args.device}")

    written = skipped = 0
    for split in args.splits.split(","):
        paths = sorted(glob.glob(os.path.join(args.data_root, split, "*.npz")))
        if args.limit is not None:
            paths = paths[:args.limit]
        out_dir = os.path.join(args.out_root, split)
        os.makedirs(out_dir, exist_ok=True)

        for path in tqdm(paths, desc=split):
            out_path = os.path.join(out_dir, os.path.basename(path))
            if os.path.exists(out_path) and not args.overwrite:
                skipped += 1
                continue

            with np.load(path) as d:
                s = args.frame_stride
                image = d["image"][::s]
                t = len(image)
                fields = {}
                for k in d.files:
                    if k == "image":
                        continue
                    arr = d[k]
                    # keep per-frame fields aligned with the strided frames
                    if arr.ndim >= 1 and arr.shape[0] == len(d["image"]):
                        arr = arr[::s]
                    fields[k] = arr

            tokens = tokenize_episode(model, image, args.batch_size, args.device)
            assert tokens.shape == (t, 16) and tokens.dtype == np.uint8

            np.savez_compressed(out_path, tokens=tokens, **fields)
            written += 1

    print(f"done: {written} written, {skipped} skipped -> {args.out_root}")


if __name__ == "__main__":
    main()

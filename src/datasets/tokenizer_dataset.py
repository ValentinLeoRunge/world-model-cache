"""Streaming frame loader for the Memory-Maze .npz episodes.

Why streaming (IterableDataset) and not a plain indexable Dataset:
  ~2900 episodes x ~1000 frames each. Holding every frame in RAM is ~35 GB, and
  reading one arbitrary frame means `np.load`-ing (decompressing) a whole episode
  just to throw 999 frames away. So we stream episodes and mix their frames.

The shuffle buffer: reading episode-by-episode would put ~250 near-identical
frames (agent barely moved) next to each other in a batch -- a useless gradient
signal. Instead each worker keeps `shuffle_buffer` episodes decompressed at once
and draws frames from them at random, so one batch spans many different mazes.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class TokenizerDataset(IterableDataset):
    """Yields single frames as float32 tensors (3, 64, 64) in [0, 1]."""

    def __init__(self, root, split="train0", frame_stride=4,
                 shuffle_buffer=8, episodes_limit=None, seed=0):
        super().__init__()
        self.paths = sorted(glob.glob(os.path.join(root, split, "*.npz")))
        if episodes_limit is not None:
            self.paths = self.paths[:episodes_limit]
        if not self.paths:
            raise FileNotFoundError(f"no .npz under {os.path.join(root, split)}")
        self.frame_stride = frame_stride
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _load_episode(self, path):
        """np.load once, decompress `image`, keep every Nth frame.

        Returns uint8 array (T, 3, 64, 64), channel-first.
        """
        with np.load(path) as d:
            img = d["image"][:: self.frame_stride]        # (T, 64, 64, 3) uint8
        return np.ascontiguousarray(img.transpose(0, 3, 1, 2))

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1

        # shard episodes across workers: no file is read by two workers
        paths = list(self.paths[wid::nworkers])
        # torch.initial_seed() varies per worker AND per epoch, so the order
        # reshuffles every epoch while staying reproducible under a fixed seed.
        rng = random.Random(torch.initial_seed() if info else self.seed)
        rng.shuffle(paths)
        src = iter(paths)

        # each slot: [frames (T,3,64,64) uint8, shuffled frame order, cursor]
        slots = []

        def refill():
            while len(slots) < self.shuffle_buffer:
                try:
                    p = next(src)
                except StopIteration:
                    return
                frames = self._load_episode(p)
                order = list(range(len(frames)))
                rng.shuffle(order)
                slots.append([frames, order, 0])

        refill()
        while slots:
            s = rng.randrange(len(slots))
            frames, order, cur = slots[s]
            frame = frames[order[cur]]                     # (3, 64, 64) uint8
            slots[s][2] += 1
            if slots[s][2] >= len(order):                  # episode exhausted
                slots.pop(s)
                refill()
            yield torch.from_numpy(frame).float().div_(255.0)

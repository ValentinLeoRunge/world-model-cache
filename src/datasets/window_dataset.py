"""Streaming (H+1)-step window loader for the world model, over data/tokens/.

Same episode-level shuffle-buffer idea as episode_dataset.py (see its docstring
for why streaming + mixing at all), but each item is now a fixed-length window
of consecutive steps -- tokens + actions -- instead of a single frame. That's
what the world model actually trains on: one window = one training example.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class WindowDataset(IterableDataset):
    """Yields (tokens, actions): (window,16) int64, (window,action_dim) float32."""

    def __init__(self, root, split="train0", window=17, shuffle_buffer=8,
                 episodes_limit=None, seed=0):
        super().__init__()
        self.paths = sorted(glob.glob(os.path.join(root, split, "*.npz")))
        if episodes_limit is not None:
            self.paths = self.paths[:episodes_limit]
        if not self.paths:
            raise FileNotFoundError(f"no .npz under {os.path.join(root, split)}")
        self.window = window
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _load_episode(self, path):
        """-> tokens uint8 (T,16), actions float32 (T,action_dim), reset bool (T,)."""
        with np.load(path) as d:
            tokens = d["tokens"]
            actions = d["action"].astype(np.float32)
            reset = d["reset"]
        return tokens, actions, reset

    def _valid_starts(self, reset, t_total, rng):
        """Start indices t where [t, t+window) has no reset after its first step."""
        starts = [
            t for t in range(t_total - self.window + 1)
            if not reset[t + 1: t + self.window].any()
        ]
        rng.shuffle(starts)
        return starts

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1

        paths = list(self.paths[wid::nworkers])
        # torch.initial_seed() varies per worker AND per epoch -- see episode_dataset.py
        rng = random.Random(torch.initial_seed() if info else self.seed)
        rng.shuffle(paths)
        src = iter(paths)

        # each slot: [tokens, actions, shuffled valid starts, cursor]
        slots = []

        def refill():
            while len(slots) < self.shuffle_buffer:
                try:
                    p = next(src)
                except StopIteration:
                    return
                tokens, actions, reset = self._load_episode(p)
                starts = self._valid_starts(reset, len(tokens), rng)
                if starts:  # episode too short for one window: skip it
                    slots.append([tokens, actions, starts, 0])

        refill()
        while slots:
            s = rng.randrange(len(slots))
            tokens, actions, starts, cur = slots[s]
            t = starts[cur]
            slots[s][3] += 1
            if slots[s][3] >= len(starts):                 # exhausted this episode
                slots.pop(s)
                refill()

            tok_win = torch.from_numpy(tokens[t: t + self.window].astype(np.int64))
            act_win = torch.from_numpy(actions[t: t + self.window])
            yield tok_win, act_win

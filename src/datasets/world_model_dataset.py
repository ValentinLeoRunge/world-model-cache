import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from src.cache.cache import Cache

class WorldModelDataset(IterableDataset):
    def __init__(self, root, split="train0", window=17, shuffle_buffer=8,
                 episodes_limit=None, seed=0, cache_size=0, history=16, p_other=1.0):
        super().__init__()
        self.paths = sorted(p for s in split.split(",") for p in glob.glob(os.path.join(root, s, "*.npz")))
        if episodes_limit is not None:
            self.paths = self.paths[:episodes_limit]
        if not self.paths:
            raise FileNotFoundError(f"no .npz under {os.path.join(root, split)}")
        self.window = window
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.cache_size = cache_size
        self.history = history
        self.p_other = p_other

    def _load_episode(self, path):
        with np.load(path) as d:
            tokens = d["tokens"]
            actions = d["action"].astype(np.float32)
            pos = d["agent_pos"].astype(np.float32)
            dir_ = d["agent_dir"].astype(np.float32)
            reset = d["reset"]
        return tokens, actions, pos, dir_, reset

    def _valid_starts(self, reset, t_total, rng):
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
                tokens, actions, pos, dir_, reset = self._load_episode(p)
                starts = self._valid_starts(reset, len(tokens), rng)
                if starts:  # episode too short for one window: skip it
                    stores = self._cache_stores(tokens, pos, dir_) if self.cache_size else None
                    useful = None
                    if stores is not None:
                        useful = [self._useful(stores[t], pos[t], dir_[t]) for t in range(len(tokens))]
                        starts = [t for t in starts
                                  if useful[t + self.window - 1] or rng.random() < self.p_other]
                    if starts:
                        slots.append([tokens, actions, pos, dir_, stores, starts, 0, useful])

        refill()
        viewer = Cache(self.cache_size)
        while slots:
            s = rng.randrange(len(slots))
            tokens, actions, pos, dir_, stores, starts, cur, useful = slots[s]
            t = starts[cur]
            slots[s][6] += 1
            if slots[s][6] >= len(starts):                 # exhausted this episode
                slots.pop(s)
                refill()

            tok_win = torch.from_numpy(tokens[t: t + self.window].astype(np.int64))
            act_win = torch.from_numpy(actions[t: t + self.window])
            if stores is None:
                yield tok_win, act_win
            else:
                # one view per frame that can be predicted: window[history:], i.e. the target
                # plus every frame the rollout may generate, each against its own bank and pose
                views = []
                for t_pred in range(t + self.history, t + self.window):
                    viewer.store = stores[t_pred]
                    views.append(viewer.view(pos[t_pred], dir_[t_pred]))
                yield (tok_win, act_win, *(torch.stack(v) for v in zip(*views)),
                       torch.tensor(useful[t + self.window - 1]))

    @staticmethod
    def _useful(store, p, d):
        return any(np.linalg.norm(e[1] - p) < 0.5 and e[2] @ d > 0.7 for e in store)

    def _cache_stores(self, tokens, pos, dir_):
        cache = Cache(self.cache_size)
        stores = []
        for t in range(len(tokens)):
            if t >= self.history:
                w = slice(t - self.history, t+1)
                cache.update(tokens[w], pos[w], dir_[w])
            stores.append(list(cache.store))
        return stores
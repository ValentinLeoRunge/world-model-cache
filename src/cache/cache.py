import numpy as np
import torch

from src.models.world_model import ACTION_DIM, COND_DIM, egocentric_offset


class Cache:
    def __init__(self, size: int):
        self.size = size
        self.store = []  # (tokens (16,), pos (2,), dir (2,)), oldest first

    # simple update method
    def update(self, tokens, pos, dir_, r_ins=0.5):
        if self.store:
            held = np.stack([e[1] for e in self.store])
            if np.linalg.norm(held - pos[0], axis=1).min() < r_ins:
                    return
        if len(self.store) == self.size:
            self.store.pop(0)
        self.store.append((tokens[0], pos[0], dir_[0]))

    def view(self, cur_pos, cur_dir):
        tokens = torch.zeros(self.size, 16, dtype=torch.long)
        cond = torch.zeros(self.size, COND_DIM)
        mask = torch.zeros(self.size, dtype=torch.bool)

        n = len(self.store)
        if n == 0:
            return tokens, cond, mask

        tokens[:n] = torch.from_numpy(np.stack([e[0] for e in self.store]).astype(np.int64))
        cond[:n, ACTION_DIM:] = egocentric_offset(
            torch.from_numpy(np.stack([e[1] for e in self.store])).float().unsqueeze(0),
            torch.from_numpy(np.stack([e[2] for e in self.store])).float().unsqueeze(0),
            torch.as_tensor(cur_pos).float().unsqueeze(0),
            torch.as_tensor(cur_dir).float().unsqueeze(0),
        )[0]
        mask[:n] = True
        return tokens, cond, mask

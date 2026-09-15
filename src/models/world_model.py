"""GPT-style world model: predicts the next frame's 16 tokens given the last
H steps of actions + frame tokens. See blueprints/WORLDMODEL.md.

Sequence layout per step: [action, tok_1, ..., tok_16] -- 17 "slots". A window
of H+1 steps flattens to one causal sequence of length L = 17*(H+1). Plain
next-token prediction over this flat sequence gives both effects for free:
predicting tok_i (i>1) from tok_1..tok_{i-1} of the same frame *is* "one token
at a time"; predicting tok_1 of a frame from pure H-step history *is* the
world-model step itself.

This file only covers training-shape forward + loss (fixed window length).
Autoregressive generation (needed for eval/notebook) reuses this same forward
by feeding a full-length window with placeholder tokens for the not-yet-known
part of the last frame -- no separate generation path needed here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SLOTS_PER_STEP = 17  # 1 action + 16 frame tokens


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, l, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        hd = d // self.n_head
        q = q.view(b, l, self.n_head, hd).transpose(1, 2)
        k = k.view(b, l, self.n_head, hd).transpose(1, 2)
        v = v.view(b, l, self.n_head, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (b,h,l,hd)
        out = out.transpose(1, 2).reshape(b, l, d)
        return self.proj(out)


class Block(nn.Module):
    """Pre-norm GPT block: x = x + Attn(LN(x)); x = x + MLP(LN(x))."""

    def __init__(self, d_model, n_head, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class WorldModel(nn.Module):
    def __init__(self, history=16, vocab_size=256, action_dim=6,
                 d_model=256, n_layer=4, n_head=4, mlp_ratio=4):
        super().__init__()
        self.history = history               # H: steps of past context
        self.window = history + 1            # H+1: steps in one window
        self.vocab_size = vocab_size
        self.seq_len = SLOTS_PER_STEP * self.window  # L

        self.tok_embed = nn.Embedding(vocab_size, d_model)     # frame tokens: lookup
        self.action_embed = nn.Linear(action_dim, d_model)     # actions: linear
        self.slot_embed = nn.Embedding(SLOTS_PER_STEP, d_model)   # which of 17 slots
        self.step_embed = nn.Embedding(self.window, d_model)      # which of H+1 steps

        slot_ids = torch.arange(SLOTS_PER_STEP).repeat(self.window)             # (L,)
        step_ids = torch.arange(self.window).repeat_interleave(SLOTS_PER_STEP)  # (L,)
        self.register_buffer("slot_ids", slot_ids)
        self.register_buffer("step_ids", step_ids)
        self.register_buffer("frame_slot_mask", slot_ids != 0)  # True at frame-token slots

        self.blocks = nn.ModuleList(
            Block(d_model, n_head, mlp_ratio) for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _interleave(self, action_item, token_items):
        """(B,W,1[,...]) + (B,W,16[,...]) along dim=2 -> flattened (B,L[,...]).

        Used both for embeddings (trailing d_model dim) and for raw ids (no
        trailing dim) -- same slot order as slot_ids/step_ids above.
        """
        b = token_items.shape[0]
        flat = torch.cat([action_item, token_items], dim=2)     # (B,W,17,...)
        return flat.reshape(b, self.seq_len, *flat.shape[3:])

    def embed(self, tokens, actions):
        """tokens (B,W,16) long in [0,vocab), actions (B,W,action_dim) float
        -> (B,L,d_model)."""
        tok_content = self.tok_embed(tokens)                    # (B,W,16,d)
        act_content = self.action_embed(actions).unsqueeze(2)   # (B,W,1,d)
        content = self._interleave(act_content, tok_content)    # (B,L,d)
        return content + self.slot_embed(self.slot_ids) + self.step_embed(self.step_ids)

    def forward(self, tokens, actions):
        """-> logits (B, L, vocab_size), one prediction-of-the-next-item per position."""
        x = self.embed(tokens, actions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def _targets_and_mask(self, tokens):
        """(B,W,16) -> target_ids (B,L) true content id at every position
        (action slots hold an unused placeholder), mask (B,L) True at
        frame-token slots. Shared by compute_loss and predict_targets."""
        b, w = tokens.shape[0], tokens.shape[1]
        dummy = torch.zeros(b, w, 1, dtype=tokens.dtype, device=tokens.device)
        target_ids = self._interleave(dummy, tokens)
        mask = self.frame_slot_mask.unsqueeze(0).expand(b, -1)
        return target_ids, mask

    def compute_loss(self, tokens, actions, target_tokens=None):
        """Masked next-token cross-entropy over frame-token positions only.
        `tokens` is what the model actually sees as input; the target
        defaults to the same tensor (pure teacher-forcing) but can be given
        separately so a scheduled-sampling-noised `tokens` is still scored
        against the true `target_tokens` (see train_world_model.py and
        predict_frame_tokens below)."""
        logits = self.forward(tokens, actions)                  # (B,L,V)
        target_ids, mask = self._targets_and_mask(
            tokens if target_tokens is None else target_tokens
        )

        pred = logits[:, :-1].reshape(-1, self.vocab_size)
        tgt = target_ids[:, 1:].reshape(-1)
        m = mask[:, 1:].reshape(-1)
        return F.cross_entropy(pred[m], tgt[m])

    @torch.no_grad()
    def predict_frame_tokens(self, tokens, actions):
        """The model's own one-step-ahead argmax prediction at every
        frame-token slot, given the true window as context throughout
        (teacher-forced, one forward pass -- NOT multi-step generation, see
        generate_last_frame for that). Used to build scheduled-sampling
        inputs: mix these self-predictions into the true tokens before
        training, so the model learns to tolerate context it got slightly
        wrong instead of only ever seeing perfect ground truth (exposure
        bias -- see train_world_model.py). -> (B,W,16), same layout as
        `tokens`.
        """
        logits = self.forward(tokens, actions)                  # (B,L,V)
        pred_flat = logits[:, :-1].argmax(-1)                   # (B,L-1): predicts position i+1's content
        pred_at_frame_slots = pred_flat[:, self.frame_slot_mask[1:]]   # (B,W*16), same order as `tokens`
        return pred_at_frame_slots.view(tokens.shape[0], self.window, 16)

    @torch.no_grad()
    def predict_targets(self, tokens, actions):
        """Teacher-forced argmax predictions at every frame-token position
        (uses true tokens as context throughout, including within the target
        frame -- for a fast accuracy sanity number, not real generation; see
        generate_last_frame for that). -> pred, target, mask, all (B, L-1),
        aligned to the shifted target positions."""
        logits = self.forward(tokens, actions)
        target_ids, mask = self._targets_and_mask(tokens)
        return logits[:, :-1].argmax(-1), target_ids[:, 1:], mask[:, 1:]

    @torch.no_grad()
    def generate_last_frame(self, tokens, actions):
        """tokens (B,W,16): last step's 16 entries are ignored (will be
        overwritten). actions (B,W,action_dim): true actions, including the
        new step's. -> (B,16) predicted token ids for the last step, filled in
        one at a time -- causal masking means nothing later than the slot
        being read has been used yet, so the ignored placeholders never leak
        in; each iteration only reads positions already filled with real data.
        """
        tokens = tokens.clone()
        base = (self.window - 1) * SLOTS_PER_STEP  # last step's action-slot position
        for i in range(16):
            logits = self.forward(tokens, actions)
            tokens[:, -1, i] = logits[:, base + i, :].argmax(-1)
        return tokens[:, -1, :]

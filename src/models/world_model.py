import torch
import torch.nn as nn
import torch.nn.functional as F

SLOTS_PER_STEP = 17  # 1 conditioning slot + 16 frame tokens
ACTION_DIM = 6       # live rows: one-hot action (all-zero on an episode's first frame)
OFFSET_DIM = 4       # cache rows: [fwd, rgt, cos dtheta, sin dtheta]
COND_DIM = ACTION_DIM + OFFSET_DIM


def egocentric_offset(cache_pos, cache_dir, cur_pos, cur_dir):
    d = cache_pos - cur_pos.unsqueeze(1)                              # (B,C,2)
    f = cur_dir.unsqueeze(1)                                          # (B,1,2)
    fwd = (d * f).sum(-1)                                             # ahead(+) / behind(-)
    rgt = d[..., 0] * f[..., 1] - d[..., 1] * f[..., 0]               # right(+) / left(-)
    cos = (cache_dir * f).sum(-1)                                     # heading alignment
    sin = f[..., 0] * cache_dir[..., 1] - f[..., 1] * cache_dir[..., 0]  # ...and its handedness
    return torch.stack([fwd, rgt, cos, sin], dim=-1)

# previous frames change context for frames afterwards
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)  # split the input into query, key, value
        self.proj = nn.Linear(d_model, d_model)     # produces one output

    def forward(self, x):
        b, l, d = x.shape   # b number of examples in a batch, l length per batch, d dimension of each element in a batch
        q, k, v = self.qkv(x).split(d, dim=2)   # 1 linear projection expands the input to 3x its size, then splits it into three separate tensors (queries, keys, values) for attention

        hd = d // self.n_head
        q = q.view(b, l, self.n_head, hd).transpose(1, 2)
        k = k.view(b, l, self.n_head, hd).transpose(1, 2)
        v = v.view(b, l, self.n_head, hd).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1,2).reshape(b, l, d)

        return self.proj(out)

# combine current frame and cached frame context
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2*d_model)
        self.proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x_live, x_cache, cache_mask=None):
        if x_cache is None:
            return torch.zeros_like(x_live)
        b, l, d = x_live.shape
        q = self.q(x_live).view(b, l, self.n_head, d // self.n_head).transpose(1, 2)
        k, v = self.kv(x_cache).split(d, dim=2)
        k = k.view(b, -1, self.n_head, d // self.n_head).transpose(1,2)
        v = v.view(b, -1, self.n_head, d // self.n_head).transpose(1, 2)
        attn_mask = None
        if cache_mask is not None:
            # one bool per cache entry -> one per sequence position (17 slots each)
            attn_mask = cache_mask.repeat_interleave(SLOTS_PER_STEP, dim=1)[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, l, d)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attention = CrossAttention(d_model, n_head)
        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x_live, x_cache, cache_mask=None):
        x_live = x_live + self.attn(self.ln1(x_live))
        x_live = x_live + self.cross_attention(self.ln2(x_live), x_cache, cache_mask)
        x_live = x_live + self.mlp(self.ln3(x_live))
        return x_live

class WorldModel(nn.Module):
    def __init__(self, history, cache_size, vocab_size, d_model, n_layer=4, n_head=4, mlp_ratio=4):
        super().__init__()

        self.history = history          # H: steps of past context the model uses for predictions
        self.window = history + 1       # H+1: live steps (history + the predicted one)
        self.seq_len = SLOTS_PER_STEP * self.window  # L: length of one sequence

        self.cache_size = cache_size    # cache and window are treated as seperate streams by the model

        self.vocab_size = vocab_size    # number of vectors in our lookup table
        self.d_model = d_model          # dimensions of vectors in the lookup table

        self.tok_embed = nn.Embedding(vocab_size, d_model)        # lookup table
        self.action_embed = nn.Linear(ACTION_DIM, d_model)   # slot 0 of a live row
        self.offset_embed = nn.Linear(OFFSET_DIM, d_model)   # slot 0 of a cache row
        self.slot_embed = nn.Embedding(SLOTS_PER_STEP, d_model)   # which of the 17 positions within a step are we at
        self.step_embed = nn.Embedding(self.window, d_model)      # what step are we in

        slot_ids = torch.arange(SLOTS_PER_STEP).repeat(self.window)
        step_ids = torch.arange(self.window).repeat_interleave(SLOTS_PER_STEP)
        self.register_buffer("slot_ids", slot_ids)
        self.register_buffer("step_ids", step_ids)
        self.register_buffer("frame_slot_mask", slot_ids != 0)

        self.n_layer = n_layer
        self.n_head = n_head
        self.mlp_ratio = mlp_ratio

        self.blocks = nn.ModuleList([
            Block(d_model, n_head, mlp_ratio) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _interleave(self, cond_item, token_items):
        b, n = token_items.shape[0], token_items.shape[1]
        flat = torch.cat([cond_item, token_items], dim=2)
        return flat.reshape(b, n * SLOTS_PER_STEP, *flat.shape[3:])

    def _targets_and_mask(self, tokens):
        b, w = tokens.shape[0], tokens.shape[1]
        dummy = torch.zeros(b, w, 1, dtype=tokens.dtype, device=tokens.device)
        target_ids = self._interleave(dummy, tokens)            # 0 | 101 102 ... 116 | 0 | 201 202 ...216 | 0 | ...
        mask = self.frame_slot_mask.unsqueeze(0).expand(b, -1)  # F | T T T ... | F | T ...
        return target_ids, mask

    def embed(self, tokens, cond, is_cache=False):
        tok_content = self.tok_embed(tokens)                        # (B,n,16,d)
        if is_cache:
            offset = cond[:, :, ACTION_DIM:]                        # (B,n,OFFSET_DIM)
            cond_content = F.gelu(self.offset_embed(offset)).unsqueeze(2)   # (B,n,1,d)
            tok_content = tok_content + cond_content                # (B,n,16,d)
            content = self._interleave(cond_content, tok_content)
            n = tokens.shape[1]
            return content + self.slot_embed(self.slot_ids[:SLOTS_PER_STEP].repeat(n))

        action = cond[:, :, :ACTION_DIM]                            # (B,n,ACTION_DIM)
        cond_content = self.action_embed(action).unsqueeze(2)       # (B,n,1,d)
        content = self._interleave(cond_content, tok_content)       # (B,L,d)
        return content + self.slot_embed(self.slot_ids) + self.step_embed(self.step_ids)

    def forward(self, tokens, cond, cache_tokens=None, cache_cond=None, cache_mask=None):
        # tokens (B, H+1, 16), cond (B, H+1, COND_DIM), live only
        # cache_tokens (B, C, 16), cache_cond (B, C, COND_DIM), cache only (optional)
        live_x = self.embed(tokens, cond, is_cache=False)

        if cache_tokens is not None:
            cache_x = self.embed(cache_tokens, cache_cond, is_cache=True)
        else:
            cache_x = None

        for block in self.blocks:
            live_x = block(live_x, cache_x, cache_mask)

        live_x = self.ln_f(live_x)
        logits = self.head(live_x)

        return logits

    @torch.no_grad()
    def predict_frame_tokens(self, tokens, cond, cache_tokens=None, cache_cond=None, cache_mask=None):
        logits = self.forward(tokens, cond, cache_tokens, cache_cond, cache_mask)
        pred_flat = logits[:,:-1].argmax(-1)
        pred_at_frame_slots = pred_flat[:, self.frame_slot_mask[1:]]
        return pred_at_frame_slots.view(tokens.shape[0], self.window, 16)

    @torch.no_grad()
    def predict_targets(self, tokens, cond, cache_tokens=None, cache_cond=None, cache_mask=None):
        logits = self.forward(tokens, cond, cache_tokens, cache_cond, cache_mask)
        target_ids, mask = self._targets_and_mask(tokens)
        return logits[:, :-1].argmax(-1), target_ids[:, 1:], mask[:, 1:]

    def compute_loss(self, tokens, cond, cache_tokens=None, cache_cond=None,
                     cache_mask=None, target_tokens=None, last_step_only=False):
        logits = self.forward(tokens, cond, cache_tokens, cache_cond, cache_mask)
        target_ids, mask = self._targets_and_mask(
            tokens if target_tokens is None else target_tokens
        )
        if last_step_only:
            mask = mask & (self.step_ids == self.window - 1)
        pred = logits[:, :-1].reshape(-1, self.vocab_size)
        tgt = target_ids[:,1:].reshape(-1)
        m = mask[:, 1:].reshape(-1)
        return F.cross_entropy(pred[m], tgt[m])

    @torch.no_grad
    def generate_last_frame(self, tokens, cond, cache_tokens=None, cache_cond=None, cache_mask=None):
        tokens = tokens.clone()
        base = (self.window - 1) * SLOTS_PER_STEP
        for i in range(16):
            logits = self.forward(tokens, cond, cache_tokens, cache_cond, cache_mask)
            tokens[:, -1, i] = logits[:, base + i, :].argmax(-1)
        return tokens[:, -1, :]
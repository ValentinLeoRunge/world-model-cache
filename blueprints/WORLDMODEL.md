# World Model (GPT-style transformer)

## Goal
Given the last `H` steps (action + 16 frame tokens each), predict the next frame's 16 tokens, one by one. This model deliberately has **no memory beyond `H` steps**. Phase 4 (the cache) is what fixes it. This phase exists to *demonstrate the failure*: prediction accuracy at a revisited place collapses once the gap since last seeing it, `N`, exceeds `H`. No retrieval, no bank, yet.

Operates entirely on `data/tokens/{split}/*.npz` (already frozen, from TOKENIZER.md). The trained VQ-VAE is only touched again to `decode_indices()` predicted tokens back to pixels for the visualization grids, training itself never sees a raw image.

## Sequence representation

One environment step = 1 action (6-dim one-hot, all-zero on the very first frame of an episode) + 16 frame tokens (the tokenizer's 4×4 grid, row-major). A training window of `H+1` consecutive steps is flattened into one sequence:

```
[ a_1, f1_1..f1_16,  a_2, f2_1..f2_16,  ...,  a_{H+1}, f{H+1}_1..f{H+1}_16 ]
```

17 positions per step ("slots": slot 0 = action, slots 1-16 = grid cells) × `(H+1)` steps = `L = 17(H+1)` positions. A single **causal** transformer runs over this flat sequence with one rule, plain next-token prediction, applied at every position — no separate cross-frame mechanism needed. Predicting `ft_i` means conditioning on everything before it: past steps, this step's action, and this frame's already-decoded tokens `1..i-1`. That last part is only non-empty when `i>1`; at `i=1` there's no same-frame context yet, so "everything before it" is purely the `H`-step history. Same rule throughout — `i>1` gives "predict the 16 tokens one by one" (ACTION_PLAN.md's phrasing), `i=1` gives the actual world-model prediction (next frame from history), and it's the same causal step in both cases, just at a slot where the same-frame condition happens to be empty.

> **Why autoregressive within a frame, not all 16 at once:** the 16 tokens are not independent, a wall's color has to agree across the cells it spans. Predicting them left-to-right lets token 5 condition on tokens 1–4 already chosen for the same frame, instead of assuming they're independent given the context.

## Architecture

### Input
One window: `tokens (H+1, 16)` int64 in `[0,256)` (on disk this is `uint8`; cast up to `int64` on load since `nn.Embedding`/`nn.CrossEntropyLoss` require it), `actions (H+1, 6)` float32 (one-hot or all-zero; float because it is matrix-multiplied by a `Linear` layer, not looked up). Each of the `17(H+1)` items (1 action + 16 tokens, per step) is embedded to its own `d_model` vector *first*; only then are those vectors stacked into one length-`L = 17(H+1)` sequence (see Embedding below)

### Embedding
- **Frame tokens** → `nn.Embedding(256, d_model)`, looked up by index.
- **Actions** → `nn.Linear(6, d_model)`, applied to the raw vector directly (handles the all-zero first-frame action for free, no extra "no-op" class needed).
- **Slot embedding** `nn.Embedding(17, d_model)` — which of the 17 positions-within-a-step (shared across steps).
- **Step embedding** `nn.Embedding(H+1, d_model)` — which of the `H+1` steps in the window.
- Combined: `token_or_action_embed + slot_embed + step_embed` → `(L, d_model)`.

### Transformer
Standard pre-norm GPT block × `n_layer`: causal self-attention (`n_head`) + MLP (`GELU`, `4×d_model` hidden), residual around each, final `LayerNorm`. Causal mask = lower-triangular over the full length-`L` sequence.

### Output head
`Linear(d_model, 256)` applied at every position → logits `(L, 256)`.

### Loss
Standard shifted next-token cross-entropy, `input = seq[:-1]`, `target = seq[1:]`, but **masked to frame-token positions only** (`target`'s slot ≠ 0). We don't ask the model to predict actions; they're observed, not generated.

> **Why this guarantees the failure we're trying to show:** attention cannot reach a position outside the window. If a location was last seen `N > H` steps ago, the frame tokens from that visit are structurally absent from the context, not "hard to recall,".

## Training

**Data**: `data/tokens/train0` (2900 episodes, `(1001,16)` tokens + `(1001,6)` actions each). Windowed `IterableDataset`, same shuffle-buffer idea as the tokenizer's loader:
- Shard episodes across workers, keep several "in flight," draw a random window start `t ∈ [0, 1001-(H+1))` per sample.
- **Frame stride = 1** (every real step, unlike the tokenizer's stride-4), the model has to learn the true action → next-frame transition, and skipping frames would mean one action no longer explains the resulting frame.
- Reject a window if it contains a `reset=True` after its first position (two different mazes stitched into one "continuous" window). Not observed in the data inspected (`reset` is only `True` at each episode's frame 0), but cheap to guard.

**Loss**: masked cross-entropy on frame-token positions (above).

**Optimizer**: Adam, lr=3e-4, batch size 64 windows. (Keep it plain; add warmup/cosine decay only if the loss curve is unstable)

**Checkpointing**: every 1000 steps → `checkpoints/worldmodel/step_{step:06d}.pt`, overwrite `latest.pt`.

**Visualization**: every 1000 steps, take a fixed held-out window, autoregressively predict the last frame's 16 tokens, `decode_indices()` them through the **frozen tokenizer** (`checkpoints/tokenizer/latest.pt`), and save a real-vs-predicted grid, same format as the tokenizer's, one more row.

**Budget**: ~30,000 steps (soft; this model is ~3.3M params vs. the tokenizer's ~24M, watch the loss curve rather than pre-committing).

**Exit criterion**: this phase isn't "make the model good", it's "make the failure visible." Loss plateaued, next-token accuracy clearly higher for `N ≤ H` than `N > H` in the evaluation below, both numbers stable across two held-out samples.

## Evaluation

The actual deliverable of this phase is the **accuracy-vs-`N` plot**, not a single accuracy number.

**Revisit detection** (`src/utils/revisits.py`, using `agent_pos` copied through into `data/tokens/`): for each frame `t2`, find the most recent `t1 < t2` with `dist(agent_pos[t1], agent_pos[t2]) < threshold` such that the agent was *away* from that spot for everything in between (`dist(agent_pos[t], agent_pos[t2]) ≥ threshold` for all `t1 < t < t2`). That pair is a revisit event with gap `N = t2 - t1`. Matches ACTION_PLAN.md step 1 ("find all revisit events, same place, N steps apart") — this is that piece, written now because it's needed to evaluate, not to train.

**Then**: for a sample of eval episodes, at every revisit event's `t2`, run the model on context ending at `t2-1`, predict `t2`'s 16 tokens, score exact-match accuracy against the real tokens. Bucket by `N` (e.g. `≤H`, `H+1..2H`, `2H+1..4H`, `>4H`) and report mean accuracy per bucket + a plot. Also report plain overall next-token accuracy on random (non-revisit) windows as a training-health sanity number, the way PSNR served the tokenizer.

Standalone script, same shape as `eval_tokenizer.py`: loads a checkpoint, runs deterministically over a fixed eval sample, saves the bucketed-accuracy numbers/plot and a few real-vs-predicted grids.

## Qualitative Path Inspection (notebook)

The accuracy-vs-`N` plot is the quantitative deliverable; this is the human-readable companion, walk one real path and *look* at where it breaks. `notebooks/inspect_world_model.ipynb` (repo root, next to `explore_data.ipynb`, human-driven inspection, not part of the `src/` pipeline).

Pick one real eval episode via `revisits.py`, choosing a revisit event whose gap `N` clearly straddles `H`(so the failure has a chance to show), and render a strip of columns (one per timestep) spanning from before the agent leaves the revisited spot through its return, **four rows**:

1. **Original** — the raw frame, straight from `data/{split}/*.npz`'s `image` field.
2. **Tokenizer reconstruction** — `decode_indices()` on the *true* recorded tokens, no world model involved. This is the ceiling: how blurry frames already are from the VQ-VAE bottleneck alone. Without this row you can't separate "the world model is wrong" from "the tokenizer was always this blurry."
3. **World model, teacher-forced** — at each shown step, feed the model the *true* recorded last-`H` steps and predict that one frame. Errors don't compound between columns; this shows raw one-step prediction quality.
4. **World model, autoregressive rollout** — seed with the true first `H` steps, then feed the model's *own* predicted tokens back in as history for every subsequent step (actions stay ground-truth throughout, they're recorded/given, never generated). This is the row that can actually show the failure: once the rollout has run past `H` self-generated steps, no real information about a revisited room survives anywhere in the context, so its prediction there is a plausible-looking guess, not a memory, the same "dementia" effect from ACTION_PLAN.md's Goal, made visible frame by frame instead of as a percentage.

Highlight the revisit column (`t2`) distinctly (e.g. a colored border) so the eye goes straight to it. Row 4 requires one full 16-token autoregressive generation per displayed column, expensive over a long strip, but this is an occasional, human-driven notebook, not part of the training loop, so that's fine. Reuses/extends `save_recon_grid` from `src/utils/viz.py` (more rows, one column per timestep instead of per-batch-sample) rather than inventing a new renderer.

## Implementation Structure

```
src/
  datasets/
    window_dataset.py       # IterableDataset: (H+1)-step windows from data/tokens/
  models/
    world_model.py          # embeddings + GPT blocks + head
  utils/
    revisits.py             # find_revisit_events(agent_pos) -> [(t1, t2, N), ...]
  train_world_model.py       # training loop (argparse)
  eval_world_model.py        # accuracy-vs-N plot + qualitative grids

notebooks/
  inspect_world_model.ipynb  # 4-row path strip: original / tokenizer recon / teacher-forced / autoregressive

checkpoints/worldmodel/      # runtime checkpoints
outputs/worldmodel/          # runtime viz grids + accuracy-vs-N plot
```

## Hyperparameters

| Param | Value |
|---|---|
| Frame-token vocab | 256 (= tokenizer codebook size, must match) |
| Action dim | 6 (raw vector, linear-projected) |
| History length `H` | 16 steps |
| Window length | `H+1 = 17` steps → `L = 289` positions |
| `d_model` | 256 |
| Layers | 4 |
| Heads | 4 |
| MLP ratio | 4× |
| Dropout | 0.0 (add only if a smoke test shows overfitting) |
| Batch size | 64 windows |
| Learning rate | 3e-4 (Adam) |
| Frame stride | 1 |
| Training steps | 30,000 (soft budget) |
| Log/ckpt/viz intervals | 100/1000/1000 steps |
| Revisit distance threshold | 0.5 (agent_pos units). Calibrated: per-step displacement ≈0.25 (fixed sub-step), maze grid ≈1.0 units/cell — 0.5 sits between the two |

## Smoke Tests

1. **Data**: window loader over 5 episodes → shapes `(H+1,16)` tokens / `(H+1,6)` actions, no window crosses a `reset`.
2. **Model**: forward a random window batch → logits `(L,256)`, `loss.backward()` clean.
3. **Training**: tiny run (`--episodes_limit 20 --max_steps 200`) → loss decreases, checkpoint saves/reloads, viz grid (through the frozen tokenizer) exists.
4. **Revisits**: run `revisits.py` on a couple of eval episodes, spot-check a few `(t1, t2, N)` triples by eye against `agent_pos`.
5. **Eval**: accuracy-vs-N binning on a small sample — bins populate, numbers come out.
6. **Full run**: train to convergence on all of `data/tokens/train0`, produce the accuracy-vs-N plot, confirm the cliff at `N ≈ H`.

# Research: memory as a cache in world models

## 1. The problem, formally

A transformer world model learns

```
p_θ(o_{t+1} | o_{t-H+1:t}, a_{t-H+1:t})
```

with fixed context `H`. Let a *revisit* be a pair `(t1, t2)` where `dist(x_{t1}, x_{t2}) < r` and the agent was away in between (`x_t` = position, from `agent_pos`). Define the gap `N = t2 - t1`.

**Attention cannot reach outside `[t2-H, t2)`.** So:

```
N > H  ⇒  o_{t1} is structurally absent from the context at t2
```

not "hard to recall" — provably unavailable. Accuracy at revisit collapses once `N > H`. This is the failure this project's Phase 3 world model is built to demonstrate (WORLDMODEL.md), and it is confirmed as the field's current headline failure mode as of Sept 2026 (§2).

## 2. State of the art (Sept 2026), condensed

| Work | Mechanism | Eviction/selection rule |
|---|---|---|
| Genie 3 (DeepMind, 2025) | Emergent, no explicit memory | none — memory ≈ minutes, then drifts |
| WRBench / "Persistent State Core" ([2606.20545](https://arxiv.org/abs/2606.20545)), MBench ([2606.00793](https://arxiv.org/abs/2606.00793)) | Benchmarks | confirm: revisit consistency is broadly unsolved across 23+ models |
| WorldMem ([2504.12369](https://arxiv.org/abs/2504.12369)) | Frame+pose bank, retrieve by FOV overlap | similarity heuristic |
| **ReWorld** ([2608.23565](https://arxiv.org/abs/2608.23565)) | **Pose-indexed landmark bank, fixed budget, retrieve-nearest-to-pose** | authors explicitly state they use this *"rather than Belady's algorithm or imitation learning approaches"* |
| DIM-WAM ([2606.27677](https://arxiv.org/abs/2606.27677)) | Bounded event memory | novelty/diversity heuristic |
| WorldKV ([2605.22718](https://arxiv.org/abs/2605.22718)) | Evicted-KV retrieval store | similarity + fixed compression, training-free |
| Recall to Imagine / R2I ([2403.04253](https://arxiv.org/pdf/2403.04253)) | Unbounded compressed recurrent state (S4), no bank | superhuman on Memory Maze — proves the *RL-policy* version is solvable without a bank; the *generative, bounded-and-inspectable-memory* version is not |
| Parrot ([2006.16239](https://arxiv.org/abs/2006.16239), ICML 2020) | CDN cache replacement | **trains a network to imitate Bélády's offline-optimal oracle** — the source of this project's method, applied to a disjoint domain (web caching) |
| Cooperative Memory Paging ([2604.12376](https://arxiv.org/abs/2604.12376), **withdrawn**) | LLM chat memory paging | tests Bélády oracle vs. FIFO/LRU/LFU — same design as this project, applied to *text*, retracted before publication |

**Reading across the table**: every video/world-model paper picks a memory bank and a *heuristic* retention rule (similarity, novelty, recency, pose-proximity). The one architecture structurally identical to this project's plan (ReWorld: fixed-size, position-indexed bank, nearest-pose retrieval) explicitly names Bélády/imitation-learning as the alternative it did *not* take. The one place Bélády *was* tried as an oracle baseline (Cooperative Memory Paging) is in an unrelated domain (text) and was withdrawn.

## 3. The gap, stated as a caching problem

Bank `M_t`, capacity `|M_t| ≤ C`, holding snapshots `(x_i, z_i)` (position, 16 VQ tokens). On overflow, an eviction policy `π: M_t → m ∈ M_t` must pick what to drop.

**Bélády's rule** (optimal *iff* the future is known — true here, since trajectories are pre-recorded):

```
evict  argmax_{m ∈ M_t}  next_use(m),      next_use(m) = min{ t' > t : dist(x_{t'}, x_m) < r }   (∞ if never again)
```

No existing world-model paper computes this quantity or benchmarks against it. This project's data (recorded Memory Maze trajectories with ground-truth `agent_pos`) makes `next_use(m)` computable exactly, which is what makes the oracle available at all — most domains don't have this offline luxury; this project's dataset does.

## 4. The solution, on this project's infrastructure

Pipeline is already specified in TOKENIZER.md / WORLDMODEL.md / ACTION_PLAN.md; the cache is the missing Phase 4.

**Snapshot** (from the frozen tokenizer): `s_i = (x_i, z_i)`, `z_i ∈ {0,...,255}^16`, ~18 bytes.

**Bank**: `M_t`, capacity `C`, one snapshot per distinct visited region.

**Retrieval** (at prediction time `t`, current position `x_t`):

```
R_t = { s_i ∈ M_t : dist(x_i, x_t) < r }        (top-k by distance if |R_t| large)
```

`R_t`'s tokens are prepended as extra slots to the existing `17×(H+1)` sequence — same causal transformer, no new mechanism, just more context rows tagged with a "retrieved" step-embedding.

**Five eviction policies** (ACTION_PLAN.md), all triggered on insert when `|M_t| = C`:

1. FIFO — evict oldest inserted.
2. LRU — evict `argmax_m (t - last_retrieved(m))`.
3. keep-spatial-coverage — evict `argmin_m` (distance to nearest other snapshot in `M_t`), i.e. drop the most redundant point.
4. **learned** — network `π_φ(M_t, x_t) → m`, trained by imitation.
5. **Bélády oracle** — §3 formula, computed offline per episode. Upper bound.

**Training the learned policy** (Parrot-style, adapted from web caching to positions): for each recorded episode, compute the Bélády label at every eviction event (which of the `C` bank slots it evicted), train `π_φ` by cross-entropy against that label:

```
L(φ) = − Σ_t  log π_φ( m*_t | M_t, x_t ),        m*_t = argmax_{m ∈ M_t} next_use(m)
```

`π_φ` only sees `(M_t, x_t)` — not the future — same restriction Parrot uses to make the imitation policy deployable online.

**Verification** (required before the eviction comparison means anything): mask `R_t = ∅` on a fine-tuned model. Accuracy must fall back to the no-memory baseline. If it doesn't, the model isn't using the bank and the eviction-policy comparison is measuring noise.

## 5. Deliverables (unchanged from WORLDMODEL.md/ACTION_PLAN.md)

1. Plot: revisit accuracy (y) vs. cache size `C` (x), 5 curves (FIFO, LRU, coverage, learned, Bélády). Claim: learned closes most of the FIFO/LRU–Bélády gap.
2. Clip: same walk, same `C`, LRU vs. learned — hallucinate vs. survive.

## Sources

- Genie 3 — https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
- Current World Models Lack a Persistent State Core (WRBench) — https://arxiv.org/abs/2606.20545
- MBench — https://arxiv.org/abs/2606.00793
- WorldMem — https://arxiv.org/abs/2504.12369
- ReWorld — https://arxiv.org/abs/2608.23565
- DIM-WAM — https://arxiv.org/abs/2606.27677
- WorldKV — https://arxiv.org/abs/2605.22718
- Recall to Imagine / R2I — https://arxiv.org/pdf/2403.04253
- Parrot: An Imitation Learning Approach for Cache Replacement — https://arxiv.org/abs/2006.16239
- Cooperative Memory Paging with Keyword Bookmarks (withdrawn) — https://arxiv.org/abs/2604.12376
- Memory Maze (Pasukonis, Lillicrap, Hafner 2022) — https://arxiv.org/abs/2210.13383

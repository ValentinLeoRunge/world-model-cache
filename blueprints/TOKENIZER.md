# Tokenizer (VQ-VAE)

## Goal
Compress each 64×64 maze frame into 16 discrete tokens (one per 4×4 grid cell in a latent bottleneck). A snapshot becomes: position (2 floats, 16 bytes) + 16 tokens (uint8, 16 bytes) = ~34 bytes on disk, ~18 bytes memory-efficient.

## Architecture

### Input
- Batched RGB images: `(B, 3, 64, 64)`, float32, values in [0, 1].

### Encoder
Four stride-2 convolutions down to 4×4 spatial:
- Conv 3→64, k4 s2 p1: 64→32→32
- Conv 64→128, k4 s2 p1: 32→16→16
- Conv 128→256, k4 s2 p1: 16→8→8
- Conv 256→256, k4 s2 p1: 8→4→4
- 2× residual blocks (3×3 conv pairs with skip) at 256×4×4
- Conv1×1 256→64 (embedding dim D): output `(B, 64, 4, 4)`

### Vector Quantizer (VQ Layer)
- **Input**: `(B, 64, 4, 4)` encoder output, flattened to `(B·16, 64)` (16 spatial cells × batch).
- **Codebook**: `(256, 64)` buffer (not learned param). K=256 codes, D=64 dims. Updated via EMA, never via gradient descent.
- **Forward**:
  - Nearest-neighbor lookup: each of the 16·B codes finds its closest codebook entry → indices `(B, 4, 4)` in [0, 256).
  - Quantized output `z_q`: `(B, 64, 4, 4)`.
  - Straight-through: `z_q_st = z_e + (z_q - z_e).detach()` (gradients flow to encoder, not codebook).
  - Loss term: commitment loss only, `β · mse(z_e, z_q.detach())`, β=0.25.
- **Codebook update** (training only, no_grad):
  - EMA cluster size: `n_i ← decay · n_i + (1 - decay) · count_i`, where count_i = number of codes assigned to cluster i in this batch.
  - EMA centroid: `m_i ← decay · m_i + (1 - decay) · sum(z_e where assigned to i)`.
  - Codebook: `c_i ← m_i / (n_i + smoothing)`. decay=0.99, smoothing=1e-5.
  - Track **perplexity** = exp(entropy of code-usage distribution): monitor codebook collapse (perp → 1 = dead codes, perp → K = uniform).

### Decoder
Mirror of encoder with transposed convolutions, up to 64×64:
- Conv1×1 64→256
- 2× residual blocks at 256×4×4
- ConvTranspose 256→256, k4 s2 p1: 4→8→8
- ConvTranspose 256→128, k4 s2 p1: 8→16→16
- ConvTranspose 128→64, k4 s2 p1: 16→32→32
- ConvTranspose 64→3, k4 s2 p1: 32→64→64
- **Sigmoid activation** → output `(B, 3, 64, 64)` in [0, 1].

### Output
- Reconstructed images: `(B, 3, 64, 64)`, float32, values in [0, 1]. Directly comparable to input for MSE loss.
- Token codes (for inference): `(B, 16)` uint8, one code per latent cell.

## Training

**Data**: Memory Maze train0 (2901 episodes, ~1001 frames each). Load via episode-level shuffle buffer (IterableDataset):
- Shard episodes across 4 DataLoader workers.
- Each worker: keeps 8 episodes "in flight", one `np.load` per episode (full decompress, once), extract `image[::4]` (stride=4 for near-duplicate suppression), drop the rest.
- Return frames as float32 (3, 64, 64) in [0, 1].

**Loss**: MSE(input, recon) + β · MSE(z_e, z_q.detach()).

**Optimizer**: Adam, lr=3e-4, batch size 256 (tiny images, trivial for 8GB GPU).

**Checkpointing**: Every 1000 steps, save model + optimizer state + step to `checkpoints/tokenizer/step_{step:06d}.pt`. Overwrite `latest.pt`.

**Visualization**: Every 1000 steps, save original-vs-recon grid (fixed held-out batch from eval set) to `outputs/tokenizer/step_{step:06d}.png`.

**Evaluation**: Every 2000 steps, run quantitative eval on random 2000-frame sample from data/eval: MSE, PSNR, perplexity, active-code fraction.

**Budget**: ~50,000 steps (soft limit, stop early by inspection).

**Exit criterion**: Rooms visibly recognizable in reconstruction grids (walls, floor, doorways, target colors preserved or blurred but identifiable), codebook perplexity healthy (no collapse to few codes), eval PSNR plateaued.

## Evaluation

Run standalone eval script on a checkpoint against data/eval:
- Random but reproducible sample: 2000–50000 frames, deterministic order per split.
- Metrics: mean MSE, PSNR, codebook perplexity, active-code fraction (# codes used ≥1 / K).
- Saves original-vs-recon grids for human visual sign-off (the true exit criterion, since "recognizable" is qualitative).

## Full-Dataset Tokenization

Once trained and frozen:
1. For each `.npz` episode in data/train0 and data/eval:
   - `np.load(path)` → image `(1001, 64, 64, 3)`.
   - Normalize to float32 [0,1], transpose to `(1001, 3, 64, 64)`.
   - Single forward pass (frozen) through encoder → quantizer → code indices `(1001, 4, 4)`.
   - Flatten to `(1001, 16)` uint8.
2. Write new `.npz` to `data/tokens/{split}/{stem}.npz`:
   - `tokens`: `(1001, 16)` uint8.
   - All non-image fields copied through: `agent_pos`, `agent_dir`, `action`, `maze_layout`, `reward`, `terminal`, `reset`, `target_*`, etc.
   - Output ~276 KB/episode (16 bytes/frame tokens + copied metadata). Total: ~1 GB across both splits.

Resume-safe: skip files that already exist unless `--overwrite`.

## Implementation Structure

```
src/
  datasets/                     # dataset-LOADING code (the .npz files live in ./data/)
    __init__.py
    episode_dataset.py          # IterableDataset, episode-level shuffle buffer
  models/
    __init__.py
    vqvae.py                    # Encoder, VQ layer, Decoder, VQVAE module
  utils/
    __init__.py
    viz.py                      # recon-grid plotting
  train_tokenizer.py            # training loop (argparse)
  eval_tokenizer.py             # quantitative + qualitative eval
  tokenize_dataset.py           # freeze + encode full dataset

requirements.txt                # add: torch, torchvision, tqdm (pinned to installed versions)
checkpoints/tokenizer/          # runtime checkpoints
outputs/tokenizer/              # runtime viz grids
data/tokens/train0/             # tokenized outputs
data/tokens/eval/
```

## Hyperparameters

| Param | Value |
|---|---|
| Codebook size K | 256 |
| Embedding dim D | 64 |
| Batch size | 256 |
| Learning rate | 3e-4 (Adam) |
| Commitment cost β | 0.25 |
| EMA decay | 0.99 |
| Frame stride (train) | 4 |
| Frame stride (tokenize) | 1 |
| Shuffle buffer | 8 episodes/worker |
| DataLoader workers | 4 |
| Training steps | 50,000 (soft budget) |
| Log/ckpt/viz/eval intervals | 100/1000/1000/2000 steps |
"""VQ-VAE tokenizer: 64x64 RGB frame -> 4x4 grid of discrete codes (16 tokens).

One model, three parts: Encoder (conv downsample), EMAVectorQuantizer (codebook,
updated by exponential moving average -- no gradients), Decoder (conv upsample).
See blueprints/TOKENIZER.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """3x3 -> 3x3 conv pair with a skip connection, channels unchanged."""

    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    """(B,3,64,64) -> (B,D,4,4).  Four stride-2 convs + 2 res blocks + 1x1 to D."""

    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),    # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 4, stride=2, padding=1), # 8 -> 4
            ResBlock(256),
            ResBlock(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, d, 1),                        # embedding dim D
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """(B,D,4,4) -> (B,3,64,64) in [0,1].  Mirror of the encoder."""

    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d, 256, 1),
            ResBlock(256),
            ResBlock(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),     # 32 -> 64
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class EMAVectorQuantizer(nn.Module):
    """Nearest-neighbour codebook lookup with straight-through gradients.

    The codebook lives in buffers and is updated by EMA (online k-means), never
    by the optimizer. Only the commitment loss beta * ||z_e - sg(z_q)||^2 is
    returned for backprop (it trains the encoder).
    """

    def __init__(self, k=256, d=64, beta=0.25, decay=0.99, eps=1e-5,
                 dead_threshold=1.0, revive_every=100):
        super().__init__()
        self.k, self.d, self.beta, self.decay, self.eps = k, d, beta, decay, eps
        self.dead_threshold, self.revive_every = dead_threshold, revive_every
        embed = torch.randn(k, d)
        self.register_buffer("embedding", embed)            # (K, D) the codebook
        # n_i and m_i start consistent (m_i / n_i == embed) so that unused codes
        # stay put instead of exploding through the tiny early denominator.
        self.register_buffer("cluster_size", torch.ones(k))  # EMA count n_i
        self.register_buffer("embed_avg", embed.clone())     # EMA sum   m_i
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def forward(self, z_e):
        # z_e: (B, D, H, W) -> flat (N, D), N = B*H*W
        b, d, h, w = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)

        # squared L2 distance to every code, then nearest
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        idx = dist.argmin(1)                                  # (N,)
        onehot = F.one_hot(idx, self.k).type_as(flat)         # (N, K)
        z_q = (onehot @ self.embedding).view(b, h, w, d).permute(0, 3, 1, 2)

        # commitment loss (only term that reaches the encoder from the VQ layer)
        loss = self.beta * F.mse_loss(z_q.detach(), z_e)

        if self.training:
            with torch.no_grad():
                self.cluster_size.mul_(self.decay).add_(
                    onehot.sum(0), alpha=1 - self.decay
                )
                self.embed_avg.mul_(self.decay).add_(
                    onehot.t() @ flat, alpha=1 - self.decay
                )
                self.embedding.copy_(
                    self.embed_avg / (self.cluster_size.unsqueeze(1) + self.eps)
                )

                # revive codes nobody uses: reseed them from real encoder
                # outputs in this batch so they get a foothold next step.
                self._step += 1
                if self.revive_every and self._step % self.revive_every == 0:
                    dead = self.cluster_size < self.dead_threshold
                    n_dead = int(dead.sum())
                    if n_dead:
                        pick = torch.randint(len(flat), (n_dead,), device=flat.device)
                        chosen = flat.detach()[pick]
                        self.embedding[dead] = chosen
                        self.embed_avg[dead] = chosen
                        self.cluster_size[dead] = 1.0

        # straight-through: identity forward, gradient passes z_q -> z_e
        z_q = z_e + (z_q - z_e).detach()

        probs = onehot.mean(0)
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
        return z_q, loss, idx.view(b, h, w), perplexity


class VQVAE(nn.Module):
    def __init__(self, k=256, d=64, beta=0.25, decay=0.99):
        super().__init__()
        self.encoder = Encoder(d)
        self.vq = EMAVectorQuantizer(k, d, beta, decay)
        self.decoder = Decoder(d)

    def forward(self, x):
        """-> recon (B,3,64,64), vq_loss scalar, indices (B,4,4), perplexity."""
        z_e = self.encoder(x)
        z_q, vq_loss, idx, perplexity = self.vq(z_e)
        recon = self.decoder(z_q)
        return recon, vq_loss, idx, perplexity

    @torch.no_grad()
    def encode_indices(self, x):
        """(B,3,64,64) -> (B,4,4) long code indices. Encoder + VQ only, no decoder."""
        _, _, idx, _ = self.vq(self.encoder(x))
        return idx

    @torch.no_grad()
    def tokenize(self, x):
        """(B,3,64,64) -> (B,16) uint8 code indices, row-major over the 4x4 grid."""
        return self.encode_indices(x).reshape(x.shape[0], -1).to(torch.uint8)

    @torch.no_grad()
    def decode_indices(self, idx):
        """(B,16) or (B,4,4) int -> reconstructed images (B,3,64,64) in [0,1]."""
        if idx.dim() == 2:
            idx = idx.view(idx.shape[0], 4, 4)
        z_q = F.embedding(idx.long(), self.vq.embedding)      # (B,4,4,D)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)



#
# 3x64x64



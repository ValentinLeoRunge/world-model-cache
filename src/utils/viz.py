import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt


def _to_hwc(x):
    return x.detach().float().clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()


def save_recon_grid(originals, recons, path, n=8, title=None):
    n = min(n, originals.shape[0])
    orig, rec = _to_hwc(originals[:n]), _to_hwc(recons[:n])

    fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.2))
    axes = axes.reshape(2, n)
    for col in range(n):
        for row, img in ((0, orig[col]), (1, rec[col])):
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=9)
    axes[1, 0].set_ylabel("recon", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(pad=0.3)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_path_strip(rows, path, row_labels, col_labels=None, highlight_col=None, title=None):
    n_cols = rows[0].shape[0]
    imgs = [_to_hwc(r) for r in rows]

    fig, axes = plt.subplots(len(rows), n_cols, figsize=(n_cols * 1.1, len(rows) * 1.3))
    axes = axes.reshape(len(rows), n_cols)
    for r in range(len(rows)):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(imgs[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            if highlight_col is not None and c == highlight_col:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3)
        axes[r, 0].set_ylabel(row_labels[r], fontsize=9)
    if col_labels is not None:
        for c in range(n_cols):
            axes[0, c].set_title(str(col_labels[c]), fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(pad=0.3)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _upscale_nn(imgs, factor):
    if factor <= 1:
        return imgs
    return imgs.repeat(factor, axis=1).repeat(factor, axis=2)


def _write_mp4(imgs, path, fps):
    import imageio  # needs imageio-ffmpeg installed for mp4 writing

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimwrite(path, list(imgs), fps=fps, codec="libx264", pixelformat="yuv420p",
                     output_params=["-crf", "1", "-movflags", "+faststart"])


def save_video(frames, path, fps=8, upscale=4):
    imgs = (_to_hwc(frames) * 255).astype("uint8")  # (T, H, W, 3) uint8
    imgs = _upscale_nn(imgs, upscale)
    _write_mp4(imgs, path, fps)


def save_side_by_side_video(left, right, path, left_label=None, right_label=None,
                            fps=8, upscale=4):
    from PIL import Image, ImageDraw

    left_imgs = _upscale_nn((_to_hwc(left) * 255).astype("uint8"), upscale)
    right_imgs = _upscale_nn((_to_hwc(right) * 255).astype("uint8"), upscale)
    assert left_imgs.shape == right_imgs.shape, "left and right must match in length and size"

    t, h, w, _ = left_imgs.shape
    # fixed, not upscale-scaled: 64*upscale is always a multiple of 16, and so
    # is 16, so the combined width stays a multiple of 16 (ffmpeg's macro
    # block size) -- avoids a silent pad-and-resize that would distort it.
    sep_w = 16
    sep = np.full((t, h, sep_w, 3), 255, dtype="uint8")
    combined = np.concatenate([left_imgs, sep, right_imgs], axis=2)

    if left_label or right_label:
        labeled = []
        for frame in combined:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            if left_label:
                draw.text((4, 4), left_label, fill=(255, 255, 255))
            if right_label:
                draw.text((w + sep_w + 4, 4), right_label, fill=(255, 255, 255))
            labeled.append(np.array(img))
        combined = np.stack(labeled)

    _write_mp4(combined, path, fps)

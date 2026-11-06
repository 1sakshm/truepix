"""Survey for colour grids: compare the pair histogram's periodicity with a same-size smooth null."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

N = 512


def fy_lattice():
    """libjpeg full-range YCbCr to RGB reachable set."""
    half = 1 << 15
    y, cb, cr = [v.astype(np.int64) for v in np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing="ij")]
    r = np.clip(y + ((91881 * (cr - 128) + half) >> 16), 0, 255)
    g = np.clip(y + ((-22554 * (cb - 128) + half - 46802 * (cr - 128)) >> 16), 0, 255)
    b = np.clip(y + ((116130 * (cb - 128) + half) >> 16), 0, 255)
    return np.unique(r << 16 | g << 8 | b)


TABLE = fy_lattice()


def peak(a):
    dr, db = (a[..., 0] - a[..., 1]).ravel(), (a[..., 2] - a[..., 1]).ravel()
    pairs, counts = np.unique(np.stack([dr, db]), axis=1, return_counts=True)
    grid = np.zeros((N, N))
    np.add.at(grid, (pairs[0] % N, pairs[1] % N), np.sqrt(counts))
    # Remove the smooth envelope so only fine periodic structure can peak.
    smooth = np.real(np.fft.ifft2(np.fft.fft2(grid) * np.exp(-((np.fft.fftfreq(N)[:, None] ** 2 + np.fft.fftfreq(N)[None, :] ** 2) * (2 * np.pi * 2.0) ** 2) / 2)))
    F = np.abs(np.fft.fft2(grid - smooth))
    F[0, 0] = 0
    fx = np.fft.fftfreq(N)
    fi, fj = np.meshgrid(fx, fx, indexing="ij")
    F[np.abs(fi) + np.abs(fj) < 0.1] = 0
    alive = F[F > 0]
    return float(alive.max() / np.median(alive)), len(pairs[0])


def run(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int64)
    key = (a[..., 0] << 16 | a[..., 1] << 8 | a[..., 2]).ravel()
    u, c = np.unique(key, return_counts=True)
    share = float(c[np.isin(u, TABLE)].sum() / c.sum())
    score, pairs = peak(a)
    return f"{path.stem:18} peak/median {score:8.1f}  pairs {pairs:7}  unique/px {len(u) / len(key):.3f}  JPEG-lattice share {share:.3f}"


paths = [Path(p) for p in sys.argv[1:]]
with ThreadPoolExecutor(4) as pool:
    for line in pool.map(run, paths):
        print(line, flush=True)

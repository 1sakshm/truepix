"""Recover the (dr, db) lattice basis from the pixels and report it next to known conversions."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

N = 1024
KNOWN = {"JPEG/BT.601 full": (1.402, 0.71414, 0.34414, 1.772), "BT.709 full": (1.5748, 0.46812, 0.18732, 1.8556),
         "BT.601 studio": (1.596, 0.813, 0.391, 2.018), "BT.709 studio": (1.793, 0.533, 0.213, 2.112),
         "BT.2020 full": (1.4746, 0.57135, 0.16455, 1.8814)}
print("expected steps: one Cr unit moves (dr, db) by ..., one Cb unit by ...")
for name, (rv, gv, gu, bu) in KNOWN.items():
    print(f"  {name:18} Cr ({rv + gv:.3f}, {gv:.3f})  Cb ({gu:.3f}, {bu + gu:.3f})")


def run(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int64)
    dr, db = (a[..., 0] - a[..., 1]).ravel(), (a[..., 2] - a[..., 1]).ravel()
    pairs, counts = np.unique(np.stack([dr, db]), axis=1, return_counts=True)
    grid = np.zeros((N, N))
    np.add.at(grid, (pairs[0] % N, pairs[1] % N), np.sqrt(counts))
    fx = np.fft.fftfreq(N)
    gauss = np.exp(-((fx[:, None] ** 2 + fx[None, :] ** 2) * (2 * np.pi * 2.0) ** 2) / 2)
    F = np.abs(np.fft.fft2(grid - np.real(np.fft.ifft2(np.fft.fft2(grid) * gauss))))
    fi, fj = np.meshgrid(fx, fx, indexing="ij")
    F[np.abs(fi) + np.abs(fj) < 0.1] = 0
    order = np.argsort(-F.ravel())[:60000]
    found = []
    for p in order:
        i, j = np.unravel_index(p, F.shape)
        v = np.array([fi[i, j], fj[i, j]])
        if not found:
            found.append(v)
        elif abs(np.linalg.det(np.stack([found[0], v]))) > 0.03:
            found.append(v)
            break
    if len(found) < 2:
        return f"{path.stem}: one direction only {found}"
    W = np.stack(found)
    B = np.linalg.inv(W)
    # Reduce to the shortest basis so results compare with the table above.
    u, v = B[:, 0], B[:, 1]
    for _ in range(8):
        if u @ u > v @ v:
            u, v = v, u
        k = round(float(u @ v / (u @ u)))
        v = v - k * u
    return f"{path.stem:12} basis ({u[0]:+.3f}, {u[1]:+.3f}) ({v[0]:+.3f}, {v[1]:+.3f})  cell area {abs(np.linalg.det(np.stack([u, v]))):.2f}"


paths = [Path(p) for p in sys.argv[1:]]
with ThreadPoolExecutor(4) as pool:
    for line in pool.map(run, paths):
        print(line, flush=True)

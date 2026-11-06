"""Refine the offsets of the fixed-point conversion, clamping to 0..255, on all Kodak colours."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

S = Path(sys.argv[1])
colours = np.concatenate([np.asarray(Image.open(p).convert("RGB")).astype(np.int64).reshape(-1, 3) for p in sorted((S / "kodak_raw").glob("*.ppm"))])
u, c = np.unique(colours, axis=0, return_counts=True)
target = u[:, 0] << 16 | u[:, 1] << 8 | u[:, 2]
ys, c1s, c2s = [v.astype(np.int64) for v in np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing="ij")]


def share(p):
    r = np.clip((p["ll"] * ys + p["r2"] * c2s + p["or"]) >> 10, 0, 255)
    g = np.clip((p["ll"] * ys + p["g1"] * c1s + p["g2"] * c2s + p["og"]) >> 10, 0, 255)
    b = np.clip((p["ll"] * ys + p["b1"] * c1s + p["ob"]) >> 10, 0, 255)
    table = np.unique(r << 16 | g << 8 | b)
    return float(c[np.isin(target, table)].sum() / c.sum()), len(table)


p = {"ll": 1391, "r2": 1865, "g1": -441, "g2": -949, "b1": 2271, "or": -255503, "og": 199140, "ob": -354284}
best, size = share(p)
print("start", round(best, 5), size, flush=True)
for step in (12, 6, 3, 1):
    for key in ("or", "og", "ob"):
        base = p[key]
        for k in (-3, -2, -1, 1, 2, 3):
            q = dict(p)
            q[key] = base + k * step
            s, n = share(q)
            if s > best + 1e-5:
                best, size, p = s, n, q
        print(step, key, p[key], round(best, 5), size, flush=True)
print(json.dumps(p), best, size, flush=True)

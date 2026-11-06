"""Fit a fixed-point PhotoYCC conversion: integer gains, one shift, integer offsets, by coordinate descent."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

S = Path(sys.argv[1])
stems = sys.argv[2:]
colours = np.concatenate([np.asarray(Image.open(S / "kodak_raw" / f"{s}.ppm").convert("RGB")).astype(np.int64).reshape(-1, 3) for s in stems])
u, c = np.unique(colours, axis=0, return_counts=True)
target = u[:, 0] << 16 | u[:, 1] << 8 | u[:, 2]
ys, c1s, c2s = [v.astype(np.int64) for v in np.meshgrid(np.arange(256), np.arange(256), np.arange(256), indexing="ij")]


def share(p):
    r = (p["ll"] * ys + p["r2"] * c2s + p["or"]) >> 10
    g = (p["ll"] * ys + p["g1"] * c1s + p["g2"] * c2s + p["og"]) >> 10
    b = (p["ll"] * ys + p["b1"] * c1s + p["ob"]) >> 10
    ok = (r >= 0) & (r <= 255) & (g >= 0) & (g <= 255) & (b >= 0) & (b <= 255)
    table = np.unique(r[ok] << 16 | g[ok] << 8 | b[ok])
    return float(c[np.isin(target, table)].sum() / c.sum()), len(table)


p = {"ll": 1391, "r2": 1865, "g1": -441, "g2": -949, "b1": 2271, "or": -255023, "og": 199000, "ob": -353784}
best, size = share(p)
print("start", best, size, flush=True)
for round_no in range(4):
    for key, steps in (("or", (-900, -500, -250, -120, -60, 60, 120, 250, 500, 900)), ("og", (-900, -500, -250, -120, -60, 60, 120, 250, 500, 900)), ("ob", (-900, -500, -250, -120, -60, 60, 120, 250, 500, 900)),
                       ("ll", (-2, -1, 1, 2)), ("r2", (-3, -1, 1, 3)), ("g1", (-3, -1, 1, 3)), ("g2", (-3, -1, 1, 3)), ("b1", (-3, -1, 1, 3))):
        base = p[key]
        for step in steps:
            q = dict(p)
            q[key] = base + step
            s, n = share(q)
            if s > best + 1e-5:
                best, size, p = s, n, q
        print(round_no, key, p[key], round(best, 5), size, flush=True)
print(json.dumps(p), best, size)

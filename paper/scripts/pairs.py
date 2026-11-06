"""Print the colour-difference lattice and a raw patch of it, and test whether it is independent of brightness."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

S = Path(sys.argv[1])
stem = sys.argv[2]
a = np.asarray(Image.open(S / "kodak_raw" / f"{stem}.ppm").convert("RGB")).astype(int)
r, g, b = a[..., 0], a[..., 1], a[..., 2]
dr, db = r - g, b - g
vals, counts = np.unique(np.stack([dr.ravel(), db.ravel()]), axis=1, return_counts=True)
top = np.argsort(-counts)[:12]
print("top pairs", [(int(vals[0, i]), int(vals[1, i]), int(counts[i])) for i in top])
print("share of pixels in top 20 / 100 / 300 pairs", [round(float(np.sort(counts)[-n:].sum() / counts.sum()), 3) for n in (20, 100, 300)])
lo_r, lo_b = np.percentile(dr, 30), np.percentile(db, 30)
cr, cb = int(np.median(dr)), int(np.median(db))
print("occupancy map around the median pair", cr, cb, "(rows dr, cols db; # common, + rare, . never)")
hist = {(int(x), int(y)): int(c) for x, y, c in zip(vals[0], vals[1], counts)}
for x in range(cr - 12, cr + 13):
    print(f"{x:5} " + "".join("#" if hist.get((x, y), 0) > 40 else "+" if hist.get((x, y), 0) else "." for y in range(cb - 30, cb + 31)))
print("levels of dr used", np.unique(dr)[:40])
h, w = dr.shape
y0, x0 = h // 3, w // 2
print("dr patch")
print(dr[y0:y0 + 10, x0:x0 + 18])
print("db patch")
print(db[y0:y0 + 10, x0:x0 + 18])
print("g patch")
print(g[y0:y0 + 10, x0:x0 + 18])
# Does brightness matter once the pair of neighbours is equal?
same = (dr[:, 1:] == dr[:, :-1]) & (db[:, 1:] == db[:, :-1])
print("pair same as left", round(float(same.mean()), 3), "pair same as up", round(float(((dr[1:] == dr[:-1]) & (db[1:] == db[:-1])).mean()), 3))
step = np.abs(g[:, 1:] - g[:, :-1])
for lo, hi in ((0, 1), (1, 3), (3, 8), (8, 256)):
    m = (step >= lo) & (step < hi)
    print(f"  green step {lo}-{hi}: pair unchanged {same[m].mean():.3f} on {m.mean():.3f} of pixels")
for yp in range(2):
    print("pair same as left, row parity", yp, [round(float(same[yp::2, xp::2].mean()), 3) for xp in range(2)])

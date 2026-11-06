"""Measure how constrained colours are in Kodak compared with other images, and whether that tracks the paq gap."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

S = Path(sys.argv[1])
paq = {}
for line in (S / "bench" / "kodak_paq.jsonl").read_text().splitlines():
    r = json.loads(line)
    paq[r["stem"]] = r["size"]
pxm = {r["stem"]: r["pxm"] for r in json.loads((S / "bench" / "kodak_pxm.json").read_text())}


def entropy(keys):
    _, counts = np.unique(keys, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def stats(a):
    r, g, b = [a[..., i].astype(np.int64) for i in range(3)]
    colour = r << 16 | g << 8 | b
    n = colour.size
    unique = len(np.unique(colour))
    chroma = (r - g + 256) << 9 | (b - g + 256)
    # Empirical conditional entropies, bits per pixel. The chroma-given-left figure is optimistic for sparse contexts.
    h_g = entropy(g)
    h_rb_g = entropy(colour) - h_g
    left = chroma[:, :-1]
    here = chroma[:, 1:]
    return dict(n=n, unique=unique, per_pixel=unique / n, chroma_pairs=len(np.unique(chroma)), h_rb_given_g=h_rb_g,
                exact_left=float((colour[:, 1:] == colour[:, :-1]).mean()), chroma_left=float((here == left).mean()), rb_parity=[float(((r - g) & 1).mean()), float(((b - g) & 1).mean())])


rows = []
union = set()
for path in sorted((S / "kodak_raw").glob("*.ppm")):
    a = np.asarray(Image.open(path).convert("RGB"))
    s = stats(a)
    s["stem"] = path.stem
    s["gap"] = pxm[path.stem] / paq[path.stem] - 1
    rows.append(s)
    c = (a[..., 0].astype(np.int64) << 16 | a[..., 1].astype(np.int64) << 8 | a[..., 2]).ravel()
    union.update(np.unique(c).tolist())
    print(f"{path.stem} unique {s['unique']:7} {s['per_pixel']:.3f}/px  chroma pairs {s['chroma_pairs']:6}  H(RB|G) {s['h_rb_given_g']:.2f}  same-left {s['exact_left']:.3f} chroma-left {s['chroma_left']:.3f}  gap {s['gap']:+.1%}")
print("union of all Kodak colours", len(union))
print("corr unique/px vs gap", round(float(np.corrcoef([r["per_pixel"] for r in rows], [r["gap"] for r in rows])[0, 1]), 3))
print("\nreference images")
for name in ("astronaut", "coffee", "motorcycle_left", "chelsea"):
    a = np.asarray(Image.open(S / "imgs" / f"{name}.png").convert("RGB"))
    s = stats(a)
    print(f"{name:16} unique {s['unique']:7} {s['per_pixel']:.3f}/px  chroma pairs {s['chroma_pairs']:6}  H(RB|G) {s['h_rb_given_g']:.2f}  same-left {s['exact_left']:.3f}")

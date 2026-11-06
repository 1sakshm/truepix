"""Score PXM on the locked Kodak set. This set must never be used for tuning.

Usage: holdout.py SCRATCHPAD
"""
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pillow_jxl
from PIL import Image

S = Path(sys.argv[1])
ROOT = Path(__file__).parent
paths = sorted((S / "kodak_raw").glob("*.ppm"))
paq = {}
log = S / "bench" / "kodak_paq.jsonl"
if log.exists():
    for line in log.read_text().splitlines():
        r = json.loads(line)
        paq[r["stem"]] = r["size"]


def run(path):
    png = (S / "downloads" / "kodak" / (path.stem + ".png")).stat().st_size
    image = Image.open(path)
    encoder = pillow_jxl.Encoder(mode="RGB", lossless=True, quality=100, use_original_profile=True, effort=9)
    jxl = len(encoder(image.tobytes(), image.width, image.height, jpeg_encode=False))
    with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
        packed, back = Path(folder) / "a.pxm", Path(folder) / "b.ppm"
        start = time.time()
        printed = subprocess.run([str(ROOT.parent / "truepix-core.exe"), "c", str(path), str(packed)], check=True, capture_output=True, text=True).stdout
        encode_seconds = time.time() - start
        start = time.time()
        subprocess.run([str(ROOT.parent / "truepix-core.exe"), "d", str(packed), str(back)], check=True)
        decode_seconds = time.time() - start
        return {"stem": path.stem, "png": png, "jxl": jxl, "pxm": packed.stat().st_size, "paq": paq.get(path.stem),
                "exact": back.read_bytes() == path.read_bytes(), "encode": encode_seconds, "decode": decode_seconds, "mode": printed.split()[-1]}


with ThreadPoolExecutor(5) as pool:
    rows = list(pool.map(run, paths))
for r in rows:
    print(r)
(S / "bench" / "kodak_pxm.json").write_text(json.dumps(rows, indent=1))
pixels = sum(Image.open(p).width * Image.open(p).height for p in paths)
for key in ("png", "jxl", "pxm"):
    total = sum(r[key] for r in rows)
    print(f"{key}: {total:,} bytes, {total * 8 / pixels:.3f} bpp")
pxm = sum(r["pxm"] for r in rows)
jxl = sum(r["jxl"] for r in rows)
print(f"PXM vs JPEG XL e9: {pxm / jxl - 1:+.2%}, beats on {sum(r['pxm'] < r['jxl'] for r in rows)}/{len(rows)}")
done = [r for r in rows if r["paq"]]
if done:
    a, b = sum(r["pxm"] for r in done), sum(r["paq"] for r in done)
    print(f"PXM vs paq8px on {len(done)} images: {a / b - 1:+.2%}, beats on {sum(r['pxm'] < r['paq'] for r in done)}")
print("all exact:", all(r["exact"] for r in rows), " mean encode", round(np.mean([r["encode"] for r in rows]), 1), "s, decode", round(np.mean([r["decode"] for r in rows]), 1), "s")

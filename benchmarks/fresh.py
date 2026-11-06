"""Score the frozen codec on a folder of PNGs: timed alone, then rivals in parallel. Never tune on this.

Usage: fresh.py SCRATCHPAD FOLDER RESULT.json [rivals]
"""
import io
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

S, FOLDER, RESULT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
RIVALS = len(sys.argv) > 4
ROOT = Path(__file__).parent
FLIF = S / "tools" / "flif" / "flif.exe"


def pxm(path):
    image = Image.open(path).convert("RGB")
    row = {"stem": path.stem, "pixels": image.width * image.height, "PNG": path.stat().st_size}
    with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
        src, packed, back = [Path(folder) / n for n in ("a.ppm", "a.pxm", "b.ppm")]
        image.save(src)
        start = time.time()
        printed = subprocess.run([str(ROOT.parent / "truepix-core.exe"), "c", str(src), str(packed)], check=True, capture_output=True, text=True).stdout
        row["encode"] = time.time() - start
        start = time.time()
        subprocess.run([str(ROOT.parent / "truepix-core.exe"), "d", str(packed), str(back)], check=True)
        row["decode"] = time.time() - start
        row["PXM"] = packed.stat().st_size
        row["mode"] = printed.split()[-1]
        row["exact"] = back.read_bytes() == src.read_bytes()
    print(row["stem"], row["PXM"], row["mode"], row["exact"], round(row["encode"], 1), round(row["decode"], 1), flush=True)
    return row


def rivals(row):
    path = FOLDER / f"{row['stem']}.png"
    image = Image.open(path).convert("RGB")
    a = np.asarray(image)
    for effort, name in ((9, "JPEG XL e9"), (10, "JPEG XL e10")):
        data = pillow_jxl.Encoder(mode="RGB", lossless=True, quality=100, use_original_profile=True, effort=effort)(image.tobytes(), image.width, image.height, jpeg_encode=False)
        if np.array_equal(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), a):
            row[name] = len(data)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", lossless=True, quality=100, method=6, exact=True)
    if np.array_equal(np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")), a):
        row["WebP lossless"] = len(buffer.getvalue())
    with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
        src, flif, out = [Path(folder) / n for n in ("a.ppm", "a.flif", "c.ppm")]
        image.save(src)
        if subprocess.run([str(FLIF), "-e", "-E100", "-K", str(src), str(flif)], capture_output=True).returncode == 0:
            subprocess.run([str(FLIF), "-d", str(flif), str(out)], capture_output=True)
            if out.exists() and out.read_bytes()[-a.size:] == a.tobytes():
                row["FLIF"] = flif.stat().st_size
    return row


rows = [pxm(p) for p in sorted(FOLDER.glob("*.png"))]
if RIVALS:
    with ThreadPoolExecutor(5) as pool:
        rows = list(pool.map(rivals, rows))
(S / "bench" / RESULT).write_text(json.dumps(rows, indent=1))
pixels = sum(r["pixels"] for r in rows)
total = sum(r["PXM"] for r in rows)
print(f"images {len(rows)}, all exact {all(r['exact'] for r in rows)}, modes {sorted({r['mode'] for r in rows})}")
print(f"PXM {total:,} bytes, {total * 8 / pixels:.3f} bpp, mean encode {np.mean([r['encode'] for r in rows]):.1f} s, decode {np.mean([r['decode'] for r in rows]):.1f} s, slowest {max(r['encode'] for r in rows):.1f} s")
for method in ("JPEG XL e10", "JPEG XL e9", "FLIF", "WebP lossless", "PNG"):
    have = [r for r in rows if method in r]
    if have:
        theirs, ours = sum(r[method] for r in have), sum(r["PXM"] for r in have)
        print(f"{method:15} on {len(have)}: {theirs:,} bytes, {theirs / ours - 1:+.2%} vs PXM, PXM smaller on {sum(r['PXM'] < r[method] for r in have)}, geo {np.exp(np.mean(np.log([r[method] / r['PXM'] for r in have]))) - 1:+.2%}")

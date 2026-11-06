"""One-time score on the locked DIV2K images 0801-0824. Never tune on this output.

Usage: locked.py SCRATCHPAD
"""
import hashlib
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

S = Path(sys.argv[1])
ROOT = Path(__file__).parent
FLIF = S / "tools" / "flif" / "flif.exe"
LOCKED = S / "locked_div2k"
for line in (S / "bench" / "locked_div2k.sha256").read_text().splitlines():
    digest, name = line.split()
    assert hashlib.sha256((LOCKED / name.lstrip("*")).read_bytes()).hexdigest() == digest, name


def run(path):
    image = Image.open(path).convert("RGB")
    a = np.asarray(image)
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
        flif = Path(folder) / "a.flif"
        out = Path(folder) / "c.ppm"
        done = subprocess.run([str(FLIF), "-e", "-E100", "-K", str(src), str(flif)], capture_output=True)
        if done.returncode == 0:
            subprocess.run([str(FLIF), "-d", str(flif), str(out)], capture_output=True)
            if out.exists() and out.read_bytes()[-a.size:] == a.tobytes():
                row["FLIF"] = flif.stat().st_size
    for effort, name in ((9, "JPEG XL e9"), (10, "JPEG XL e10")):
        start = time.time()
        data = pillow_jxl.Encoder(mode="RGB", lossless=True, quality=100, use_original_profile=True, effort=effort)(image.tobytes(), image.width, image.height, jpeg_encode=False)
        row[name + " seconds"] = time.time() - start
        if np.array_equal(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), a):
            row[name] = len(data)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", lossless=True, quality=100, method=6, exact=True)
    if np.array_equal(np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")), a):
        row["WebP lossless"] = len(buffer.getvalue())
    print(path.stem, row["PXM"], row["mode"], row["exact"], round(row["encode"]), round(row["decode"]), flush=True)
    return row


with ThreadPoolExecutor(5) as pool:
    rows = list(pool.map(run, sorted(LOCKED.glob("*.png"))))
(S / "bench" / "locked_result.json").write_text(json.dumps(rows, indent=1))
pixels = sum(r["pixels"] for r in rows)
pxm = sum(r["PXM"] for r in rows)
print(f"images {len(rows)}, all exact {all(r['exact'] for r in rows)}, modes {sorted({r['mode'] for r in rows})}")
print(f"PXM {pxm:,} bytes, {pxm * 8 / pixels:.3f} bpp, mean encode {np.mean([r['encode'] for r in rows]):.0f} s, decode {np.mean([r['decode'] for r in rows]):.0f} s")
for method in ("JPEG XL e10", "JPEG XL e9", "FLIF", "WebP lossless", "PNG"):
    have = [r for r in rows if method in r]
    if not have:
        continue
    theirs, ours = sum(r[method] for r in have), sum(r["PXM"] for r in have)
    ratios = [r[method] / r["PXM"] for r in have]
    print(f"{method:15} on {len(have)}: {theirs:,} bytes, {theirs / ours - 1:+.2%} vs PXM, PXM smaller on {sum(r['PXM'] < r[method] for r in have)}, geo mean {np.exp(np.mean(np.log(ratios))) - 1:+.2%}")
print("JPEG XL seconds e9, e10:", round(np.mean([r["JPEG XL e9 seconds"] for r in rows]), 1), round(np.mean([r["JPEG XL e10 seconds"] for r in rows]), 1))

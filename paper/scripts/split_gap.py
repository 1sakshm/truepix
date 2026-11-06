"""Split the PXM vs paq8px gap: each codec on the green plane alone, and on the whole image."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

S = Path(sys.argv[1])
PXM = Path(r"D:\Pixel\research\pxm.exe")
PAQ = S / "tools" / "paq" / "paq8px.exe"
full = {r["stem"]: r["pxm"] for r in json.loads((S / "bench" / "kodak_pxm.json").read_text())}
paq_full = {}
for line in (S / "bench" / "kodak_paq.jsonl").read_text().splitlines():
    r = json.loads(line)
    paq_full[r["stem"]] = r["size"]


def run(stem):
    a = np.asarray(Image.open(S / "kodak_raw" / f"{stem}.ppm").convert("RGB"))
    h, w = a.shape[:2]
    out = {"stem": stem}
    with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
        work = Path(folder)
        for name, plane in (("g", a[..., 1]), ("gray", np.rint(0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.uint8))):
            target = work / f"{name}.pgm"
            target.write_bytes(f"P5\n{w} {h}\n255\n".encode() + plane.tobytes())
            subprocess.run([str(PXM), "c", str(target), str(work / f"{name}.pxm")], check=True, capture_output=True)
            subprocess.run([str(PAQ), "-6", str(target), str(work / f"{name}.paq")], capture_output=True, cwd=folder)
            out[name + "_pxm"] = (work / f"{name}.pxm").stat().st_size
            out[name + "_paq"] = next(work.glob(f"{name}.paq*")).stat().st_size
    return out


stems = sys.argv[2:]
with ThreadPoolExecutor(4) as pool:
    rows = list(pool.map(run, stems))
for r in rows:
    s = r["stem"]
    print(s, r)
    print(f"  whole: PXM {full[s]:,} paq {paq_full[s]:,} gap {full[s] - paq_full[s]:,}")
    print(f"  green: PXM {r['g_pxm']:,} paq {r['g_paq']:,} gap {r['g_pxm'] - r['g_paq']:,}")
    print(f"  rest:  PXM {full[s] - r['g_pxm']:,} paq {paq_full[s] - r['g_paq']:,} gap {(full[s] - r['g_pxm']) - (paq_full[s] - r['g_paq']):,}")

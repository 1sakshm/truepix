"""Run PXM on the raw sample images, prove exact decoding, and compare sizes.

Usage: evaluate.py SCRATCHPAD [--names a,b,c] [--no-decode]
"""
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S = Path(sys.argv[1])
ROOT = Path(__file__).parent
names = None
if "--names" in sys.argv:
    names = sys.argv[sys.argv.index("--names") + 1].split(",")
decode = "--no-decode" not in sys.argv
results = json.loads((S / "bench" / "results.json").read_text())["results"]
paq = {}
for line in (S / "bench" / "heavy.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["method"] == "paq8px":
        paq[r["stem"]] = r["size"]
inputs = sorted((S / "bench" / "input").iterdir())
inputs = [p for p in inputs if p.stem in results and not (S / "imgs" / (p.stem + ".jpg")).exists()]
if names:
    inputs = [p for p in inputs if p.stem in names]


def run(path):
    with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
        packed, back = Path(folder) / "a.pxm", Path(folder) / "b.raw"
        start = time.time()
        subprocess.run([str(ROOT.parent / "truepix-core.exe"), "c", str(path), str(packed)], check=True, capture_output=True)
        encode_seconds = time.time() - start
        exact = None
        if decode:
            subprocess.run([str(ROOT.parent / "truepix-core.exe"), "d", str(packed), str(back)], check=True, capture_output=True)
            exact = back.read_bytes() == path.read_bytes()
        return path.stem, packed.stat().st_size, exact, encode_seconds


with ThreadPoolExecutor(6) as pool:
    rows = list(pool.map(run, inputs))

print(f"{'image':18}{'pxm':>9}{'jxl':>9}{'paq':>9}{'vs jxl':>9}{'vs paq':>9}{'exact':>7}{'sec':>7}")
total = {"pxm": 0, "jxl": 0, "paq": 0}
beat_jxl = beat_paq = 0
for stem, size, exact, seconds in rows:
    jxl = results[stem]["JPEG XL e9 (plain)"]["size"]
    p = paq[stem]
    total["pxm"] += size
    total["jxl"] += jxl
    total["paq"] += p
    beat_jxl += size < jxl
    beat_paq += size < p
    print(f"{stem:18}{size:>9}{jxl:>9}{p:>9}{size / jxl - 1:>+9.1%}{size / p - 1:>+9.1%}{str(exact):>7}{seconds:>7.1f}")
print(f"TOTAL {total['pxm']:,}  vs JPEG XL {total['pxm'] / total['jxl'] - 1:+.2%} (beats on {beat_jxl}/{len(rows)})"
      f"  vs paq8px {total['pxm'] / total['paq'] - 1:+.2%} (beats on {beat_paq}/{len(rows)})")
print("all exact:", all(r[2] for r in rows) if decode else "not checked")

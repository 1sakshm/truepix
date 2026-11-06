"""Sizes on the unlocked DIV2K crops: tune.py SCRATCHPAD EXE [gray|colour|both]. Encode only, for speed."""
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S, exe = Path(sys.argv[1]), sys.argv[2]
which = sys.argv[3] if len(sys.argv) > 3 else "both"


def run(path):
    with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
        out = Path(folder) / "a.pxm"
        start = time.time()
        subprocess.run([exe, "c", str(path), str(out)], check=True, capture_output=True)
        return path.suffix, out.stat().st_size, time.time() - start


paths = []
if which != "colour":
    paths += sorted((S / "tune").glob("*.pgm"))
if which != "gray":
    paths += sorted((S / "tune").glob("*.ppm"))
with ThreadPoolExecutor(6) as pool:
    rows = list(pool.map(run, paths))
for suffix in (".pgm", ".ppm"):
    part = [r for r in rows if r[0] == suffix]
    if part:
        print(f"{suffix}: {sum(r[1] for r in part):,} bytes, mean {sum(r[2] for r in part) / len(part):.1f} s", flush=True)

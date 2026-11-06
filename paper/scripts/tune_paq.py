"""paq8px level 6 on the unlocked crops, for reference only."""
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S = Path(sys.argv[1])
PAQ = S / "tools" / "paq" / "paq8px.exe"


def run(path):
    with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
        target = Path(folder) / ("in" + path.suffix)
        shutil.copy(path, target)
        subprocess.run([str(PAQ), "-6", str(target), str(Path(folder) / "a.paq")], capture_output=True, cwd=folder)
        return path.suffix, path.stem, next(Path(folder).glob("a.paq*")).stat().st_size


paths = sorted((S / "tune").glob("*.pgm")) + sorted((S / "tune").glob("*.ppm"))
with ThreadPoolExecutor(4) as pool:
    rows = list(pool.map(run, paths))
(S / "bench" / "tune_paq.json").write_text(json.dumps(rows))
for suffix in (".pgm", ".ppm"):
    print(suffix, sum(r[2] for r in rows if r[0] == suffix))

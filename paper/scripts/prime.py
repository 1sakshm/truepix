"""How much does an image save if the model has already seen a different image? Stacks A above B."""
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

S = Path(sys.argv[1])
exe = str(S / "bench" / "pxm_t.exe")


def size(im):
    with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
        src, out = Path(folder) / "a.pgm", Path(folder) / "a.pxm"
        im.save(src)
        subprocess.run([exe, "c", str(src), str(out)], check=True, capture_output=True)
        return out.stat().st_size


def run(pair):
    a, b = [Image.open(S / "tune" / f"{n}.pgm") for n in pair]
    both = Image.new("L", (a.width, a.height + b.height))
    both.paste(a, (0, 0))
    both.paste(b, (0, a.height))
    sa, sb, sab = size(a), size(b), size(both)
    return pair, sa, sb, sab, (sab - sa) / sb - 1


pairs = [("0825", "0826"), ("0827", "0828"), ("0829", "0830"), ("0831", "0832"), ("0833", "0834"), ("0835", "0836")]
with ThreadPoolExecutor(6) as pool:
    rows = list(pool.map(run, pairs))
for r in rows:
    print(r[0], r[1], r[2], r[3], f"B after A vs B alone {r[4]:+.2%}")
print("mean", sum(r[4] for r in rows) / len(rows))

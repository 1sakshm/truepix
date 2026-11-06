"""Score JPEG recompression once, on photos the JPEG model never saw.

Primary test: DIV2K 0865 to 0888 saved by Pillow at quality 85, baseline, 4:2:0. The rival is JPEG XL's
own JPEG recompression at efforts 9 and 10, with paq8px for reference. Quality 75, quality 95 and
progressive files are secondary. Every Truepix file must rebuild the JPEG byte for byte.
"""

import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pillow_jxl
from PIL import Image

SCRATCH = Path(sys.argv[1])
OUT = Path(sys.argv[2])
CORE = Path(__file__).parent.parent / "truepix-core.exe"
PAQ = SCRATCH / "tools" / "paq" / "paq8px.exe"
SETS = {"q85": {"quality": 85}, "q75": {"quality": 75}, "q95": {"quality": 95}, "q85_progressive": {"quality": 85, "progressive": True}}


def jxl_size(data, width, height, mode, effort):
    encoder = pillow_jxl.Encoder(mode=mode, lossless=True, quality=100, use_container=True, effort=effort)
    return len(encoder(data, width, height, jpeg_encode=True))


def truepix(path):
    with tempfile.TemporaryDirectory() as folder:
        packed, back = Path(folder) / "a.tpix", Path(folder) / "b.jpg"
        start = time.perf_counter()
        subprocess.run([str(CORE), "c", str(path), str(packed)], check=True, capture_output=True)
        middle = time.perf_counter()
        subprocess.run([str(CORE), "d", str(packed), str(back)], check=True, capture_output=True)
        end = time.perf_counter()
        return packed.stat().st_size, back.read_bytes() == path.read_bytes(), middle - start, end - middle


def paq(path):
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "a.jpg"
        target.write_bytes(path.read_bytes())
        subprocess.run([str(PAQ), "-8", str(target), str(Path(folder) / "a.paq")], check=True, capture_output=True)
        return next(p for p in Path(folder).iterdir() if p.name != "a.jpg").stat().st_size


work = OUT.parent / "jpeg_work"
work.mkdir(exist_ok=True)
results = {}
sources = sorted((SCRATCH / "locked_final").glob("*.png"))
for name, options in SETS.items():
    rows = []
    for source in sources:
        image = Image.open(source).convert("RGB")
        path = work / f"{source.stem}_{name}.jpg"
        image.save(path, **options)
        data = path.read_bytes()
        size, exact, encode, decode = truepix(path)
        row = {"stem": source.stem, "jpeg": len(data), "truepix": size, "exact": exact, "encode": encode, "decode": decode,
               "jxl9": jxl_size(data, image.width, image.height, "RGB", 9)}
        if name == "q85":
            row["jxl10"] = jxl_size(data, image.width, image.height, "RGB", 10)
        rows.append(row)
        print(name, row["stem"], row["jpeg"], row["truepix"], row["jxl9"], row["exact"], flush=True)
    if name == "q85":
        with ThreadPoolExecutor(4) as pool:
            for row, size in zip(rows, pool.map(paq, sorted(work.glob("*_q85.jpg")))):
                row["paq8px"] = size
    results[name] = rows
    OUT.write_text(json.dumps(results, indent=1))

for name, rows in results.items():
    total = sum(r["jpeg"] for r in rows)
    line = f"{name}: JPEG {total:,}, Truepix {sum(r['truepix'] for r in rows) / total - 1:+.2%}, JPEG XL e9 {sum(r['jxl9'] for r in rows) / total - 1:+.2%}"
    line += f", wins {sum(r['truepix'] < r['jxl9'] for r in rows)}/{len(rows)}, exact {sum(r['exact'] for r in rows)}/{len(rows)}"
    print(line)
    if "jxl10" in rows[0]:
        print(f"  e10 {sum(r['jxl10'] for r in rows) / total - 1:+.2%}, paq8px {sum(r['paq8px'] for r in rows) / total - 1:+.2%}, encode {sum(r['encode'] for r in rows) / len(rows):.1f} s, decode {sum(r['decode'] for r in rows) / len(rows):.1f} s")

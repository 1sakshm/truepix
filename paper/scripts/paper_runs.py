"""Measurements for the paper, all on the final build. Run one stage at a time so timings stay clean."""
import hashlib
import io
import json
import os
import re
import shutil
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
stage = sys.argv[2]
PXM = r"D:\Pixel\research\pxm.exe"
OUT = S / "bench" / "paper"
OUT.mkdir(exist_ok=True)
FLIF = S / "tools" / "flif" / "flif.exe"


def timed(command):
    start = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True)
    return time.perf_counter() - start, done


def speed():
    """Each codec alone, on five images of the final locked set."""
    rows = []
    for path in sorted((S / "locked_final").glob("*.png"))[:5]:
        image = Image.open(path).convert("RGB")
        a = np.asarray(image)
        with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
            src = Path(folder) / "a.ppm"
            image.save(src)
            for name, side in (("PXM 2x2 tiles", "2"), ("PXM whole", "1"), ("PXM 4x4 tiles", "4")):
                packed, back = Path(folder) / "p.pxm", Path(folder) / "b.ppm"
                enc, _ = timed([PXM, "c", str(src), str(packed), side])
                dec, _ = timed([PXM, "d", str(packed), str(back)])
                assert back.read_bytes() == src.read_bytes()
                rows.append(dict(stem=path.stem, method=name, size=packed.stat().st_size, encode=enc, decode=dec))
            flif, out = Path(folder) / "a.flif", Path(folder) / "c.ppm"
            enc, _ = timed([str(FLIF), "-e", "-E100", "-K", str(src), str(flif)])
            dec, _ = timed([str(FLIF), "-d", str(flif), str(out)])
            assert out.read_bytes()[-a.size:] == a.tobytes()
            rows.append(dict(stem=path.stem, method="FLIF", size=flif.stat().st_size, encode=enc, decode=dec))
        for effort, name in ((9, "JPEG XL e9"), (10, "JPEG XL e10")):
            start = time.perf_counter()
            data = pillow_jxl.Encoder(mode="RGB", lossless=True, quality=100, use_original_profile=True, effort=effort)(image.tobytes(), image.width, image.height, jpeg_encode=False)
            enc = time.perf_counter() - start
            start = time.perf_counter()
            back = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            dec = time.perf_counter() - start
            assert np.array_equal(back, a)
            rows.append(dict(stem=path.stem, method=name, size=len(data), encode=enc, decode=dec))
        start = time.perf_counter()
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", lossless=True, quality=100, method=6, exact=True)
        enc = time.perf_counter() - start
        start = time.perf_counter()
        back = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
        dec = time.perf_counter() - start
        assert np.array_equal(back, a)
        rows.append(dict(stem=path.stem, method="WebP lossless", size=len(buffer.getvalue()), encode=enc, decode=dec))
        print(path.stem, "timed", flush=True)
    (OUT / "speed.json").write_text(json.dumps(rows, indent=1))


def tiles():
    """Tile count against size and time, on unlocked images."""
    rows = []
    for path in sorted((S / "tune" / "full").glob("*.png"))[:6]:
        image = Image.open(path).convert("RGB")
        with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
            src, packed = Path(folder) / "a.ppm", Path(folder) / "p.pxm"
            image.save(src)
            for side in (1, 2, 3, 4, 6):
                enc, _ = timed([PXM, "c", str(src), str(packed), str(side)])
                rows.append(dict(stem=path.stem, side=side, size=packed.stat().st_size, encode=enc))
        print(path.stem, "tiled", flush=True)
    (OUT / "tiles.json").write_text(json.dumps(rows, indent=1))


def design():
    """Per-image final sizes on the 17 design PNGs, and per-plane costs on Kodak."""
    stems = "astronaut brick camera cell chelsea coffee coins color grass gravel ihc logo logo2 moon motorcycle_left page text".split()

    def one(stem):
        src = next((S / "bench" / "input").glob(stem + ".*"))
        with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
            packed, back = Path(folder) / "p.pxm", Path(folder) / "b.raw"
            printed = subprocess.run([PXM, "c", str(src), str(packed)], check=True, capture_output=True, text=True).stdout
            subprocess.run([PXM, "d", str(packed), str(back)], check=True)
            return dict(stem=stem, size=packed.stat().st_size, exact=back.read_bytes() == src.read_bytes(), mode=printed.split()[-1])

    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(one, stems))
    assert all(r["exact"] for r in rows)
    (OUT / "design.json").write_text(json.dumps(rows, indent=1))
    stats = {}
    env = dict(os.environ, PXM_STATS="1")
    for stem in ("kodim01", "kodim03", "kodim13", "kodim20"):
        with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
            stats[stem] = subprocess.run([PXM, "c", str(S / "kodak_raw" / f"{stem}.ppm"), str(Path(folder) / "p.pxm")], check=True, capture_output=True, text=True, env=env).stdout
    (OUT / "planes.json").write_text(json.dumps(stats, indent=1))


stage_fn = {"speed": speed, "tiles": tiles, "design": design}[stage]
stage_fn()
print(stage, "done", flush=True)

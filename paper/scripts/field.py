"""Compare stored PXM sizes with every non-paq lossless method on both image sets."""
import json
import lzma
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import brotli
import bz2
import gzip
import imagecodecs as ic
import numpy as np
import pillow_heif
import pillow_jxl
import tifffile
import io
import zstandard
from PIL import Image

pillow_heif.register_heif_opener()
S = Path(sys.argv[1])
FLIF = S / "tools" / "flif" / "flif.exe"

# PXM sizes from the verified evaluate.py run on the 17 PNGs.
PXM_DEV = {"astronaut": 311692, "brick": 80363, "camera": 116184, "cell": 28909, "chelsea": 103056, "coffee": 315747, "coins": 65128,
           "color": 31019, "grass": 206521, "gravel": 178747, "ihc": 188817, "logo": 83909, "logo2": 12775, "moon": 27742,
           "motorcycle_left": 457215, "page": 35659, "text": 38012}
results = json.loads((S / "bench" / "results.json").read_text())["results"]
for line in (S / "bench" / "heavy.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["method"] == "FLIF":
        results[r["stem"]]["FLIF"] = {"size": r["size"], "exact": r["exact"]}


def report(title, table, pxm):
    print(f"\n{title}\n{'method':22}{'PXM smaller on':>16}{'theirs vs PXM':>15}")
    out = []
    for method in sorted({m for row in table.values() for m in row}):
        stems = [s for s in table if method in table[s]]
        theirs = sum(table[s][method] for s in stems)
        ours = sum(pxm[s] for s in stems)
        out.append((theirs / ours - 1, method, sum(pxm[s] < table[s][method] for s in stems), len(stems), [s for s in stems if pxm[s] >= table[s][method]]))
    for diff, method, wins, n, losses in sorted(out):
        print(f"{method:22}{wins:>10} of {n:<3}{diff:>+15.1%}  {'lost: ' + ', '.join(losses[:5]) if losses else ''}")


dev = {}
skip = {"ORIGINAL", "PIXEL", "PIXEL --max", "paq8px"}
for stem in PXM_DEV:
    dev[stem] = {m: v["size"] for m, v in results[stem].items() if m not in skip and v.get("exact")}
report("17 PNGs, used for design", dev, PXM_DEV)

rows = json.loads((S / "bench" / "kodak_pxm.json").read_text())
assert all(r["exact"] for r in rows)
PXM_KODAK = {r["stem"]: r["pxm"] for r in rows}


def pil(image, fmt, **options):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    data = buffer.getvalue()
    return data, np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def tiff(a, compression):
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, a, compression=compression, predictor=True, photometric="rgb")
    return buffer.getvalue(), tifffile.imread(io.BytesIO(buffer.getvalue()))


def one(path):
    image = Image.open(path).convert("RGB")
    a = np.asarray(image)
    out = {}

    def check(name, data, back):
        if np.array_equal(np.asarray(back).reshape(a.shape), a):
            out[name] = len(data)
        else:
            print("NOT EXACT", path.stem, name, flush=True)

    def attempt(name, fn):
        try:
            check(name, *fn())
        except Exception as error:
            print("FAILED", path.stem, name, type(error).__name__, str(error)[:80], flush=True)

    for effort, name in ((9, "JPEG XL e9"), (10, "JPEG XL e10")):
        data = pillow_jxl.Encoder(mode="RGB", lossless=True, quality=100, use_original_profile=True, effort=effort)(image.tobytes(), image.width, image.height, jpeg_encode=False)
        out[name] = len(data)
    attempt("WebP lossless", lambda: pil(image, "WEBP", lossless=True, quality=100, method=6, exact=True))
    attempt("PNG optimised", lambda: pil(image, "PNG", optimize=True))
    attempt("HEIC lossless", lambda: pil(image, "HEIF", quality=-1, chroma=444, matrix_coefficients=0))
    attempt("JPEG-LS", lambda: ((d := bytes(ic.jpegls_encode(a))), ic.jpegls_decode(d)))
    attempt("JPEG 2000", lambda: ((d := bytes(ic.jpeg2k_encode(a, level=0, reversible=True))), ic.jpeg2k_decode(d)))
    attempt("JPEG XR", lambda: ((d := bytes(ic.jpegxr_encode(a, level=1.0))), ic.jpegxr_decode(d)))
    attempt("AVIF lossless", lambda: ((d := bytes(ic.avif_encode(a, level=ic.AVIF.QUALITY.LOSSLESS if hasattr(ic.AVIF, "QUALITY") else 100, speed=4))), ic.avif_decode(d)))
    attempt("QOI", lambda: ((d := bytes(ic.qoi_encode(a))), ic.qoi_decode(d)))
    attempt("TIFF deflate", lambda: tiff(a, "zlib"))
    attempt("TIFF zstd", lambda: tiff(a, "zstd"))
    png = (S / "downloads" / "kodak" / (path.stem + ".png")).read_bytes()
    packers = {"ZIP/gzip": (lambda d: gzip.compress(d, 9), gzip.decompress), "bzip2": (lambda d: bz2.compress(d, 9), bz2.decompress),
               "7z/LZMA": (lambda d: lzma.compress(d, preset=9 | lzma.PRESET_EXTREME), lzma.decompress),
               "Zstandard": (lambda d: zstandard.ZstdCompressor(level=22).compress(d), lambda d: zstandard.ZstdDecompressor().decompress(d, max_output_size=1 << 28)),
               "Brotli": (lambda d: brotli.compress(d, quality=11, lgwin=24), brotli.decompress)}
    raw = a.tobytes()
    for name, (pack, unpack) in packers.items():
        best = None
        for payload in (png, raw):
            packed = pack(payload)
            assert unpack(packed) == payload
            best = len(packed) if best is None else min(best, len(packed))
        out[name] = best
    with tempfile.TemporaryDirectory(dir=S / "bench") as folder:
        packed, back = Path(folder) / "a.flif", Path(folder) / "b.ppm"
        done = subprocess.run([str(FLIF), "-e", "-E100", "-K", str(path), str(packed)], capture_output=True)
        if done.returncode == 0:
            subprocess.run([str(FLIF), "-d", str(packed), str(back)], capture_output=True)
            if back.exists() and back.read_bytes()[-a.size:] == a.tobytes():
                out["FLIF"] = packed.stat().st_size
            else:
                print("NOT EXACT", path.stem, "FLIF", flush=True)
    print(path.stem, "done", flush=True)
    return path.stem, out


start = time.time()
with ThreadPoolExecutor(6) as pool:
    kodak = dict(pool.map(one, sorted((S / "kodak_raw").glob("*.ppm"))))
(S / "bench" / "kodak_field.json").write_text(json.dumps(kodak, indent=1))
report("24 Kodak images, locked", kodak, PXM_KODAK)
print("seconds", round(time.time() - start))

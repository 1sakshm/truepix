import io, json, subprocess, sys, tempfile, time
from pathlib import Path
from PIL import Image
sys.path.insert(0, r"D:\Photophone")
S = Path(sys.argv[1]); D = S/"jpg"; CORE = sys.argv[2] if len(sys.argv) > 2 else r"D:\Truepix\truepix-core.exe"
T = D/"tune"; T.mkdir(exist_ok=True)
if not list(T.glob("*.jpg")):
    srcs = sorted((S/"tune").glob("*.ppm"))
    for i, p in enumerate(srcs[:8]):
        im = Image.open(p).convert("RGB")
        q = [75, 85, 95, 50][i % 4]; ss = [2, 2, 0, 2][i % 4]
        im.save(T/f"{p.stem}_q{q}.jpg", quality=q, subsampling=ss)
    im = Image.open(srcs[8]).convert("RGB")
    im.save(T/"opt_q85.jpg", quality=85, optimize=True)
    im.convert("L").save(T/"grey_q85.jpg", quality=85)
    im.save(T/"ss1_q90.jpg", quality=90, subsampling=1)
    im.save(T/"cmyk.jpg", quality=85) if False else None
    Image.open(srcs[9]).convert("RGB").save(T/"exif_q80.jpg", quality=80, exif=b"Exif\0\0MM\0*\0\0\0\x08\0\0", comment=b"hello", icc_profile=b"x"*3000)
    im.save(T/"prog.jpg", quality=85, progressive=True)
files = sorted(T.glob("*.jpg")) + [Path(p) for p in map(str, [D/"rocket.jpg", D/"retina.jpg", D/"hubble_deep_field.jpg"])]
try:
    import pillow_jxl
    has_jxl = True
except Exception as e:
    has_jxl = False; print("no jxl", e)
total = tot_jxl = tot_ours = 0; rows = []
for f in files:
    raw = f.read_bytes()
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d)/"a.tpj", Path(d)/"b.jpg"
        t = time.time(); r = subprocess.run([CORE, "c", str(f), str(a)], capture_output=True, text=True); enc = time.time() - t
        if r.returncode:
            print(f"{f.name:22} {len(raw):>9,}  skipped: {r.stderr.strip()}"); continue
        t = time.time(); subprocess.run([CORE, "d", str(a), str(b)], check=True, capture_output=True); dec = time.time() - t
        exact = b.read_bytes() == raw; size = a.stat().st_size
    jx = 0
    if has_jxl:
        try:
            buf = io.BytesIO()
            Image.open(f).save(buf, format="JXL", lossless_jpeg=True, effort=9)
            jx = len(buf.getvalue())
        except Exception as e:
            jx = 0
    total += len(raw); tot_ours += size; tot_jxl += jx or size
    print(f"{f.name:22} {len(raw):>9,} {size:>9,} {size/len(raw)-1:+7.2%}  jxl {jx:>9,} {(jx/len(raw)-1) if jx else 0:+7.2%} {'exact' if exact else 'DIFFERENT'} enc {enc:.1f} dec {dec:.1f}", flush=True)
print(f"TOTAL {total:,} ours {tot_ours:,} {tot_ours/total-1:+.2%}  jxl {tot_jxl:,} {tot_jxl/total-1:+.2%}")

"""Turn the raw results into the tables, plots and macros the paper reads.

Usage: make_data.py SCRATCHPAD
Copies the raw JSON into results/, so the paper can be rebuilt without the scratch folder.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

S = Path(sys.argv[1]) if len(sys.argv) > 1 else None
HERE = Path(__file__).parent
RAW = HERE / "results"
DATA = HERE / "data"
RAW.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
if S:
    for source, name in ((S / "bench" / "final_result.json", "set_c.json"), (S / "bench" / "fresh_result.json", "set_b.json"), (S / "bench" / "locked_result.json", "set_a.json"),
                         (S / "bench" / "spent_rerun.json", "set_a_tiled.json"), (S / "bench" / "kodak_mode2.json", "kodak_final.json"), (S / "bench" / "kodak_pxm.json", "kodak_before.json"),
                         (S / "bench" / "kodak_field.json", "kodak_rivals.json"), (S / "bench" / "results.json", "design_rivals.json"), (S / "bench" / "kodak_paq.jsonl", "kodak_paq.jsonl"),
                         (S / "bench" / "paper" / "speed.json", "speed.json"), (S / "bench" / "paper" / "tiles.json", "tiles.json"), (S / "bench" / "paper" / "design.json", "design.json"),
                         (S / "bench" / "paper" / "planes.json", "planes.json"), (S / "bench" / "paper" / "matrix.json", "matrix.json")):
        shutil.copy(source, RAW / name)
    for name, path in (("lattice_kodak", S / "kodak_raw" / "kodim01.ppm"), ("lattice_ordinary", S / "imgs" / "astronaut.png")):
        a = np.asarray(Image.open(path).convert("RGB")).astype(np.int64)
        dr, db = a[..., 0] - a[..., 1], a[..., 2] - a[..., 1]
        cr, cb = int(np.median(dr)), int(np.median(db))
        lo_r, hi_r, lo_b, hi_b = cr - 22, cr + 22, cb - 22, cb + 22
        grid = np.zeros((hi_r - lo_r + 1, hi_b - lo_b + 1))
        keep = (dr >= lo_r) & (dr <= hi_r) & (db >= lo_b) & (db <= hi_b)
        np.add.at(grid, (dr[keep] - lo_r, db[keep] - lo_b), 1)
        shade = np.log1p(grid)
        shade = 255 - (shade / max(shade.max(), 1) * 255)
        shade[grid > 0] = np.minimum(shade[grid > 0], 190)
        Image.fromarray(shade[::-1].astype(np.uint8)).resize((grid.shape[1] * 8, grid.shape[0] * 8), Image.NEAREST).save(DATA / f"{name}.png")
        (RAW / f"{name}.json").write_text(json.dumps({"dr": [lo_r, hi_r], "db": [lo_b, hi_b], "pairs": len(np.unique((dr + 512) * 1024 + db + 512))}))

load = lambda name: json.loads((RAW / name).read_text())
macros = {}


def num(name, value):
    macros[name] = f"{value:,}".replace(",", "{,}") if isinstance(value, int) else str(value)


def pct(a, b, digits=2):
    return f"{(a / b - 1) * 100:+.{digits}f}"


def plain(a, b, digits=2):
    return f"{(a / b - 1) * 100:.{digits}f}"


# Hold-out sets.
sets = {"A": load("set_a_tiled.json"), "B": load("set_b.json"), "C": load("set_c.json")}
for r, old in zip(sets["A"], load("set_a.json")):
    assert r["stem"] == old["stem"]
    for key in ("JPEG XL e9", "JPEG XL e10", "FLIF", "WebP lossless"):
        r[key] = old[key]
for label, rows in sets.items():
    assert all(r["exact"] for r in rows) and len(rows) == 24
    total = sum(r["PXM"] for r in rows)
    pixels = sum(r["pixels"] for r in rows)
    num(f"set{label}pxm", total)
    num(f"set{label}bpp", f"{total * 8 / pixels:.3f}")
    for key, short in (("JPEG XL e10", "ten"), ("JPEG XL e9", "nine"), ("FLIF", "flif"), ("WebP lossless", "webp"), ("PNG", "png")):
        theirs = sum(r[key] for r in rows)
        num(f"set{label}{short}wins", sum(r["PXM"] < r[key] for r in rows))
        num(f"set{label}{short}", theirs)
        num(f"set{label}{short}gain", plain(theirs, total))
        num(f"set{label}{short}bpp", f"{theirs * 8 / pixels:.3f}")
    num(f"set{label}enc", f"{np.mean([r['encode'] for r in rows]):.1f}")
    num(f"set{label}dec", f"{np.mean([r['decode'] for r in rows]):.1f}")

rows = sets["C"]
ratios = np.array([r["JPEG XL e10"] / r["PXM"] for r in rows])
rng = np.random.default_rng(1)
boot = np.exp(np.log(ratios)[rng.integers(0, 24, (20000, 24))].mean(axis=1))
num("cgeo", f"{(np.exp(np.log(ratios).mean()) - 1) * 100:.2f}")
num("cgeolo", f"{(np.percentile(boot, 2.5) - 1) * 100:.2f}")
num("cgeohi", f"{(np.percentile(boot, 97.5) - 1) * 100:.2f}")
num("cminsave", f"{(ratios.min() - 1) * 100:.2f}")
num("cmaxsave", f"{(ratios.max() - 1) * 100:.2f}")
num("cslowest", f"{max(r['encode'] for r in rows):.1f}")
num("cslowestdec", f"{max(r['decode'] for r in rows):.1f}")
with (DATA / "savings.dat").open("w") as f:
    f.write("rank stem saving\n")
    for i, (saving, stem) in enumerate(sorted(((1 - r["PXM"] / r["JPEG XL e10"]) * 100, r["stem"]) for r in rows)):
        f.write(f"{i + 1} {stem} {saving:.3f}\n")

# Tiling, on set A with the same codec family.
whole = sum(r["PXM"] for r in load("set_a.json"))
num("tilecostA", plain(sum(r["PXM"] for r in sets["A"]), whole))

# Kodak.
final = {r["stem"]: r for r in load("kodak_final.json")}
before = {r["stem"]: r for r in load("kodak_before.json")}
paq = {}
for line in (RAW / "kodak_paq.jsonl").read_text().splitlines():
    r = json.loads(line)
    paq[r["stem"]] = r["size"]
assert all(r["exact"] for r in final.values())
stems = sorted(final)
pixels = 24 * 768 * 512
with (DATA / "kodak.dat").open("w") as f:
    f.write("i stem before final paq\n")
    for i, s in enumerate(stems):
        f.write(f"{i + 1} {s[-2:]} {before[s]['pxm'] * 8 / (768 * 512):.3f} {final[s]['size'] * 8 / (768 * 512):.3f} {paq[s] * 8 / (768 * 512):.3f}\n")
kfinal, kbefore, kpaq = sum(r["size"] for r in final.values()), sum(r["pxm"] for r in before.values()), sum(paq.values())
num("kfinal", kfinal)
num("kbefore", kbefore)
num("kpaq", kpaq)
num("kfinalbpp", f"{kfinal * 8 / pixels:.3f}")
num("kbeforebpp", f"{kbefore * 8 / pixels:.3f}")
num("kpaqbpp", f"{kpaq * 8 / pixels:.3f}")
num("kdrop", plain(kbefore, kfinal) and f"{(1 - kfinal / kbefore) * 100:.1f}")
num("kpaqgap", plain(kfinal, kpaq))
num("kpaqgapbefore", plain(kbefore, kpaq))
num("kpaqwins", sum(final[s]["size"] < paq[s] for s in stems))
num("kpaqvsjxl", f"{(1 - kpaq / sum(r['jxl'] for r in before.values())) * 100:.1f}")
num("kmodetwo", sum(r["mode"] == "2" for r in final.values()))
shares = [r["share"] for r in final.values()]
num("ksharelo", f"{min(shares) * 100:.1f}")
num("ksharehi", f"{max(shares) * 100:.1f}")
num("kshare", f"{np.mean(shares) * 100:.1f}")

rivals = load("kodak_rivals.json")
field = []
for method in sorted({m for row in rivals.values() for m in row}):
    have = [s for s in rivals if method in rivals[s]]
    assert len(have) == 24
    field.append((sum(rivals[s][method] for s in have), method, sum(final[s]["size"] < rivals[s][method] for s in have)))
field.sort()
design = {r["stem"]: r["size"] for r in load("design.json")}
results = load("design_rivals.json")["results"]
num("dfinal", sum(design.values()))
with (DATA / "kodak_field.tex").open("w") as f:
    for size, method, wins in field[:9]:
        f.write(f"{method.replace("PXM", "Truepix")} & {size:,} & {size * 8 / pixels:.3f} & {pct(size, kfinal, 1)}\\% & {wins}/24 \\\\\n".replace(",", "{,}"))

skip = {"ORIGINAL", "PIXEL", "PIXEL --max", "paq8px"}
dev = {}
for stem in design:
    for method, value in results[stem].items():
        if method not in skip and value.get("exact"):
            dev.setdefault(method, {})[stem] = value["size"]
order = []
for method, sizes in dev.items():
    if len(sizes) == len(design):
        ours = sum(design[s] for s in sizes)
        theirs = sum(sizes.values())
        order.append((theirs / ours, method, theirs, sum(design[s] < sizes[s] for s in sizes)))
order.sort()
num("dbestname", order[0][1].replace("(plain)", "").strip())
num("dbestgain", f"{(order[0][0] - 1) * 100:.1f}")
num("dbestwins", order[0][3])

# Speed, tiles.
speed = load("speed.json")
methods = ["PXM whole", "PXM 2x2 tiles", "PXM 4x4 tiles", "JPEG XL e10", "JPEG XL e9", "FLIF", "WebP lossless"]
base = sum(r["size"] for r in speed if r["method"] == "PXM 2x2 tiles")
with (DATA / "speed.tex").open("w") as f, (DATA / "speed_pxm.dat").open("w") as d, (DATA / "speed_rivals.dat").open("w") as e:
    d.write("enc dec rel label\n")
    e.write("enc dec rel label\n")
    for method in methods:
        part = [r for r in speed if r["method"] == method]
        assert len(part) == 5
        enc, dec, size = np.mean([r["encode"] for r in part]), np.mean([r["decode"] for r in part]), sum(r["size"] for r in part)
        f.write(f"{method.replace('PXM', 'Truepix').replace('2x2', '$2{{\\times}}2$').replace('4x4', '$4{{\\times}}4$')} & {enc:.1f} & {dec:.2f} & {(size / base - 1) * 100:+.2f}\\% \\\\\n")
        (d if method.startswith("PXM") else e).write(f"{enc:.2f} {dec:.3f} {(size / base - 1) * 100:.3f} {method.replace('PXM', 'Truepix').replace(' ', '~')}\n")
        key = method.replace(" ", "").replace("2x2tiles", "two").replace("4x4tiles", "four").replace("JPEGXLe", "jxl").replace("10", "ten").replace("9", "nine").replace("WebPlossless", "webp").lower()
        num("sp" + key + "enc", f"{enc:.1f}")
        num("sp" + key + "dec", f"{dec:.2f}")
num("decodegap", int(round(np.mean([r["decode"] for r in speed if r["method"] == "PXM 2x2 tiles"]) / np.mean([r["decode"] for r in speed if r["method"] == "JPEG XL e9"]))))

tiles = load("tiles.json")
sides = sorted({r["side"] for r in tiles})
one = sum(r["size"] for r in tiles if r["side"] == 1)
with (DATA / "tiles.dat").open("w") as f:
    f.write("side cost time\n")
    for side in sides:
        part = [r for r in tiles if r["side"] == side]
        f.write(f"{side} {(sum(r['size'] for r in part) / one - 1) * 100:.3f} {np.mean([r['encode'] for r in part]):.2f}\n")
num("tilecosttwo", f"{(sum(r['size'] for r in tiles if r['side'] == 2) / one - 1) * 100:.2f}")
num("tilefactor", f"{np.mean([r['encode'] for r in tiles if r['side'] == 1]) / np.mean([r['encode'] for r in tiles if r['side'] == 2]):.1f}")

# Per-plane costs in bits per pixel.
planes = load("planes.json")
for stem, text in planes.items():
    costs = {}
    for line in text.splitlines():
        if line.startswith("mode"):
            words = line.split()
            costs[(int(words[1]), int(words[3].rstrip(":")))] = [float(words[i]) for i in (6, 10, 14, 18)]
    if stem == "kodim01":
        for (mode, plane), values in costs.items():
            key = f"pl{'abc'[mode]}{'xyz'[plane]}"
            num(key, f"{np.mean(values):.2f}")
            for i, value in enumerate(values):
                num(key + "ABCD"[i], f"{value:.2f}")

matrix = load("matrix.json")
assert matrix["exact"] == matrix["decodes"] == 245 and all(matrix["identical"].values())
num("matrixdecodes", matrix["decodes"])
lattice = {n: load(f"{n}.json") for n in ("lattice_kodak", "lattice_ordinary")}
for name, value in lattice.items():
    short = "K" if "kodak" in name else "O"
    num("lat" + short + "pairs", value["pairs"])
    num("lat" + short + "xmin", value["db"][0])
    num("lat" + short + "xmax", value["db"][1])
    num("lat" + short + "ymin", value["dr"][0])
    num("lat" + short + "ymax", value["dr"][1])

fast = json.loads((HERE / "results" / "speedup.json").read_text())
race = fast["race"]
num("fastdecold", f"{race['decode_old']:.2f}")
num("fastdecnew", f"{race['decode_new']:.2f}")
num("fastencold", f"{race['encode_old']:.2f}")
num("fastencnew", f"{race['encode_new']:.2f}")
num("fastdecsave", int(round((1 - race["decode_new"] / race["decode_old"]) * 100)))
num("fastencsave", int(round((1 - race["encode_new"] / race["encode_old"]) * 100)))
num("fastgap", int(round(race["decode_new"] / np.mean([r["decode"] for r in speed if r["method"] == "JPEG XL e9"]))))
num("fastfiles", fast["matrix"]["match_old_baseline"])
num("fastdecodes", fast["matrix"]["exact"])

jpeg = json.loads((HERE / "results" / "jpeg.json").read_text())
for key, rows in jpeg.items():
    short = key.replace("_progressive", "p").replace("85", "E").replace("75", "S").replace("95", "N")
    total = sum(r["jpeg"] for r in rows)
    saved = lambda field: f"{(1 - sum(r[field] for r in rows) / total) * 100:.2f}"
    num("jp" + short + "tp", saved("truepix"))
    num("jp" + short + "jxl", saved("jxl9"))
    num("jp" + short + "wins", sum(r["truepix"] < r["jxl9"] for r in rows))
    num("jp" + short + "exact", sum(r["exact"] for r in rows))
rows = jpeg["q85"]
total = sum(r["jpeg"] for r in rows)
num("jptenS", f"{(1 - sum(r['jxl10'] for r in rows) / total) * 100:.2f}")
num("jppaq", f"{(1 - sum(r['paq8px'] for r in rows) / total) * 100:.2f}")
num("jpenc", f"{sum(r['encode'] for r in rows) / len(rows):.1f}")
num("jpdec", f"{sum(r['decode'] for r in rows) / len(rows):.1f}")
jm = json.loads((HERE / "results" / "jpeg_matrix.json").read_text())
num("jmfiles", jm["match_old_baseline"])
num("jmdecodes", jm["exact"])

with (HERE / "numbers.tex").open("w") as f:
    f.write("% Generated by make_data.py. Do not edit.\n")
    for name, value in macros.items():
        # TeX macro names may only hold letters.
        f.write(f"\\newcommand{{\\{name.replace('0', 'zero')}}}{{{value}}}\n")
for key in sorted(macros):
    print(key, macros[key])

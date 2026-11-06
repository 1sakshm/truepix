"""Build the Truepix website from the paper's data, so its charts match the results."""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
PAPER = HERE.parent / "paper"

# Set C, the final test: bits per pixel and how much larger each rival's total was.
METHODS = [
    ("Truepix", 8.110, None),
    ("JPEG XL, effort 10", 8.404, 3.62),
    ("JPEG XL, effort 9", 8.596, 5.99),
    ("FLIF", 8.863, 9.28),
    ("WebP lossless", 9.447, 16.48),
    ("PNG as distributed", 12.832, 58.22),
]
SPEED = [
    ("Truepix", "12.0", "smallest"),
    ("JPEG XL, effort 10", "42.0", "+3.71%"),
    ("JPEG XL, effort 9", "1.8", "+6.17%"),
    ("FLIF", "14.3", "+10.62%"),
    ("WebP lossless", "10.0", "+20.27%"),
]
BUILDS = ["GCC AVX2", "GCC FMA", "GCC SSE2", "GCC -O0", "Clang x64", "Clang x86", "WebAssembly"]
IMAGES = ["Plain RGB", "Photo CD", "JPEG history", "Greyscale", "RGBA"]


def method_rows():
    top = max(bpp for _, bpp, _ in METHODS)
    rows = []
    for index, (name, bpp, larger) in enumerate(METHODS):
        ours = " ours" if larger is None else ""
        note = "smallest" if larger is None else f"+{larger:.2f}%"
        rows.append(
            f'<li class="rank{ours}" style="--w:{bpp / top * 100:.1f}%;--i:{index}"><span class="rank-name">{name}</span>'
            f'<span class="rank-bar"></span><span class="rank-bpp">{bpp:.3f}</span><span class="rank-note">{note}</span></li>'
        )
    return "\n".join(rows)


def savings():
    lines = (PAPER / "data" / "savings.dat").read_text().split("\n")[1:]
    values = [(fields[1], float(fields[2])) for fields in map(str.split, filter(None, lines))]
    bars = "".join(
        f'<i style="--h:{saving / 8 * 100:.1f}%;--i:{index}" title="Image {stem}: {saving:.2f}% saved"></i>'
        for index, (stem, saving) in enumerate(values)
    )
    return bars, values


def speed_rows():
    rows = []
    for name, encode, size in SPEED:
        ours = ' class="ours"' if name == "Truepix" else ""
        rows.append(f"<tr{ours}><th scope=\"row\">{name}</th><td>{encode}</td><td>{size}</td></tr>")
    return "\n".join(rows)


def wall():
    groups = []
    for image in IMAGES:
        cells = "".join(f'<i style="--i:{index}"></i>' for index in range(len(BUILDS) ** 2))
        groups.append(f'<figure class="wall-set"><div class="wall-grid">{cells}</div><figcaption>{image}</figcaption></figure>')
    return "".join(groups)


bars, values = savings()
page = (HERE / "template.html").read_text(encoding="utf-8")
for key, value in {
    "@@METHODS@@": method_rows(),
    "@@SAVINGS@@": bars,
    "@@LOW@@": f"{values[0][1]:.2f}",
    "@@HIGH@@": f"{values[-1][1]:.2f}",
    "@@SPEED@@": speed_rows(),
    "@@WALL@@": wall(),
    "@@BUILDS@@": ", ".join(BUILDS),
    "@@MARK@@": (HERE / "logo.svg").read_text(encoding="utf-8").strip().replace(' role="img" aria-label="Truepix"', ' aria-hidden="true"'),
}.items():
    page = page.replace(key, value)
assert "@@" not in page

(HERE / "index.html").write_text(page, encoding="utf-8")
shutil.copyfile(PAPER / "paper.pdf", HERE / "paper.pdf")

if len(sys.argv) > 1:  # A copy without the document tags, which the Artifact tool adds itself.
    body = page[page.index("<title>"):].replace("</head>\n<body>\n", "").replace("</body>\n</html>\n", "")
    Path(sys.argv[1]).write_text(body, encoding="utf-8")

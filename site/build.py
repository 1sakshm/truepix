"""Build the Truepix website from the paper's data, so its charts match the results."""

import base64
import io
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

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
    ("Truepix", "17.2", "10.59", "+0.00%"),
    ("JPEG XL, effort 10", "42.0", "0.08", "+3.71%"),
    ("JPEG XL, effort 9", "1.8", "0.08", "+6.17%"),
    ("FLIF", "14.3", "0.95", "+10.62%"),
    ("WebP lossless", "10.0", "0.07", "+20.27%"),
]


def mask(name):
    """Turn a dark-on-white lattice plot into an alpha mask, so CSS can colour it for each theme."""
    grey = Image.open(PAPER / "data" / name).convert("L")
    image = Image.new("RGBA", grey.size, (0, 0, 0, 0))
    image.putalpha(ImageOps.invert(grey))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def method_rows():
    top = max(bpp for _, bpp, _ in METHODS)
    rows = []
    for name, bpp, larger in METHODS:
        ours = " ours" if larger is None else ""
        note = "smallest" if larger is None else f"{larger:.2f}% larger"
        rows.append(
            f'<li class="row{ours}"><span class="row-name">{name}</span>'
            f'<span class="track"><span class="bar" style="width:{bpp / top * 100:.1f}%"></span></span>'
            f'<span class="row-bpp">{bpp:.3f}</span><span class="row-note">{note}</span></li>'
        )
    return "\n".join(rows)


def savings_chart():
    lines = (PAPER / "data" / "savings.dat").read_text().split("\n")[1:]
    savings = [(fields[1], float(fields[2])) for fields in map(str.split, filter(None, lines))]
    width, height, left, bottom, top = 720, 210, 34, 26, 12
    plot = height - bottom - top
    scale = plot / 8
    step = (width - left) / len(savings)
    parts = []
    for tick in range(0, 9, 2):
        y = height - bottom - tick * scale
        parts.append(f'<line class="grid" x1="{left}" x2="{width}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick}%</text>')
    for index, (stem, saving) in enumerate(savings):
        x = left + index * step + step * 0.14
        bar = saving * scale
        parts.append(
            f'<rect class="saving" x="{x:.1f}" y="{height - bottom - bar:.1f}" width="{step * 0.72:.1f}" height="{bar:.1f}">'
            f"<title>Image {stem}: {saving:.2f}% saved</title></rect>"
        )
    parts.append(f'<text class="tick" x="{left}" y="{height - 6}">Each bar is one photo, smallest saving first</text>')
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Truepix saved between '
        f'{savings[0][1]:.2f} and {savings[-1][1]:.2f} percent against JPEG XL effort 10 on each of 24 photos.">'
        + "".join(parts) + "</svg>"
    ), savings


def speed_rows():
    rows = []
    for name, encode, decode, size in SPEED:
        ours = ' class="ours"' if name == "Truepix" else ""
        rows.append(f"<tr{ours}><th scope=\"row\">{name}</th><td>{encode}</td><td>{decode}</td><td>{size}</td></tr>")
    return "\n".join(rows)


chart, savings = savings_chart()
page = (HERE / "template.html").read_text(encoding="utf-8")
for key, value in {
    "@@KODAK@@": mask("lattice_kodak.png"),
    "@@ORDINARY@@": mask("lattice_ordinary.png"),
    "@@METHODS@@": method_rows(),
    "@@SAVINGS@@": chart,
    "@@LOW@@": f"{savings[0][1]:.2f}",
    "@@HIGH@@": f"{savings[-1][1]:.2f}",
    "@@SPEED@@": speed_rows(),
}.items():
    page = page.replace(key, value)
assert "@@" not in page

(HERE / "index.html").write_text(page, encoding="utf-8")
shutil.copyfile(PAPER / "paper.pdf", HERE / "paper.pdf")

if len(sys.argv) > 1:  # A copy without the document tags, which the Artifact tool adds itself.
    body = page[page.index("<title>"):].replace("</head>\n<body>\n", "").replace("</body>\n</html>\n", "")
    Path(sys.argv[1]).write_text(body, encoding="utf-8")

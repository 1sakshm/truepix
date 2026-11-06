"""Draw the Truepix logo: a T of exact square pixels, with the last one in tomato."""

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
SUN, INK, TOMATO = "#ffc400", "#17110b", "#ff4a1c"
CELL, GAP, MARGIN, SIZE = 9, 1, 5.5, 64
# Column and row of each pixel in a six by six grid. The last is the tomato one.
PIXELS = [(x, y) for y in (0, 1) for x in range(6)] + [(x, y) for y in (2, 3, 4) for x in (2, 3)] + [(2, 5), (3, 5)]


def rects(ink):
    out = []
    for index, (x, y) in enumerate(PIXELS):
        fill = TOMATO if index == len(PIXELS) - 1 else ink
        out.append(f'<rect x="{MARGIN + x * CELL}" y="{MARGIN + y * CELL}" width="{CELL - GAP}" height="{CELL - GAP}" fill="{fill}"/>')
    return "".join(out)


def svg(background):
    plate = f'<rect width="{SIZE}" height="{SIZE}" fill="{background}"/>' if background else ""
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" role="img" aria-label="Truepix">{plate}{rects(INK)}</svg>\n'


def png(path, size):
    scale = 8
    image = Image.new("RGB", (size * scale, size * scale), SUN)
    draw = ImageDraw.Draw(image)
    unit = size * scale / SIZE
    for index, (x, y) in enumerate(PIXELS):
        left, top = (MARGIN + x * CELL) * unit, (MARGIN + y * CELL) * unit
        side = (CELL - GAP) * unit
        draw.rectangle((left, top, left + side - 1, top + side - 1), fill=TOMATO if index == len(PIXELS) - 1 else INK)
    image.resize((size, size), Image.LANCZOS).save(path)


(HERE / "favicon.svg").write_text(svg(SUN), encoding="utf-8")
(HERE / "logo.svg").write_text(svg(None), encoding="utf-8")
png(HERE / "logo-512.png", 512)
png(HERE / "apple-touch-icon.png", 180)
png(HERE / "favicon-32.png", 32)

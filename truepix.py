"""Compress images into Truepix files and restore them exactly."""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

MAGIC = b"TPX1"
HEADER = struct.Struct("<4sIIBBI")  # magic, width, height, channels, tiles per side, CRC-32
MODES = ("L", "LA", "RGB", "RGBA")


class TruepixError(Exception):
    """A problem with the input that the user can act on."""


def core_binary():
    name = "truepix-core.exe" if os.name == "nt" else "truepix-core"
    binary = Path(os.environ.get("TRUEPIX_CORE") or Path(__file__).parent / name)
    if not binary.is_file():
        raise TruepixError(f"the codec was not found at {binary}; see README.md to build it")
    return binary


def run_core(*arguments):
    done = subprocess.run([str(core_binary()), *map(str, arguments)], capture_output=True, text=True)
    if done.returncode:
        raise TruepixError(f"the codec failed: {done.stderr.strip()}")


def load_pixels(path):
    """Read an image as 8-bit pixels, refusing anything that would lose information."""
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) > 1:
                raise TruepixError(f"{path} is animated, which Truepix does not store")
            if image.mode == "P":
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            elif image.mode == "1":
                image = image.convert("L")
            if image.mode not in MODES:
                raise TruepixError(f"{path} uses mode {image.mode}; Truepix stores 8-bit grey or RGB, with or without alpha")
            pixels = np.array(image)
    except (OSError, UnidentifiedImageError, SyntaxError, ValueError) as error:
        raise TruepixError(f"{path} is not a readable image") from error
    return pixels if pixels.ndim == 3 else pixels[..., None]


def compress_pixels(pixels, output):
    height, width, channels = pixels.shape
    header = f"P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH {channels}\nMAXVAL 255\nTUPLTYPE X\nENDHDR\n"
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "in.pam"
        raw.write_bytes(header.encode() + pixels.tobytes())
        run_core("c", raw, output)


def decompress_pixels(path):
    data = Path(path).read_bytes()
    if len(data) < HEADER.size or data[:4] != MAGIC:
        raise TruepixError(f"{path} is not a Truepix file")
    _, width, height, channels, _, _ = HEADER.unpack_from(data)
    if not 1 <= channels <= len(MODES):
        raise TruepixError(f"{path} is damaged")
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "out.pnm"
        run_core("d", path, raw)
        body = raw.read_bytes()[-(width * height * channels):]
    return np.frombuffer(body, dtype=np.uint8).reshape(height, width, channels)


def save_png(pixels, output):
    channels = pixels.shape[-1]
    shaped = pixels[..., 0] if channels == 1 else pixels
    Image.fromarray(shaped, MODES[channels - 1]).save(output, format="PNG")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="truepix", description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    compress = actions.add_parser("compress", help="turn an image into a .tpix file")
    compress.add_argument("input", type=Path)
    compress.add_argument("-o", "--output", type=Path)
    compress.add_argument("--verify", action="store_true", help="decode the result and compare every pixel")
    restore = actions.add_parser("decompress", help="turn a .tpix file into a PNG")
    restore.add_argument("input", type=Path)
    restore.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.input.with_suffix(".tpix" if args.action == "compress" else ".png")
    try:
        if output.exists():
            raise TruepixError(f"{output} already exists")
        if args.action == "compress":
            pixels = load_pixels(args.input)
            compress_pixels(pixels, output)
            if args.verify and not np.array_equal(decompress_pixels(output), pixels):
                output.unlink()
                raise TruepixError("the check failed, so nothing was written")
            print(f"{args.input} -> {output}: {args.input.stat().st_size:,} -> {output.stat().st_size:,} bytes")
        else:
            save_png(decompress_pixels(args.input), output)
            print(f"{args.input} -> {output}")
    except TruepixError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

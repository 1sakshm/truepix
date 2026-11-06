# Truepix

A lossless image codec for archiving. Truepix files are smaller than JPEG XL at its highest
effort, and the cost is slow decoding. Every pixel comes back exactly.

| Final test: 24 unseen DIV2K photos | Total size | Compared with Truepix |
|---|---|---|
| Truepix | 69.8 MB | |
| JPEG XL effort 10 | | 3.62% larger |
| FLIF | | 9.28% larger |
| WebP lossless | | 16.48% larger |
| PNG | | 58.22% larger |

Truepix won on all 24 images. Two earlier test sets gave leads of 4.28% and 4.29%. The full
method and results are in [`paper/paper.pdf`](paper/paper.pdf).

## Usage

Build the codec, then use the command-line tool:

```
g++ -std=c++17 -O3 -mavx2 -ffp-contract=off -static truepix-core.cpp -o truepix-core.exe

uv run truepix compress photo.png            # writes photo.tpix
uv run truepix compress photo.png --verify   # also decodes it and compares every pixel
uv run truepix decompress photo.tpix         # writes photo.png
```

On Linux or macOS, build without `-static` and with `-pthread`, and name the file
`truepix-core`. Without AVX2, leave out `-mavx2`; the files are the same, but it runs slower. Keep
`-ffp-contract=off`: without it, files may not decode on other builds. Set `TRUEPIX_CORE` to use
a codec stored somewhere else.

## Limits

- **Speed:** a 3-megapixel photo takes about 12 seconds to encode and 7 to decode on an 8-core
  laptop, about 100 times slower to decode than JPEG XL. It suits archives, not web pages. The
  paper's timings came from an earlier build (17 and 11 seconds) that wrote the same files.
- **Pixels only:** colour profiles and other metadata are not stored.
- **Formats:** 8-bit grey or RGB, with or without alpha. Other images are refused, not converted.
- **File format:** not stable yet, so keep your originals until it is.
- **paq8px** compresses further, and is far slower.
- **Testing:** files were byte-identical across GCC, Clang, 32-bit x86 and WebAssembly on x86-64
  Windows. ARM64 compiles but has never been run.
- **Not independently replicated.** The raw results and scripts are in `paper/`. They use the
  working name PXM.

## Tests

```
uv run pytest
```

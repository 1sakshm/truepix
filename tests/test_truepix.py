import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import truepix


def photo(width, height, channels):
    y, x = np.mgrid[:height, :width]
    bands = [(x * (i + 1) // 3 + y * 2 + i * 40) % 256 for i in range(channels)]
    return np.stack(bands, axis=-1).astype(np.uint8)


def save(path, array):
    shaped = array[..., 0] if array.shape[-1] == 1 else array
    Image.fromarray(shaped).save(path)
    return path


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_round_trip_keeps_every_pixel(tmp_path, channels):
    array = photo(64, 48, channels)
    source = save(tmp_path / "a.png", array)
    assert truepix.main(["compress", str(source), "--verify"]) == 0
    assert truepix.main(["decompress", str(tmp_path / "a.tpix"), "-o", str(tmp_path / "b.png")]) == 0
    restored = np.asarray(Image.open(tmp_path / "b.png"))
    assert np.array_equal(restored, array[..., 0] if channels == 1 else array)


def test_files_match_the_scored_build(tmp_path):
    # Speed changes must not move a single byte, or the published results stop describing the codec.
    truepix.compress_pixels(photo(64, 48, 3), tmp_path / "a.tpix")
    assert hashlib.sha256((tmp_path / "a.tpix").read_bytes()).hexdigest() == "2878d8c174d70fc23bb141b3fd1c5909bb80c5b8fb223be5654e5ef386a7d1b7"


def save_jpeg(path, **options):
    Image.fromarray(photo(96, 64, 3)).save(path, quality=85, **options)
    return path


@pytest.mark.parametrize("options", [{}, {"subsampling": 0}, {"progressive": True}, {"optimize": True}, {"progressive": True, "optimize": True}])
def test_jpeg_comes_back_byte_for_byte(tmp_path, options):
    source = save_jpeg(tmp_path / "a.jpg", **options)
    assert truepix.main(["compress", str(source)]) == 0
    assert (tmp_path / "a.tpix").read_bytes()[:4] == truepix.JPEG_MAGIC
    assert truepix.main(["decompress", str(tmp_path / "a.tpix"), "-o", str(tmp_path / "b.jpg")]) == 0
    assert (tmp_path / "b.jpg").read_bytes() == source.read_bytes()


def test_decompress_names_a_jpeg_correctly(tmp_path):
    source = save_jpeg(tmp_path / "a.jpg")
    packed = tmp_path / "packed" / "a.tpix"
    packed.parent.mkdir()
    assert truepix.main(["compress", str(source), "-o", str(packed)]) == 0
    assert truepix.main(["decompress", str(packed)]) == 0
    assert (packed.parent / "a.jpg").read_bytes() == source.read_bytes()


def test_cut_jpeg_falls_back_to_pixels(tmp_path):
    good = Image.fromarray(photo(256, 256, 3))
    good.save(tmp_path / "a.jpg", quality=85)
    data = (tmp_path / "a.jpg").read_bytes()
    Image.open(tmp_path / "a.jpg").load()
    (tmp_path / "cut.jpg").write_bytes(data + b"")
    assert truepix.compress_jpeg(tmp_path / "cut.jpg", tmp_path / "ok.tpix") is True
    (tmp_path / "cut.jpg").write_bytes(data[: len(data) // 2])
    assert truepix.compress_jpeg(tmp_path / "cut.jpg", tmp_path / "no.tpix") is False


def test_damaged_jpeg_file_is_refused(tmp_path):
    source = save_jpeg(tmp_path / "a.jpg")
    packed = tmp_path / "a.tpix"
    assert truepix.main(["compress", str(source)]) == 0
    data = bytearray(packed.read_bytes())
    data[len(data) // 2] ^= 0xFF
    packed.write_bytes(data)
    assert truepix.main(["decompress", str(packed), "-o", str(tmp_path / "b.jpg")]) == 1


def test_files_start_with_the_magic(tmp_path):
    truepix.compress_pixels(photo(32, 32, 3), tmp_path / "a.tpix")
    assert (tmp_path / "a.tpix").read_bytes()[:4] == truepix.MAGIC


def test_palette_image_keeps_its_colours(tmp_path):
    source = tmp_path / "a.png"
    Image.fromarray(photo(32, 32, 3)).quantize(16).save(source)
    expected = np.asarray(Image.open(source).convert("RGB"))
    assert np.array_equal(truepix.load_pixels(source), expected)


def test_sixteen_bit_image_is_refused(tmp_path):
    source = tmp_path / "a.png"
    Image.fromarray(np.arange(400, dtype=np.uint16).reshape(20, 20)).save(source)
    with pytest.raises(truepix.TruepixError, match="8-bit"):
        truepix.load_pixels(source)


def test_animation_is_refused(tmp_path):
    frames = [Image.fromarray(np.full((8, 8), value, dtype=np.uint8)) for value in (0, 255)]
    frames[0].save(tmp_path / "a.gif", save_all=True, append_images=frames[1:])
    with pytest.raises(truepix.TruepixError, match="animated"):
        truepix.load_pixels(tmp_path / "a.gif")


def test_not_an_image_is_refused(tmp_path):
    source = tmp_path / "a.png"
    source.write_bytes(b"plain text")
    assert truepix.main(["compress", str(source)]) == 1


def test_not_a_truepix_file_is_refused(tmp_path):
    source = tmp_path / "a.tpix"
    source.write_bytes(b"plain text that is long enough for a header")
    assert truepix.main(["decompress", str(source)]) == 1


def test_damaged_file_is_refused(tmp_path):
    packed = tmp_path / "a.tpix"
    truepix.compress_pixels(photo(64, 64, 3), packed)
    data = bytearray(packed.read_bytes())
    data[len(data) // 2] ^= 0xFF
    packed.write_bytes(data)
    assert truepix.main(["decompress", str(packed)]) == 1


def test_existing_output_is_not_overwritten(tmp_path):
    source = save(tmp_path / "a.png", photo(32, 32, 3))
    (tmp_path / "a.tpix").write_bytes(b"keep")
    assert truepix.main(["compress", str(source)]) == 1
    assert (tmp_path / "a.tpix").read_bytes() == b"keep"

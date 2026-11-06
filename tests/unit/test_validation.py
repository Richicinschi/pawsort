import io

import pytest
from PIL import Image, ImageFile

from interface.validation import MAX_BYTES, UploadError, check_site_code, validate_upload


@pytest.fixture(autouse=True)
def strict_pillow(monkeypatch):
    """Importing MegaDetector makes Pillow accept cut off files. The website runs without it, in its own process."""
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", False)


def image_bytes(fmt: str, size: tuple[int, int] = (200, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (0, 128, 0)).save(buf, format=fmt)
    return buf.getvalue()


def test_accepts_jpeg_and_png():
    assert validate_upload(image_bytes("JPEG")) == "JPEG"
    assert validate_upload(image_bytes("PNG")) == "PNG"


def test_accepts_a_file_of_exactly_10_mb():
    data = image_bytes("JPEG")
    assert validate_upload(data + b"\0" * (MAX_BYTES - len(data))) == "JPEG"


BAD_FILES = {
    "empty": (lambda: b"", "empty"),
    "over-10-mb": (lambda: b"x" * (MAX_BYTES + 1), "larger than 10 MB"),
    "gif": (lambda: image_bytes("GIF"), "Only JPEG and PNG"),
    "webp": (lambda: image_bytes("WEBP"), "Only JPEG and PNG"),
    "text-renamed-jpg": (lambda: b"just some text, renamed to .jpg", "cannot be opened"),
    "cut-off-jpeg": (lambda: image_bytes("JPEG")[:400], "cannot be opened"),
    "cut-off-png": (lambda: image_bytes("PNG")[:60], "cannot be opened"),
}


@pytest.mark.parametrize("case", BAD_FILES)
def test_rejects_bad_files(case):
    make_data, message = BAD_FILES[case]
    with pytest.raises(UploadError, match=message):
        validate_upload(make_data())


@pytest.mark.parametrize("size", [(127, 127), (127, 200), (200, 127)])
def test_rejects_too_small(size):
    with pytest.raises(UploadError, match="at least 128 x 128"):
        validate_upload(image_bytes("JPEG", size))


@pytest.mark.parametrize("size", [(6001, 6001), (6001, 200), (200, 6001)])
def test_rejects_too_large(size):
    with pytest.raises(UploadError, match="at most 6000 x 6000"):
        validate_upload(image_bytes("JPEG", size))


@pytest.mark.parametrize("size", [(128, 128), (6000, 6000)])
def test_accepts_the_exact_limits(size):
    assert validate_upload(image_bytes("JPEG", size)) == "JPEG"


def test_rejects_a_giant_image_with_a_tiny_file():
    buf = io.BytesIO()
    Image.new("1", (20000, 20000)).save(buf, format="PNG")
    assert len(buf.getvalue()) < MAX_BYTES
    with pytest.raises(UploadError, match="at most 6000 x 6000"):
        validate_upload(buf.getvalue())


@pytest.mark.parametrize("code", ["A12", "site-7", "x"])
def test_accepts_a_site_code(code):
    check_site_code(code)


@pytest.mark.parametrize("code", ["", "two words", "52.08,5.83", "a" * 21])
def test_rejects_a_bad_site_code(code):
    with pytest.raises(UploadError, match="camera site code"):
        check_site_code(code)

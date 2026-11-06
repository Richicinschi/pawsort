"""Checks an upload before anything else happens (FR-02, FR-03)."""
import io
import re

from PIL import Image

MAX_BYTES = 10 * 1024 * 1024
MIN_SIDE = 128
MAX_SIDE = 6000
ALLOWED = {"JPEG", "PNG"}
TOO_LARGE = "The photo is too large, it can be at most 6000 x 6000 pixels."


class UploadError(Exception):
    pass


def check_site_code(code: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9-]{1,20}", code):
        raise UploadError("Please enter one camera site code: at most 20 letters, numbers or dashes, for example A12.")


def check_resolution(width: int, height: int) -> None:
    if width < MIN_SIDE or height < MIN_SIDE:
        raise UploadError("The photo is too small, it must be at least 128 x 128 pixels.")
    if width > MAX_SIDE or height > MAX_SIDE:
        raise UploadError(TOO_LARGE)


def validate_upload(data: bytes) -> str:
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise UploadError("The file is larger than 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ALLOWED:
                raise UploadError("Only JPEG and PNG photos are accepted.")
            check_resolution(*image.size)
            image.load()  # reads every pixel, so a cut off file gets caught here
            return image.format
    except Image.DecompressionBombError:
        raise UploadError(TOO_LARGE)
    except (OSError, SyntaxError, ValueError):
        raise UploadError("This file cannot be opened as a photo.")

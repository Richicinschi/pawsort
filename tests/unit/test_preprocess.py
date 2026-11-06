from PIL import Image

from ai_server.model import preprocess


def test_output_shape():
    assert tuple(preprocess(Image.new("RGB", (300, 200), "red")).shape) == (3, 224, 224)


def test_grey_night_photo_gets_three_channels():
    assert tuple(preprocess(Image.new("L", (50, 50), 128)).shape) == (3, 224, 224)

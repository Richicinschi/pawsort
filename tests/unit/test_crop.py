from PIL import Image

from ai_server.model import crop_to_box


def test_crop_adds_a_margin():
    assert crop_to_box(Image.new("RGB", (1000, 1000)), [0.4, 0.4, 0.2, 0.2]).size == (240, 240)


def test_crop_stays_inside_the_photo():
    assert crop_to_box(Image.new("RGB", (1000, 1000)), [0.0, 0.0, 1.0, 1.0]).size == (1000, 1000)


def test_tiny_box_still_gives_a_picture():
    width, height = crop_to_box(Image.new("RGB", (100, 100)), [0.5, 0.5, 0.0, 0.0]).size
    assert width >= 1 and height >= 1


import pytest

from assessment.severity import estimate_severity


IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100


def test_small_bounding_box_is_low():
    assert estimate_severity(
        [10, 10, 20, 20], IMAGE_WIDTH, IMAGE_HEIGHT
    ) == "LOW"


def test_medium_bounding_box_is_medium():
    assert estimate_severity(
        [10, 10, 30, 30], IMAGE_WIDTH, IMAGE_HEIGHT
    ) == "MEDIUM"


def test_large_bounding_box_is_high():
    assert estimate_severity(
        [10, 10, 50, 50], IMAGE_WIDTH, IMAGE_HEIGHT
    ) == "HIGH"


def test_exactly_two_percent_is_low():
    assert estimate_severity(
        [0, 0, 10, 20], IMAGE_WIDTH, IMAGE_HEIGHT
    ) == "LOW"


def test_exactly_ten_percent_is_medium():
    assert estimate_severity(
        [0, 0, 20, 50], IMAGE_WIDTH, IMAGE_HEIGHT
    ) == "MEDIUM"


@pytest.mark.parametrize(
    "bbox,width,height",
    [
        ([0, 0, 0, 10], 100, 100),
        ([0, 0, 101, 10], 100, 100),
        ([0, 0, 10, 101], 100, 100),
        ([10, 10, 5, 20], 100, 100),
        ([0, 0, 10], 100, 100),
        ([0, 0, "10", 20], 100, 100),
    ],
)
def test_invalid_bounding_box_raises_value_error(bbox, width, height):
    with pytest.raises(ValueError):
        estimate_severity(bbox, width, height)


@pytest.mark.parametrize(
    "width,height",
    [
        (0, 100),
        (100, 0),
        (-1, 100),
        (100, -1),
        (100.5, 100),
    ],
)
def test_invalid_image_dimensions_raise_value_error(width, height):
    with pytest.raises(ValueError):
        estimate_severity([0, 0, 10, 10], width, height)

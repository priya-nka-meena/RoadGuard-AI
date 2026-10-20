
from unittest.mock import MagicMock, patch

from common.location import extract_gps


def test_image_without_gps_returns_none():
    assert extract_gps("data/road.jpg") is None


def test_missing_image_returns_none():
    assert extract_gps("data/nonexistent.jpg") is None


def make_mock_image(gps_data):
    image = MagicMock()
    image.__enter__.return_value = image
    image.__exit__.return_value = False

    exif = MagicMock()
    exif.__contains__.side_effect = lambda tag: tag == 34853
    exif.get_ifd.return_value = gps_data
    image.getexif.return_value = exif

    return image


def test_extracts_north_east_coordinates():
    gps_data = {
        1: "N",
        2: (28, 36, 0),
        3: "E",
        4: (77, 12, 0),
    }

    mock_image = make_mock_image(gps_data)

    with patch("common.location.Image.open", return_value=mock_image):
        result = extract_gps("sample.jpg")

    assert result == {
        "latitude": 28.6,
        "longitude": 77.2,
    }


def test_extracts_south_west_coordinates():
    gps_data = {
        1: "S",
        2: (33, 51, 0),
        3: "W",
        4: (151, 12, 0),
    }

    mock_image = make_mock_image(gps_data)

    with patch("common.location.Image.open", return_value=mock_image):
        result = extract_gps("sample.jpg")

    assert result == {
        "latitude": -33.85,
        "longitude": -151.2,
    }

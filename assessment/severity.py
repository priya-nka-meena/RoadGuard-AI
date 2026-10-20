
"""Visual severity estimation for detected potholes.

Severity is estimated from bounding-box area relative to image area.
This is a visual proxy, not a measurement of actual pothole depth or danger.
"""

from numbers import Real


# Initial configurable thresholds for visual area ratio.
LOW_MAX_RATIO = 0.02
MEDIUM_MAX_RATIO = 0.10


def estimate_severity(bbox, image_width, image_height):
    """Estimate pothole severity from its bounding box and image dimensions.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] pixel coordinates.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.

    Returns:
        "LOW", "MEDIUM", or "HIGH".

    Raises:
        ValueError: If dimensions or bounding-box coordinates are invalid.
    """
    if (
        isinstance(image_width, bool)
        or not isinstance(image_width, Real)
        or image_width <= 0
        or isinstance(image_height, bool)
        or not isinstance(image_height, Real)
        or image_height <= 0
        or not float(image_width).is_integer()
        or not float(image_height).is_integer()
    ):
        raise ValueError("Image dimensions must be positive integers.")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("Bounding box must contain four coordinates.")

    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in bbox
    ):
        raise ValueError("Bounding-box coordinates must be numeric.")

    x1, y1, x2, y2 = bbox

    if any(not float(value).is_integer() for value in bbox):
        raise ValueError("Bounding-box coordinates must be integers.")

    if not (
        0 <= x1 < x2 <= image_width
        and 0 <= y1 < y2 <= image_height
    ):
        raise ValueError("Bounding box must be valid and within the image.")

    box_area = (x2 - x1) * (y2 - y1)
    image_area = image_width * image_height
    area_ratio = box_area / image_area

    if area_ratio <= LOW_MAX_RATIO:
        return "LOW"
    if area_ratio <= MEDIUM_MAX_RATIO:
        return "MEDIUM"
    return "HIGH"

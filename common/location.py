
"""Extract GPS coordinates from image EXIF metadata."""

from pathlib import Path
from PIL import Image


def extract_gps(image_path):
    """Return image GPS coordinates as {'latitude': ..., 'longitude': ...}.

    Returns None if GPS metadata is missing or invalid.
    Coordinates identify the camera location, not the exact pothole location.
    """
    try:
        with Image.open(Path(image_path)) as image:
            exif = image.getexif()
            gps = exif.get_ifd(34853) if 34853 in exif else {}

            if not gps:
                return None

            latitude = gps.get(2)
            latitude_ref = gps.get(1)
            longitude = gps.get(4)
            longitude_ref = gps.get(3)

            if (
                latitude is None
                or latitude_ref is None
                or longitude is None
                or longitude_ref is None
            ):
                return None

            def to_decimal(dms):
                degrees, minutes, seconds = (
                    float(value) for value in dms
                )
                return degrees + minutes / 60 + seconds / 3600

            lat = to_decimal(latitude)
            lon = to_decimal(longitude)

            if str(latitude_ref).upper() == "S":
                lat = -lat
            elif str(latitude_ref).upper() != "N":
                return None

            if str(longitude_ref).upper() == "W":
                lon = -lon
            elif str(longitude_ref).upper() != "E":
                return None

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return None

            return {"latitude": lat, "longitude": lon}

    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError):
        return None

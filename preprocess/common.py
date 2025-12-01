# preprocess/common.py
import base64
import io
from typing import Tuple
from PIL import Image


def decode_base64_image(data: str) -> Image.Image:
    """
    Accepts a base64 string, optionally starting with
    'data:image/png;base64,...', and returns a RGB PIL.Image.
    """
    # Handle "data:image/png;base64,XXXX" format
    if data.startswith("data:image"):
        try:
            header, data = data.split(",", 1)
        except ValueError:
            # malformed header, but we'll just try decode anyway
            pass

    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img

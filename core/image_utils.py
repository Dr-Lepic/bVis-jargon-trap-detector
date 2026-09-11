"""Image processing utilities for the Med-Jargon Benchmark.

Handles high-quality downsampling (capped at 768px Lanczos) and base64 encoding
for multimodal vision-language model API consumption.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Union
from PIL import Image

ImageInputType = Union[str, Path, bytes, io.BytesIO, Image.Image]


def resize_and_encode_image(
    image_input: ImageInputType,
    max_dim: int = 768,
    jpeg_quality: int = 88,
) -> str:
    """Accepts a PIL Image, raw bytes, BytesIO, or file path.

    Downsamples using Lanczos filter so maximum dimension does not exceed max_dim,
    preserving aspect ratio. Converts image to RGB and returns a base64-encoded JPEG string.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError(
            f"Unsupported image input type: {type(image_input)}. "
            f"Expected PIL Image, bytes, BytesIO, or file path string/Path."
        )

    # Downsample if larger than max_dim in either dimension
    if img.width > max_dim or img.height > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_str: str) -> Image.Image:
    """Decodes a base64 string back into a PIL Image in RGB format."""
    # Strip optional data URI prefix if present
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]

    raw_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def get_image_dimensions(image_input: ImageInputType) -> tuple[int, int]:
    """Returns (width, height) of the provided image."""
    if isinstance(image_input, Image.Image):
        return image_input.size

    if isinstance(image_input, (str, Path)):
        with Image.open(str(image_input)) as img:
            return img.size
    elif isinstance(image_input, bytes):
        with Image.open(io.BytesIO(image_input)) as img:
            return img.size
    elif isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        with Image.open(image_input) as img:
            return img.size
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

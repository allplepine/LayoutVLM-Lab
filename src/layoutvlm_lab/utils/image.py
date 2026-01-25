"""Image processing utilities."""

import base64
import io
from typing import List, Optional
from PIL import Image


def crop_by_boxes(image_path: str, boxes: List[List[int]]) -> List[Image.Image]:
    """Crop regions from an image by bounding boxes.
    
    Args:
        image_path: Path to the image
        boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        
    Returns:
        List of cropped PIL Images
    """
    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
    except Exception:
        return []
    
    cropped_images = []
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = map(int, box)
        if x1 >= x2 or y1 >= y2:
            continue
        cropped = img.crop((x1, y1, x2, y2))
        cropped_images.append(cropped)
    
    return cropped_images


def pil_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string.
    
    Args:
        img: PIL Image
        format: Image format (JPEG, PNG, etc.)
        
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

"""Base class for VLM (Vision Language Model)."""

from abc import ABC, abstractmethod
from PIL import Image
from ..core.registry import VLM_REGISTRY


class BaseVLM(ABC):
    """Abstract base class for Vision Language Models."""
    
    @abstractmethod
    def recognize(self, image: Image.Image, prompt_type: str) -> str:
        """Recognize content from an image.
        
        Args:
            image: PIL Image object
            prompt_type: Type of recognition ("ocr", "table", "formula")
            
        Returns:
            str: Recognized content
        """
        pass

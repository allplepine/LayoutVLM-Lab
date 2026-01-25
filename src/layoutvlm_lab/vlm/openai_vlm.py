"""OpenAI-compatible VLM implementation."""

import io
import base64
from typing import Dict, Any
from PIL import Image
from openai import OpenAI
from .base import BaseVLM, VLM_REGISTRY
from .prompts import get_prompt


@VLM_REGISTRY.register("openai")
class OpenAIVLM(BaseVLM):
    """VLM using OpenAI-compatible API (works with vLLM, etc.)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI-compatible VLM.
        
        Args:
            config: Configuration dictionary (from YAML vlm.config)
        """
        api_key = config.get("api_key", "").strip()
        base_url = config.get("base_url", "").strip()
        
        if not api_key or not base_url:
            raise ValueError("OpenAIVLM requires 'api_key' and 'base_url' in config")
            
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = config.get("model_name")
        self.text_before_image = config.get("text_before_image", True)
        
        # Filter out internal config keys, pass everything else to OpenAI API
        internal_keys = {
            "api_key", "base_url", "model_name", "text_before_image",
            "type"
        }
        self.base_kwargs = {
            k: v for k, v in config.items() 
            if k not in internal_keys
        }
        
        # Set defaults if not provided (but do not force them if user wants something else)
        self.base_kwargs.setdefault("max_tokens", 20000)
        self.base_kwargs.setdefault("temperature", 0.0)
        self.base_kwargs.setdefault("timeout", 6000)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    
    def recognize(self, image: Image.Image, prompt_type: str) -> str:
        """Recognize content from an image using VLM."""
        prompt = get_prompt(prompt_type)
        image_url = self._image_to_base64(image)
        
        content = []
        if self.text_before_image:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        else:
            content = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt}
            ]

        messages = [{"role": "user", "content": content}]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.base_kwargs
        )
        
        return response.choices[0].message.content

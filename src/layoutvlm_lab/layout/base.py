# layout/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import os
import json
from ..core.registry import LAYOUT_REGISTRY

class BaseLayoutDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> Dict[str, Any]:
        """Run layout detection on an image."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def post_process(content: str, label: str, layout_config: Dict[str, Any]) -> str:
        """Post-process VLM output based on label and layout config."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def json2md(json_data: Dict[str, Any], layout_config: Dict[str, Any]) -> str:
        """Convert Layout result JSON to Markdown based on layout config."""
        raise NotImplementedError

    @staticmethod
    def save_json(filepath: str, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_md(filepath: str, content: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

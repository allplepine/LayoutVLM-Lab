"""Layout detection modules."""

from .base import BaseLayoutDetector
from .paddle_layout import PaddleLayoutDetector

__all__ = ["BaseLayoutDetector", "PaddleLayoutDetector"]

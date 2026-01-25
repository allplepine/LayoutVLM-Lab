"""Prompts for VLM recognition."""

from .ocr import OCR_PROMPT
from .table import TABLE_PROMPT
from .formula import FORMULA_PROMPT


PROMPTS = {
    "ocr": OCR_PROMPT,
    "table": TABLE_PROMPT,
    "formula": FORMULA_PROMPT,
}


def get_prompt(prompt_type: str) -> str:
    """Get prompt by type."""
    if prompt_type not in PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}, available: {list(PROMPTS.keys())}")
    return PROMPTS[prompt_type]


__all__ = ["get_prompt", "OCR_PROMPT", "TABLE_PROMPT", "FORMULA_PROMPT"]

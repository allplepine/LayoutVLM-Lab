# Add a Custom Layout

This guide shows how to add a new Layout detector to LayoutVLM-Lab.

## 1) Create a new layout module

Create a new file, for example:

`src/layoutvlm_lab/layout/my_layout.py`

```python
from typing import Dict, Any
from .base import BaseLayoutDetector, LAYOUT_REGISTRY


@LAYOUT_REGISTRY.register("my_layout")
class MyLayoutDetector(BaseLayoutDetector):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def detect(self, image_path: str) -> Dict[str, Any]:
        # TODO: run your model and return a dict with:
        # {
        #   "input_path": image_path,
        #   "parsing_res_list": [
        #     {"block_label": "text", "block_bbox": [x1,y1,x2,y2], "block_content": "", "block_id": 0},
        #     ...
        #   ]
        # }
        raise NotImplementedError

    @staticmethod
    def post_process(content: str, label: str, layout_config: Dict[str, Any]) -> str:
        # Optional post-processing for VLM output
        return content

    @staticmethod
    def json2md(json_data: Dict[str, Any], layout_config: Dict[str, Any]) -> str:
        # Convert layout result to Markdown
        raise NotImplementedError
```

## 2) Make sure the module is imported

The registry only contains classes that have been imported. Add your class to:

`src/layoutvlm_lab/layout/__init__.py`

```python
from .my_layout import MyLayoutDetector
```

## 3) Update config

Set the layout type in your config:

```yaml
layout:
  type: "my_layout"
  config:
    # your layout config here
```

## 4) Run

```bash
python run.py -c config/your_config.yaml
```

## Notes
- The framework normalizes layout labels into four categories: `text`, `table`, `formula`, and `image`.
- Different layout models use different label names, so you need to use `label_mapping` in the configuration to map them into the unified categories. The VLM will then use the mapped category (e.g., `text`) to select the corresponding category-specific prompt.

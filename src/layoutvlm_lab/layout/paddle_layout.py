"""PP-DocLayoutV2 (PaddleOCR) layout detector implementation."""

from typing import Dict, Any, Optional, List
from .base import BaseLayoutDetector, LAYOUT_REGISTRY

@LAYOUT_REGISTRY.register("paddle")
class PaddleLayoutDetector(BaseLayoutDetector):
    """Layout detector using PaddleOCR's LayoutDetection.
    
    Wraps PP-DocLayoutV2 or other Paddle layout models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PP-DocLayoutV2 detector.
        
        Args:
            config: Configuration dictionary (from YAML layout.config)
        """
        self.model_name = config.get("model_name", "PP-DocLayoutV2")
        self.model_dir = config.get("model_dir")
        self.device = config.get("device")
        # Add merge_blocks_before_md attribute from config
        self.merge_blocks_before_md = config.get("merge_blocks_before_md", False)
        # Initialize label mapping
        # These are defaults for PP-DocLayoutV2, overridden by config if provided
        self.model = None
        from paddlex.inference.common.reader import ReadImage
        self.reader = ReadImage(format="BGR")
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """Run layout detection on an image."""
        if self.model is None:
            from ..utils.paddle_layout_utils import create_predictor
            kwargs = {"model_name": self.model_name}
            if self.model_dir:
                kwargs["model_dir"] = self.model_dir
            if self.device:
                kwargs["device"] = self.device
            kwargs["batch_size"] = 8
            kwargs["use_hpip"] = False
            kwargs["hpi_config"] = None
            kwargs["genai_config"] = {'backend': 'native'}
            default_settings = {'threshold': {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.4, 6: 0.4, 7: 0.5, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5, 12: 0.5, 13: 0.5, 14: 0.5, 15: 0.4, 16: 0.5, 17: 0.4, 18: 0.5, 19: 0.5, 20: 0.45, 21: 0.5, 22: 0.4, 23: 0.4, 24: 0.5}, 'layout_nms': True, 'layout_unclip_ratio': [1.0, 1.0], 'layout_merge_bboxes_mode': {0: 'union', 1: 'union', 2: 'union', 3: 'large', 4: 'union', 5: 'large', 6: 'large', 7: 'union', 8: 'union', 9: 'union', 10: 'union', 11: 'union', 12: 'union', 13: 'union', 14: 'union', 15: 'large', 16: 'union', 17: 'large', 18: 'union', 19: 'union', 20: 'union', 21: 'union', 22: 'union', 23: 'union', 24: 'union'}}
            for k,v in default_settings.items():
                kwargs[k] = v
            self.model = create_predictor(**kwargs)
        img = self.reader([image_path])
        output = list(self.model(img, layout_nms=None,threshold=None,
                                    layout_unclip_ratio=None,layout_merge_bboxes_mode=None))[0]
        from ..utils.paddle_layout_utils import convert_to_parsing_format, filter_overlap_boxes
        output = filter_overlap_boxes(output)
        output = convert_to_parsing_format(output, image_path)
        return output
    
    @staticmethod
    def post_process(content: str, label: str, layout_config: Dict[str, Any]) -> str:
        """Post-process VLM output."""
        from ..utils.paddle_layout_utils import (
            remove_markdown_fences, 
            normalized_latex_table, 
            normalized_html_table, 
            compress_html_table, 
            truncate_repetitive_content
        )
        
        result_str = content
        result_str = remove_markdown_fences(result_str)

        if label == "table":
            if r'\begin{tabular}' in result_str:
                result_str = normalized_latex_table(result_str)
                res = normalized_html_table(result_str)
                result_str = res.replace('<html><body>','').replace('</body></html>','')
            result_str = compress_html_table(result_str.strip())
            return result_str

        result_str = truncate_repetitive_content(result_str)

        if ("\\(" in result_str and "\\)" in result_str) or (
            "\\[" in result_str and "\\]" in result_str
        ):
            result_str = result_str.replace("$", "")
            result_str = (
                result_str.replace("\\(", " $ ")
                .replace("\\)", " $ ")
                .replace("\\[", " $$ ")
                .replace("\\]", " $$ ")
            )
            
        if label == "formula_number":
            result_str = result_str.replace("$", "")
            
        return result_str
    
    @staticmethod
    def json2md(json_data: Dict[str, Any], layout_config: Dict[str, Any]) -> str:
        """Convert Layout Analysis result to Markdown.
        
        Strictly implemented based on build_markdown_from_json logic.
        """
        from types import SimpleNamespace
        import os
        
        try:
            from paddlex.inference.pipelines.layout_parsing.result_v2 import simplify_table_func
            from paddlex.inference.pipelines.paddleocr_vl.uilts import merge_blocks as merge_blocks_for_markdown
            from paddlex.inference.pipelines.paddleocr_vl.result import (
                build_handle_funcs_dict,
                format_image_plain_func,
                merge_formula_and_number,
            )
            from ..utils.paddle_layout_utils import merge_blocks_for_markdown
        except ImportError:
            raise ImportError("paddlex is required for PP-DocLayoutV2.json2md but not found.")

        label_mapping = layout_config.get("label_mapping", {})
        markdown_ignore_labels = label_mapping.get("ignore", [])

        format_text_func = lambda block: block.content
        format_image_func = format_image_plain_func
        format_chart_func = format_image_func
        format_table_func = lambda block: simplify_table_func("\n" + block.content)
        format_formula_func = lambda block: block.content
        format_seal_func = format_image_func

        handle_funcs_dict = build_handle_funcs_dict(
            text_func=format_text_func,
            image_func=format_image_func,
            chart_func=format_chart_func,
            table_func=format_table_func,
            formula_func=format_formula_func,
            seal_func=format_seal_func,
        )
        for label in markdown_ignore_labels:
            handle_funcs_dict.pop(label, None)

        image_labels = label_mapping.get("image", [])
        
        # Check merge_blocks_before_md from layout_config
        merge_blocks_before_md = layout_config.get("merge_blocks_before_md", False)
        
        raw_blocks = json_data.get("parsing_res_list", [])
        if merge_blocks_before_md:
            raw_blocks = merge_blocks_for_markdown(
                raw_blocks,  json_data.get("input_path"), image_labels
            )

        blocks = []
        base_name = os.path.splitext(os.path.basename(json_data['input_path']))[0]
        for b in raw_blocks:
            label = b.get("block_label")
            block = SimpleNamespace(
                label=label,
                content=b.get("block_content", ""),
                image=None,
            )
            if label in image_labels and b.get("block_bbox"):
                block.image = {
                    "path": f"imgs/{base_name}_{b['block_id']}.jpg",
                    "img": None,
                }
            blocks.append(block)

        markdown_content = ""
        for idx, block in enumerate(blocks):
            handle_func = handle_funcs_dict.get(block.label)
            
            # Logic for formula number showing
            show_formula_number = False # Default logic
            
            if (
                show_formula_number
                and block.label in {"display_formula", "formula"}
                and idx != len(blocks) - 1
                and blocks[idx + 1].label == "formula_number"
            ):
                block.content = merge_formula_and_number(
                    block.content, blocks[idx + 1].content
                )
            if handle_func:
                markdown_content += (
                    "\n\n" + handle_func(block)
                    if markdown_content
                    else handle_func(block)
                )
                
        return markdown_content

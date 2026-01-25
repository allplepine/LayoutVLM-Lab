"""Layout processing utilities."""
import numpy as np
import re
import html
import unicodedata
from pathlib import Path
from PIL import Image
from collections import Counter
from bs4 import BeautifulSoup
import subprocess
import shutil
import uuid
import os
from copy import deepcopy
from typing import Dict, List, Union, Optional, Any


def calculate_bbox_area(bbox):
    """Calculate bounding box area"""
    x1, y1, x2, y2 = map(float, bbox)
    area = abs((x2 - x1) * (y2 - y1))
    return area

def calculate_overlap_ratio(
    bbox1: Union[np.ndarray, list, tuple],
    bbox2: Union[np.ndarray, list, tuple],
    mode="union",
) -> float:
    """
    Calculate the overlap ratio between two bounding boxes using NumPy.

    Args:
        bbox1 (np.ndarray, list or tuple): The first bounding box, format [x_min, y_min, x_max, y_max]
        bbox2 (np.ndarray, list or tuple): The second bounding box, format [x_min, y_min, x_max, y_max]
        mode (str): The mode of calculation, either 'union', 'small', or 'large'.

    Returns:
        float: The overlap ratio value between the two bounding boxes
    """
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)

    x_min_inter = np.maximum(bbox1[0], bbox2[0])
    y_min_inter = np.maximum(bbox1[1], bbox2[1])
    x_max_inter = np.minimum(bbox1[2], bbox2[2])
    y_max_inter = np.minimum(bbox1[3], bbox2[3])

    inter_width = np.maximum(0, x_max_inter - x_min_inter)
    inter_height = np.maximum(0, y_max_inter - y_min_inter)

    inter_area = inter_width * inter_height

    bbox1_area = calculate_bbox_area(bbox1)
    bbox2_area = calculate_bbox_area(bbox2)

    if mode == "union":
        ref_area = bbox1_area + bbox2_area - inter_area
    elif mode == "small":
        ref_area = np.minimum(bbox1_area, bbox2_area)
    elif mode == "large":
        ref_area = np.maximum(bbox1_area, bbox2_area)
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    if ref_area == 0:
        return 0.0

    return inter_area / ref_area


def filter_overlap_boxes(
    layout_det_res: Dict[str, List[Dict]]
) -> Dict[str, List[Dict]]:
    """
    Remove overlapping boxes from layout detection results based on a given overlap ratio.

    Args:
        layout_det_res (Dict[str, List[Dict]]): Layout detection result dict containing a 'boxes' list.

    Returns:
        Dict[str, List[Dict]]: Filtered dict with overlapping boxes removed.
    """
    layout_det_res_filtered = deepcopy(layout_det_res)
    boxes = [
        box for box in layout_det_res_filtered["boxes"] if box["label"] != "reference"
    ]
    dropped_indexes = set()

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if i in dropped_indexes or j in dropped_indexes:
                continue
            overlap_ratio = calculate_overlap_ratio(
                boxes[i]["coordinate"], boxes[j]["coordinate"], "small"
            )
            if overlap_ratio > 0.7:
                box_area_i = calculate_bbox_area(boxes[i]["coordinate"])
                box_area_j = calculate_bbox_area(boxes[j]["coordinate"])
                if (
                    boxes[i]["label"] == "image" or boxes[j]["label"] == "image"
                ) and boxes[i]["label"] != boxes[j]["label"]:
                    continue
                if box_area_i >= box_area_j:
                    dropped_indexes.add(j)
                else:
                    dropped_indexes.add(i)
    layout_det_res_filtered["boxes"] = [
        box for idx, box in enumerate(boxes) if idx not in dropped_indexes
    ]
    return layout_det_res_filtered

def truncate_repetitive_content(
    content: str, line_threshold: int = 10, char_threshold: int = 10, min_len: int = 10
) -> str:
    """Detect and truncate repetitive content."""
    def find_repeating_suffix(s: str, min_len: int = 8, min_repeats: int = 5):
        for i in range(len(s) // (min_repeats), min_len - 1, -1):
            unit = s[-i:]
            if s.endswith(unit * min_repeats):
                count = 0
                temp_s = s
                while temp_s.endswith(unit):
                    temp_s = temp_s[:-i]
                    count += 1
                start_index = len(s) - (count * i)
                return s[:start_index], unit, count
        return None
    
    def find_shortest_repeating_substring(s: str):
        n = len(s)
        for i in range(1, n // 2 + 1):
            if n % i == 0:
                substring = s[:i]
                if substring * (n // i) == s:
                    return substring
        return None

    stripped_content = content.strip()
    if not stripped_content:
        return content

    if "\n" not in stripped_content and len(stripped_content) > 100:
        suffix_match = find_repeating_suffix(stripped_content, min_len=8, min_repeats=5)
        if suffix_match:
            prefix, repeating_unit, count = suffix_match
            if len(repeating_unit) * count > len(stripped_content) * 0.5:
                return prefix

    if "\n" not in stripped_content and len(stripped_content) > min_len:
        repeating_unit = find_shortest_repeating_substring(stripped_content)
        if repeating_unit:
            count = len(stripped_content) // len(repeating_unit)
            if count >= char_threshold:
                return repeating_unit

    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if not lines:
        return content
    total_lines = len(lines)
    if total_lines < line_threshold:
        return content
    line_counts = Counter(lines)
    most_common_line, count = line_counts.most_common(1)[0]
    if count >= line_threshold and (count / total_lines) >= 0.8:
        return most_common_line

    return content


def compress_html_table(html_str: str) -> str:
    html_str = re.sub(r'>\s+<', '><', html_str)
    html_str = re.sub(r'>\s*([^<>\s][^<>]*?)\s*<', r'>\1<', html_str)
    return html_str


def remove_markdown_fences(content):
    content = re.sub(r'^```markdown\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```html\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```latex\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'```\n?$', '', content, flags=re.MULTILINE)
    return content


def normalized_html_table(text):
    def process_table_html(md_i):
        """
        pred_md format edit
        """
        def process_table_html(html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            th_tags = soup.find_all('th')
            for th in th_tags:
                th.name = 'td'
            thead_tags = soup.find_all('thead')
            for thead in thead_tags:
                thead.unwrap()  # unwrap()会移除标签但保留其内容
            math_tags = soup.find_all('math')
            for math_tag in math_tags:
                alttext = math_tag.get('alttext', '')
                alttext = f'${alttext}$'
                if alttext:
                    math_tag.replace_with(alttext)
            span_tags = soup.find_all('span')
            for span in span_tags:
                span.unwrap()
            return str(soup)

        table_res=''
        table_res_no_space=''
        if '<table' in md_i.replace(" ","").replace("'",'"'):
            md_i = process_table_html(md_i)
            table_res = html.unescape(md_i).replace('\n', '')
            table_res = unicodedata.normalize('NFKC', table_res).strip()
            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, table_res, re.DOTALL | re.IGNORECASE)
            table_res = ''.join(tables)
            # table_res = re.sub('<table.*?>','',table_res)
            table_res = re.sub('( style=".*?")', "", table_res)
            table_res = re.sub('( height=".*?")', "", table_res)
            table_res = re.sub('( width=".*?")', "", table_res)
            table_res = re.sub('( align=".*?")', "", table_res)
            table_res = re.sub('( class=".*?")', "", table_res)
            table_res = re.sub('</?tbody>',"",table_res)
            
            table_res = re.sub(r'\s+', " ", table_res)
            table_res_no_space = '<html><body><table border="1" >' + table_res.replace(' ','') + '</table></body></html>'
            # table_res_no_space = re.sub(' (style=".*?")',"",table_res_no_space)
            # table_res_no_space = re.sub(r'[ ]', " ", table_res_no_space)
            table_res_no_space = re.sub('colspan="', ' colspan="', table_res_no_space)
            table_res_no_space = re.sub('rowspan="', ' rowspan="', table_res_no_space)
            table_res_no_space = re.sub('border="', ' border="', table_res_no_space)

            table_res = '<html><body><table border="1" >' + table_res + '</table></body></html>'
            # table_flow.append(table_res)
            # table_flow_no_space.append(table_res_no_space)

        return table_res, table_res_no_space
    
    def clean_table(input_str,flag=True):
        if flag:
            input_str = input_str.replace('<sup>', '').replace('</sup>', '')
            input_str = input_str.replace('<sub>', '').replace('</sub>', '')
            input_str = input_str.replace('<span>', '').replace('</span>', '')
            input_str = input_str.replace('<div>', '').replace('</div>', '')
            input_str = input_str.replace('<p>', '').replace('</p>', '')
            input_str = input_str.replace('<spandata-span-identity="">', '')
            input_str = re.sub('<colgroup>.*?</colgroup>','',input_str)
        return input_str
    
    norm_text, _ = process_table_html(text)
    norm_text = clean_table(norm_text)
    return norm_text


def normalized_latex_table(text):
    def latex_template(latex_code):  
        template = r'''
        \documentclass[border=20pt]{article}
        \usepackage{subcaption}
        \usepackage{url}
        \usepackage{graphicx}
        \usepackage{caption}
        \usepackage{multirow}
        \usepackage{booktabs}
        \usepackage{color}
        \usepackage{colortbl}
        \usepackage{xcolor,soul,framed}
        \usepackage{fontspec}
        \usepackage{amsmath,amssymb,mathtools,bm,mathrsfs,textcomp}
        \setlength{\parindent}{0pt}''' + \
        r'''
        \begin{document}
        ''' + \
        latex_code + \
        r'''
        \end{document}'''
    
        return template

    def process_table_latex(latex_code):
        SPECIAL_STRINGS= [
            ['\\\\vspace\\{.*?\\}', ''],
            ['\\\\hspace\\{.*?\\}', ''],
            ['\\\\rule\{.*?\\}\\{.*?\\}', ''],
            ['\\\\addlinespace\\[.*?\\]', ''],
            ['\\\\addlinespace', ''],
            ['\\\\renewcommand\\{\\\\arraystretch\\}\\{.*?\\}', ''],
            ['\\\\arraystretch\\{.*?\\}', ''],
            ['\\\\(row|column)?colors?\\{[^}]*\\}(\\{[^}]*\\}){0,2}', ''],
            ['\\\\color\\{.*?\\}', ''],
            ['\\\\textcolor\\{.*?\\}', ''],
            ['\\\\rowcolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\columncolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\cellcolor(\\[.*?\\])?\\{.*?\\}', ''],
            ['\\\\colorbox\\{.*?\\}', ''],
            ['\\\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)', ''],
            [r'\s+', ' '],
            ['\\\\centering', ''],
            ['\\\\begin\\{table\\}\\[.*?\\]', '\\\\begin{table}'],
            ['\t', ''],
            ['@{}', ''],
            ['\\\\toprule(\\[.*?\\])?', '\\\\hline'],
            ['\\\\bottomrule(\\[.*?\\])?', '\\\\hline'],
            ['\\\\midrule(\\[.*?\\])?', '\\\\hline'],
            ['p\\{[^}]*\\}', 'l'],
            ['m\\{[^}]*\\}', 'c'],
            ['\\\\scalebox\\{[^}]*\\}\\{([^}]*)\\}', '\\1'],
            ['\\\\textbf\\{([^}]*)\\}', '\\1'],
            ['\\\\textit\\{([^}]*)\\}', '\\1'],
            ['\\\\cmidrule(\\[.*?\\])?\\(.*?\\)\\{([0-9]-[0-9])\\}', '\\\\cline{\\2}'],
            ['\\\\hline', ''],
            [r'\\multicolumn\{1\}\{[^}]*\}\{((?:[^{}]|(?:\{[^{}]*\}))*)\}', r'\1']
        ]
        pattern = r'\\begin\{tabular\}.*\\end\{tabular\}'  # 注意这里不用 .*?
        matches = re.findall(pattern, latex_code, re.DOTALL)
        latex_code = ' '.join(matches)

        for special_str in SPECIAL_STRINGS:
            latex_code = re.sub(fr'{special_str[0]}', fr'{special_str[1]}', latex_code)

        return latex_code
    
    def convert_latex_to_html(latex_content, cache_dir='./temp'):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        uuid_str = str(uuid.uuid1())
        with open(f'{cache_dir}/{uuid_str}.tex', 'w') as f:
            f.write(latex_template(latex_content))

        cmd = ['latexmlc', '--quiet', '--nocomments', f'--log={cache_dir}/{uuid_str}.log',
               f'{cache_dir}/{uuid_str}.tex', f'--dest={cache_dir}/{uuid_str}.html']
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(f'{cache_dir}/{uuid_str}.html', 'r') as f:
                html_content = f.read()

            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
            tables = [f'<table>{table}</table>' for table in tables]
            html_content = '\n'.join(tables)
        
        except Exception as e:
            html_content = ''
        
        shutil.rmtree(cache_dir)
        return html_content
    
    html_text = convert_latex_to_html(text)
    return html_text

def merge_blocks_for_markdown(
    blocks: list, img_path: str | None, image_labels
) -> list:
    if not img_path or not os.path.isfile(img_path):
        return blocks

    non_merge_labels = image_labels + ["table"]

    try:
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
    except Exception:
        return blocks

    merged_input = []
    for block in blocks:
        bbox = block.get("block_bbox")
        block_img = None
        if bbox and len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
            block_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        merged_input.append(
            {
                **block,
                "label": block.get("block_label"),
                "box": bbox,
                "img": block_img,
            }
        )
    from paddlex.inference.pipelines.paddleocr_vl.uilts import merge_blocks
    merged_blocks = merge_blocks(merged_input, non_merge_labels=non_merge_labels)
    normalized_blocks = []
    for block in merged_blocks:
        out = dict(block)
        if "block_label" not in out:
            out["block_label"] = out.get("label")
        if "block_bbox" not in out:
            out["block_bbox"] = out.get("box")
        out.pop("label", None)
        out.pop("box", None)
        out.pop("img", None)
        out.pop("merge_aligns", None)
        normalized_blocks.append(out)
    return normalized_blocks

def convert_to_parsing_format(output, image_path: str) -> dict:
        # Extract boxes from output
        boxes = output['boxes']
                
        # Convert numpy types to native Python types for JSON serialization
        boxes = [_convert_box_to_native(box) for box in boxes]
        # Convert to parsing_res_list format
        parsing_res_list = []
        for idx, box in enumerate(boxes):
            coordinate = box.get('coordinate')
            parsing_res_list.append({
                "block_label": box.get('label', 'text'),
                "block_bbox": [int(c) for c in coordinate],
                "block_content": "",  # Will be filled by VLM
                "block_id": idx,
            })

        return {
            "input_path": image_path,
            "parsing_res_list": parsing_res_list
        }
    

def _convert_box_to_native(box: dict) -> dict:
        """Convert numpy types to native Python types."""
        result = {}
        for key, value in box.items():
            if key == 'coordinate':
                result[key] = [float(v) for v in value]
            elif key == 'score':
                result[key] = float(value)
            elif key == 'cls_id':
                result[key] = int(value)
            else:
                result[key] = value
        return result


def create_predictor(
    model_name: str,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    pp_option=None,
    use_hpip: bool = False,
    hpi_config: Optional[Union[Dict[str, Any], Any]] = None,
    genai_config: Optional[Union[Dict[str, Any], Any]] = None,
    *args,
    **kwargs):
    from paddlex.inference.utils.pp_option import PaddlePredictorOption
    pp_option = PaddlePredictorOption(run_mode="paddle",enable_cinn=False,trt_cfg_setting=None)
    from paddlex.inference.models.base import BasePredictor
    from paddlex.inference.models.common.genai import GenAIConfig, need_local_model
    from paddlex.inference.utils.official_models import official_models
    if genai_config is not None:
        genai_config = GenAIConfig.model_validate(genai_config)

    if need_local_model(genai_config):
        if model_dir is None:
            model_dir = official_models[model_name]
        else:
            assert Path(model_dir).exists(), f"{model_dir} is not exists!"
            model_dir = Path(model_dir)
        config = BasePredictor.load_config(model_dir)
        assert (
            model_name == config["Global"]["model_name"]
        ), f"Model name mismatch，please input the correct model dir."
    else:
        config = None

    return BasePredictor.get(model_name)(
        model_dir=model_dir,
        config=config,
        device=device,
        pp_option=pp_option,
        use_hpip=use_hpip,
        hpi_config=hpi_config,
        genai_config=genai_config,
        model_name=model_name,
        *args,
        **kwargs,
    )
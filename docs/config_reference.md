# Configuration Reference

This document explains the YAML configuration structure used by LayoutVLM-Lab.
Example file: `config/paddle_layoutv2.yaml`.

## Output structure

Results are written to `output.output_dir/<experiment.name>/`:
- Markdown outputs are saved directly in this folder
- `json/` structured results
- `imgs/` cropped image blocks (optional)

## Global sections

### experiment
| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | string | `"default"` | Run name used to create the output folder. |

### output
| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `output_dir` | string | `PROJECT_ROOT/results` | Base output directory for all runs. |
| `save_images` | bool | `true` | Save cropped image blocks for labels mapped to `image`. |

### input
| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `image_root` | string | `PROJECT_ROOT/images` | Directory containing input images. |
| `filter_processed` | bool | `true` | Skip images that already have JSON results in `json/`. |

### pipeline
| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `vlm_workers` | int | `20` | Number of VLM worker threads. |
| `vlm_queue_max_size` | int | `50` | Max VLM queue size (recommend ~2x workers). |

## Layout section

### layout
| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | string | required | Layout backend id (e.g. `paddle`). |
| `config` | object | `{}` | Backend-specific settings passed through. |
| `label_mapping` | object | `{}` | Mapping from layout labels to metric types. |

### layout.config (common fields)
- `model_name`: layout model name (e.g. `PP-DocLayoutV2`)
- `model_dir`: model weights directory
- `device_list`: GPU ids, comma-separated (e.g. `"0,1"`)
- `nums_per_gpu`: processes per GPU
- `merge_blocks_before_md`: merge blocks before Markdown export

### label_mapping
- `ignore`: labels ignored during Markdown export
- `image`: labels skipped by VLM; crops saved to `imgs/` if `save_images` is true
- `table`: labels processed with VLM table prompt
- `formula`: labels processed with VLM formula prompt
- others default to `ocr`

## VLM section

### vlm
| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | string | required | VLM backend id (e.g. `openai`). |
| `config` | object | `{}` | Backend-specific settings passed through. |
| `metrics` | list | `[]` | Enabled metric types; only these blocks are sent to VLM. |

### vlm.config (OpenAI-compatible)
- `model_name`
- `api_key` / `base_url` (can be `${OPENAI_API_KEY}` / `${OPENAI_API_BASE_URL}`)
- `max_tokens`, `temperature`, `timeout`
- `text_before_image`

## Environment variables
`Config.from_yaml` resolves `${VAR}` placeholders using environment variables from `.env`.

## Backward compatibility
If `experiment.output_dir` is present and `output.output_dir` is missing, the loader will still use it.

If both are omitted, it falls back to `PROJECT_ROOT/results/<experiment.name>` (with `experiment.name` defaulting to `default`), and the `json/` and `imgs/` subfolders are created automatically.

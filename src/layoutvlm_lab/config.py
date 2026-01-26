"""Configuration loader for LayoutVLM-Lab with modular architecture."""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

@dataclass
class ExperimentConfig:
    name: str = "default"


@dataclass
class ModuleConfig:
    """Generic config for a module (type + specific config)."""
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields that might be present in Layout/VLM configs
    label_mapping: Dict[str, List[str]] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)


@dataclass
class ParallelConfig:
    layout_num_per_device: int = 1
    vlm_workers: int = 20
    vlm_queue_max_size: int = 50
    block_workers: int = 0


@dataclass
class InputConfig:
    image_root: str = os.path.join(PROJECT_ROOT, "images")
    filter_processed: bool = True


@dataclass
class OutputConfig:
    output_dir: str = os.path.join(PROJECT_ROOT, "results")
    save_images: bool = True


@dataclass
class Config:
    experiment: ExperimentConfig
    layout: ModuleConfig
    vlm: ModuleConfig
    pipeline: ParallelConfig
    input: InputConfig
    output: OutputConfig
    
    # Environment variables
    openai_api_key: str = ""
    openai_api_base_url: str = ""
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        # Load environment variables
        load_dotenv()
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        # Helper to resolve environment variables in config values
        def resolve_env_vars(d):
            if isinstance(d, dict):
                return {k: resolve_env_vars(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [resolve_env_vars(v) for v in d]
            elif isinstance(d, str) and d.startswith("${") and d.endswith("}"):
                env_key = d[2:-1]
                return os.getenv(env_key, "")
            return d
            
        data = resolve_env_vars(data)
        
        # Parse sections
        experiment_data = data.get("experiment", {})
        output_data = data.get("output", {})
        # Backward compatibility: allow experiment.output_dir if output.output_dir is missing
        if "output_dir" in experiment_data and "output_dir" not in output_data:
            output_data = {**output_data, "output_dir": experiment_data["output_dir"]}
        experiment = ExperimentConfig(**{k: v for k, v in experiment_data.items() if k != "output_dir"})
        
        layout_data = data.get("layout", {})
        layout = ModuleConfig(
            type=layout_data.get("type"),
            config=layout_data.get("config", {}),
            label_mapping=layout_data.get("label_mapping", {})
        )
        
        vlm_data = data.get("vlm", {})
        vlm = ModuleConfig(
            type=vlm_data.get("type"),
            config=vlm_data.get("config", {}),
            metrics=vlm_data.get("metrics", [])
        )
        
        pipeline = ParallelConfig(**data.get("pipeline", {}))
        input_config = InputConfig(**data.get("input", {}))
        output_config = OutputConfig(**output_data)
        
        config = cls(
            experiment=experiment,
            layout=layout,
            vlm=vlm,
            pipeline=pipeline,
            input=input_config,
            output=output_config
        )
        
        # Load API keys (fallback logic)
        config.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        config.openai_api_base_url = os.getenv("OPENAI_API_BASE_URL", "")
        
        # Create output directories
        os.makedirs(config.run_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        os.makedirs(config.json_dir, exist_ok=True)
        os.makedirs(config.imgs_dir, exist_ok=True)
        
        return config
    
    @property
    def results_dir(self) -> str:
        return self.run_dir
    
    @property
    def json_dir(self) -> str:
        return os.path.join(self.run_dir, "json")
    
    @property
    def imgs_dir(self) -> str:
        return os.path.join(self.run_dir, "imgs")

    @property
    def run_dir(self) -> str:
        return os.path.join(self.output.output_dir, self.experiment.name)


def load_config(config_path: str = "config/config.yaml") -> Config:
    """Load configuration from file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return Config.from_yaml(config_path)


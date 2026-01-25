"""Entry point for LayoutVLM-Lab."""

import argparse
import logging
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
from layoutvlm_lab.config import load_config
from layoutvlm_lab.pipeline import run_pipeline


def main():
    """Main entry point."""
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.WARNING,
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="LayoutVLM-Lab: Document parsing with Layout+VLM")
    parser.add_argument(
        "--config", "-c",
        default="config/paddle_layoutv2.yaml",
        help="Path to config file (default: config/config.yaml)"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()


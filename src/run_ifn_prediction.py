"""Run the two-stage IFNγ expression-profile prediction workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from ifn_gamma_mlp.pipeline import run_pipeline


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="src/config.ifn_gamma_mlp.yaml",
        help="Path to the IFNγ MLP YAML configuration.",
    )
    args = parser.parse_args()
    run_pipeline(load_config(args.config))


if __name__ == "__main__":
    main()

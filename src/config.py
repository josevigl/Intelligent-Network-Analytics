"""Configuration loading and validation for reusable pipeline components."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_pipeline_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline configuration not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Pipeline configuration must be a YAML mapping: {path}")
    return deepcopy(config)


def validate_pipeline_config(config: dict[str, Any]) -> None:
    project = config.get("project")
    training = config.get("training")
    preprocessing = config.get("preprocessing")
    if not all(isinstance(value, dict) for value in (project, training, preprocessing)):
        raise ValueError(
            "Configuration requires project, training, and preprocessing mappings."
        )

    test_size = project.get("test_size")
    validation_size = training.get("validation_size")
    if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        raise ValueError("project.test_size must be between zero and one.")
    if (
        not isinstance(validation_size, (int, float))
        or not 0 < validation_size < 1 - test_size
    ):
        raise ValueError(
            "training.validation_size must be positive and smaller than "
            "1 - project.test_size."
        )

    feature_map = preprocessing.get("selected_kpis")
    if not isinstance(feature_map, dict):
        raise ValueError("preprocessing.selected_kpis must map task names to KPIs.")

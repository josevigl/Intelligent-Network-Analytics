"""Reproducible, data-verified test holdout manifests."""

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.pipeline import prepare_task_data_component


HOLDOUT_MANIFEST_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Hash row order and values for the dataset used to derive the holdout."""
    columns = [str(column) for column in frame.columns]
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            sorted(columns),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for values in frame.itertuples(index=False, name=None):
        record = {
            str(column): _jsonable(value)
            for column, value in zip(frame.columns, values)
        }
        payload = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(b"\n")
        digest.update(payload)
    return digest.hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(
        _jsonable(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_holdout_manifest(
    frame: pd.DataFrame,
    task_name: str,
    config: dict[str, Any],
    dataset_name: str,
    source_split: str,
    code_revision: str,
    dataset_sha256: str | None = None,
) -> dict[str, Any]:
    """Record the exact test rows and provenance without fitting a model."""
    splits = prepare_task_data_component(frame, task_name, config)
    if dataset_sha256 is None:
        dataset_sha256 = dataset_fingerprint(frame)
    elif not isinstance(dataset_sha256, str) or not dataset_sha256:
        raise ValueError("dataset_sha256 must be a non-empty SHA-256 string.")
    return {
        "schema_version": HOLDOUT_MANIFEST_VERSION,
        "task_name": task_name,
        "dataset_name": dataset_name,
        "source_split": source_split,
        "dataset_sha256": dataset_sha256,
        "project_config_sha256": config_fingerprint(config),
        "code_revision": code_revision,
        "test_positions": [int(position) for position in splits.positions_test],
        "test_sample_count": int(len(splits.y_test)),
        "test_group_count": (
            int(len(np.unique(splits.groups_test)))
            if splits.groups_test is not None
            else None
        ),
        "grouping_strategy": (
            "overlapping_time_intervals"
            if splits.groups_test is not None
            else "stratified_samples"
        ),
    }


def validate_holdout_manifest(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    task_name: str,
    config: dict[str, Any],
) -> Any:
    """Recreate and verify a manifest before using its test rows."""
    if manifest.get("schema_version") != HOLDOUT_MANIFEST_VERSION:
        raise ValueError(
            "Unsupported holdout manifest schema version: "
            f"{manifest.get('schema_version')!r}."
        )
    if manifest.get("task_name") != task_name:
        raise ValueError("The holdout manifest belongs to a different task.")
    if manifest.get("dataset_sha256") != dataset_fingerprint(frame):
        raise ValueError(
            "The source dataset does not match the dataset used to create "
            "the registered holdout."
        )
    if manifest.get("project_config_sha256") != config_fingerprint(config):
        raise ValueError(
            "The project configuration differs from the registered holdout."
        )

    splits = prepare_task_data_component(frame, task_name, config)
    positions = [int(position) for position in splits.positions_test]
    if manifest.get("test_positions") != positions:
        raise ValueError(
            "The current test split differs from the registered holdout manifest."
        )
    if manifest.get("test_sample_count") != len(splits.y_test):
        raise ValueError("The registered holdout sample count does not match.")
    return splits

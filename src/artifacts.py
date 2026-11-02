"""Versioned model bundles that keep each fitted preprocessor with its model."""

from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import joblib
import sklearn

from src.models.training import ModelTrainingResult
from src.preprocessing.pipeline import PreprocessedTask


ARTIFACT_SCHEMA_VERSION = 1


@dataclass
class TaskModelArtifact:
    """Complete task inference contract and fitted model state."""

    schema_version: int
    task_name: str
    sequence_length: int
    feature_channels: tuple[str, ...]
    transformed_feature_names: tuple[str, ...]
    model_feature_names: tuple[str, ...]
    input_mode: str
    preprocessor: Any
    model: Any
    target_encoder: Any
    target_classes: tuple[str, ...]
    config: dict[str, Any]
    validation_metrics: dict[str, float]
    library_versions: dict[str, str]
    model_name: str = "random_forest"
    model_parameters: dict[str, Any] | None = None
    decision_thresholds: tuple[float, ...] = ()
    cv_metrics: dict[str, float] | None = None
    test_metrics: dict[str, float] | None = None
    selected: bool = False


def build_task_artifact(
    preprocessed: PreprocessedTask,
    training_result: ModelTrainingResult,
    config: dict[str, Any],
) -> TaskModelArtifact:
    if preprocessed.splits.task_name != training_result.task_name:
        raise ValueError("The preprocessor and model belong to different tasks.")
    return TaskModelArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        task_name=training_result.task_name,
        sequence_length=preprocessed.preprocessor.sequence_length_,
        feature_channels=tuple(preprocessed.splits.feature_channels),
        transformed_feature_names=tuple(
            preprocessed.preprocessor.get_feature_names_out()
        ),
        model_feature_names=training_result.model_feature_names,
        input_mode=training_result.input_mode,
        preprocessor=preprocessed.preprocessor,
        model=training_result.model,
        target_encoder=training_result.target_encoder,
        target_classes=training_result.target_classes,
        config=config,
        validation_metrics=training_result.validation_metrics,
        library_versions={
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    )


def save_task_artifact(
    artifact: TaskModelArtifact,
    artifact_directory: str | Path,
) -> Path:
    """Atomically persist one task bundle and a human-readable JSON manifest."""
    directory = Path(artifact_directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifact_path = directory / "model_bundle.joblib"
    manifest_path = directory / "metadata.json"

    with tempfile.NamedTemporaryFile(
        prefix=".model_bundle-",
        suffix=".tmp",
        dir=directory,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(artifact, temporary_path)
        os.replace(temporary_path, artifact_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    excluded_fields = {"preprocessor", "model", "target_encoder"}
    manifest = {
        field.name: getattr(artifact, field.name)
        for field in fields(artifact)
        if field.name not in excluded_fields
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".metadata-",
        suffix=".tmp",
        dir=directory,
        delete=False,
    ) as temporary:
        json.dump(manifest, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        manifest_temporary_path = Path(temporary.name)
    try:
        os.replace(manifest_temporary_path, manifest_path)
    except Exception:
        manifest_temporary_path.unlink(missing_ok=True)
        raise
    return artifact_path


def load_task_artifact(artifact_path: str | Path) -> TaskModelArtifact:
    """Load a trusted joblib bundle from a file or task artifact directory."""
    path = Path(artifact_path)
    if path.is_dir():
        path = path / "model_bundle.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, TaskModelArtifact):
        raise TypeError(f"Unsupported model artifact type: {type(artifact).__name__}")
    if artifact.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact schema version: {artifact.schema_version}"
        )
    return artifact

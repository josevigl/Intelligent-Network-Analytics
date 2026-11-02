"""Inference components that reuse fitted task artifact bundles without fitting."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import TaskModelArtifact, load_task_artifact
from src.models.training import summarize_windows
from src.preprocessing.data import extract_inference_windows


@dataclass(frozen=True)
class PredictionResult:
    """Predictions aligned to valid source rows from an inference batch."""

    task_name: str
    positions: np.ndarray
    predictions: np.ndarray


def predict_task_component(
    frame: pd.DataFrame,
    artifact: TaskModelArtifact,
) -> PredictionResult:
    """Transform and predict with a saved bundle; this function never calls fit."""
    windows, positions = extract_inference_windows(
        frame,
        feature_channels=artifact.feature_channels,
        sequence_length=artifact.sequence_length,
    )
    transformed = artifact.preprocessor.transform(windows)
    if artifact.input_mode == "window_statistics":
        model_input, feature_names = summarize_windows(
            transformed, artifact.transformed_feature_names
        )
        if feature_names != artifact.model_feature_names:
            raise ValueError("Inference feature order differs from the saved model.")
    elif artifact.input_mode == "per_timestep":
        model_input = transformed.reshape(-1, transformed.shape[-1])
        feature_names = tuple(artifact.preprocessor.get_feature_names_out())
        if feature_names != artifact.model_feature_names:
            raise ValueError("Inference feature order differs from the saved model.")
    else:
        raise ValueError(f"Unknown model input mode: {artifact.input_mode!r}.")

    predictions = artifact.model.predict(model_input)
    if artifact.task_name == "temporal_localization":
        predictions = predictions.reshape(len(windows), artifact.sequence_length)
    elif artifact.task_name == "affected_kpis":
        if artifact.target_encoder is None:
            raise ValueError("Affected-KPI artifact is missing its label encoder.")
        predictions = artifact.target_encoder.inverse_transform(
            np.asarray(predictions, dtype=np.int8)
        )
        decoded_predictions = [
            tuple(str(label) for label in labels) for labels in predictions
        ]
        predictions = np.empty(len(decoded_predictions), dtype=object)
        predictions[:] = decoded_predictions

    return PredictionResult(
        task_name=artifact.task_name,
        positions=positions,
        predictions=np.asarray(predictions),
    )


def load_and_predict_component(
    frame: pd.DataFrame,
    artifact_path: str | Path,
) -> PredictionResult:
    """Load a persisted artifact and run its corresponding inference transform."""
    return predict_task_component(frame, load_task_artifact(artifact_path))

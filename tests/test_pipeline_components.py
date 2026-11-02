"""Focused tests for train-only preprocessing and artifact-based inference."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.artifacts import (
    build_task_artifact,
    load_task_artifact,
    save_task_artifact,
)
from src.config import load_pipeline_config
from src.inference import predict_task_component
from src.models.training import summarize_windows, train_model_component
from src.preprocessing.pipeline import preprocess_task_component
from src.preprocessing.splitting import TaskSplits
from pipelines.train import run_training_pipeline


class PipelineComponentTests(unittest.TestCase):
    def test_preprocessing_fits_only_train_and_ignores_unknown_category(self) -> None:
        splits = TaskSplits(
            task_name="binary_detection",
            X_train=np.array(
                [
                    [[1.0, "A"], [3.0, "A"]],
                    [[3.0, "A"], [5.0, "A"]],
                ],
                dtype=object,
            ),
            y_train=np.array([0, 1]),
            positions_train=np.array([0, 1]),
            X_validation=np.array([[[1000.0, "B"], [1000.0, "B"]]], dtype=object),
            y_validation=np.array([1]),
            positions_validation=np.array([2]),
            X_test=np.array([[[2000.0, "C"], [2000.0, "C"]]], dtype=object),
            y_test=np.array([0]),
            positions_test=np.array([3]),
            feature_channels=("RSRP", "DL_Protocol"),
        )
        config = load_pipeline_config()
        processed = preprocess_task_component(splits, config)

        numeric_pipeline = processed.preprocessor.transformer_.named_transformers_[
            "numeric"
        ]
        np.testing.assert_allclose(
            numeric_pipeline.named_steps["scaler"].mean_, [3.0]
        )
        self.assertEqual(
            processed.preprocessor.transformer_.named_transformers_[
                "categorical"
            ].named_steps["encoder"].categories_[0].tolist(),
            ["A"],
        )
        self.assertEqual(processed.X_validation.shape, (1, 2, 2))
        self.assertTrue(np.all(processed.X_validation[:, :, 1] == 0))
        self.assertTrue(np.isfinite(processed.X_test).all())

    def test_artifact_round_trip_inference_does_not_refit(self) -> None:
        train_windows = np.array(
            [
                [[0.0], [1.0]],
                [[0.1], [1.1]],
                [[4.0], [5.0]],
                [[4.1], [5.1]],
                [[0.2], [1.2]],
                [[3.9], [4.9]],
                [[0.3], [1.3]],
                [[4.2], [5.2]],
            ],
            dtype=object,
        )
        splits = TaskSplits(
            task_name="binary_detection",
            X_train=train_windows,
            y_train=np.array([0, 0, 1, 1, 0, 1, 0, 1]),
            positions_train=np.arange(8),
            X_validation=train_windows[:4],
            y_validation=np.array([0, 0, 1, 1]),
            positions_validation=np.arange(8, 12),
            X_test=train_windows[4:],
            y_test=np.array([0, 1, 0, 1]),
            positions_test=np.arange(12, 16),
            feature_channels=("RSRP",),
        )
        config = load_pipeline_config()
        config["training"]["random_forest"].update(
            n_estimators=5,
            max_depth=3,
            min_samples_leaf=1,
        )
        processed = preprocess_task_component(splits, config)
        trained = train_model_component(processed, config)
        artifact = build_task_artifact(processed, trained, config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = save_task_artifact(artifact, temporary_directory)
            loaded = load_task_artifact(artifact_path)
            frame = pd.DataFrame(
                {
                    "KPIs": [
                        {"RSRP": [0.2, 1.2]},
                        {"RSRP": [4.1, 5.1]},
                    ]
                }
            )
            with patch.object(
                loaded.preprocessor,
                "fit",
                side_effect=AssertionError("Inference attempted to fit."),
            ):
                predictions = predict_task_component(frame, loaded)

        self.assertEqual(loaded.task_name, "binary_detection")
        self.assertEqual(Path(artifact_path).name, "model_bundle.joblib")
        self.assertEqual(predictions.positions.tolist(), [0, 1])
        self.assertEqual(predictions.predictions.shape, (2,))

    def test_window_summary_names_match_feature_order(self) -> None:
        values = np.array([[[1.0, 4.0], [3.0, 8.0]]])
        matrix, feature_names = summarize_windows(values, ("RSRP", "UL_SNR"))

        self.assertEqual(matrix.shape, (1, 12))
        self.assertEqual(
            feature_names[:4],
            ("RSRP__mean", "UL_SNR__mean", "RSRP__std", "UL_SNR__std"),
        )

    def test_all_four_training_branches_save_independent_artifacts(self) -> None:
        rows = []
        for index in range(120):
            anomalous = index % 2 == 0
            anomaly_type = f"type_{(index // 2) % 3}"
            annotation = (
                {
                    "exists": True,
                    "type": anomaly_type,
                    "anomaly_duration": {"start": 1, "end": 2},
                    "affected_kpis": [
                        "RSRP",
                        "UL_SNR" if (index // 2) % 2 == 0 else "DL_MCS",
                    ],
                }
                if anomalous
                else {"exists": False}
            )
            signal = float(index % 17)
            rows.append(
                {
                    "KPIs": {
                        "RSRP": [signal, signal + 1, signal + 2, signal + 3]
                    },
                    "anomalies": annotation,
                }
            )
        frame = pd.DataFrame(rows)
        config = load_pipeline_config()
        config["preprocessing"]["sequence_length"] = 4
        config["preprocessing"]["selected_kpis"] = {
            task: ["RSRP"]
            for task in (
                "binary_detection",
                "anomaly_type",
                "temporal_localization",
                "affected_kpis",
            )
        }
        config["training"]["max_parallel_tasks"] = 1
        config["training"]["random_forest"].update(
            n_estimators=5,
            max_depth=3,
            min_samples_leaf=1,
            max_temporal_training_timesteps=100,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            results = run_training_pipeline(
                frame,
                config,
                artifact_directory=temporary_directory,
            )

            self.assertEqual(len(results), 4)
            for task_name, result in results.items():
                artifact = load_task_artifact(result["artifact_path"])
                self.assertEqual(artifact.task_name, task_name)
                self.assertEqual(
                    artifact.feature_channels,
                    ("RSRP",),
                )
                self.assertTrue(result["validation_metrics"])


if __name__ == "__main__":
    unittest.main()

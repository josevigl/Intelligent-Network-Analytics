"""Focused tests for train-only preprocessing and artifact-based inference."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
from src.preprocessing.pipeline import (
    prepare_task_data_component,
    preprocess_task_component,
)
from src.preprocessing.splitting import TaskSplits
from pipelines.train import run_training_pipeline
from src.models.experiments import run_task_model_comparison
from pipelines.train_models import run_model_comparison_pipeline
from src.models.estimators import SafeBinaryLogisticRegression
from pipelines.mlrun_components import (
    _resolve_frame,
    load_telecomts_component,
    train_model_comparison_component,
)
from pipelines.mlrun_train import (
    _selected_kpis,
    _validate_source,
    register_and_run_pipeline,
)


class PipelineComponentTests(unittest.TestCase):
    def test_mlrun_loader_registers_only_source_train_split(self) -> None:
        frame = pd.DataFrame(
            {
                "start_time": ["2026-01-01", "2026-01-02"],
                "end_time": ["2026-01-01", "2026-01-02"],
                "KPIs": [
                    {"RSRP": [1, 2], "UL_SNR": [3, 4], "unused": [5, 6]},
                    {"RSRP": [7, 8], "UL_SNR": [9, 10], "unused": [11, 12]},
                ],
                "anomalies": [{}, {}],
                "QnA": ["unused", "unused"],
            }
        )

        class Context:
            def __init__(self) -> None:
                self.datasets = []
                self.results = {}
                self.labels = {}

            def log_dataset(self, **kwargs: object) -> None:
                self.datasets.append(kwargs)

            def log_result(self, key: str, value: object) -> None:
                self.results[key] = value

            def set_label(self, key: str, value: str) -> None:
                self.labels[key] = value

        context = Context()
        with patch(
            "pipelines.mlrun_components.load_training_frame",
            return_value=frame,
        ) as load_frame:
            load_telecomts_component(
                context,
                split="train",
                selected_kpis_json='["RSRP"]',
            )
            with self.assertRaisesRegex(ValueError, "only accepts"):
                load_telecomts_component(context, split="test")

        load_frame.assert_called_once_with("AliMaatouk/TelecomTS", "train")
        self.assertEqual(context.datasets[0]["key"], "train_dataset")
        artifact_frame = context.datasets[0]["df"]
        self.assertEqual(
            artifact_frame.columns.tolist(),
            ["start_time", "end_time", "KPIs", "anomalies"],
        )
        self.assertEqual(artifact_frame.loc[0, "KPIs"], {"RSRP": [1, 2]})
        self.assertEqual(context.datasets[0]["format"], "parquet")
        self.assertEqual(context.results, {"rows": 2, "columns": 4})
        self.assertEqual(context.labels["source_split"], "train")
        self.assertEqual(context.labels["selected_kpi_count"], "1")

    def test_mlrun_training_logs_metrics_candidate_artifacts_and_selected_model(
        self,
    ) -> None:
        class DataItem:
            def as_df(self) -> pd.DataFrame:
                return pd.DataFrame({"value": [1]})

        self.assertEqual(_resolve_frame(DataItem()).iloc[0, 0], 1)

        class Context:
            def __init__(self) -> None:
                self.results = {}
                self.artifacts = []
                self.models = []
                self.comparison_payload = None
                self.labels = {}

            def set_label(self, key: str, value: str) -> None:
                self.labels[key] = value

            def log_result(self, key: str, value: object) -> None:
                self.results[key] = value

            def log_artifact(self, **kwargs: object) -> SimpleNamespace:
                self.artifacts.append(kwargs)
                if kwargs["key"] == "comparison":
                    self.comparison_payload = json.loads(
                        Path(kwargs["local_path"]).read_text(encoding="utf-8")
                    )
                return SimpleNamespace(uri=f"store://test/{kwargs['key']}")

            def log_model(self, **kwargs: object) -> SimpleNamespace:
                self.models.append(kwargs)
                return SimpleNamespace(uri="store://test/selected_model")

        expected = {
            "task_name": "binary_detection",
            "cross_validation_folds": 2,
            "sample_counts": {"train": 10, "validation": 2, "test": 2},
            "models": [
                {
                    "model_name": "random_forest",
                    "selected": True,
                    "parameters": {"n_estimators": 10},
                    "artifact_path": "",
                    "cv_metrics": {"f1_macro_mean": 0.8},
                    "validation_metrics": {"f1_macro": 0.75},
                    "test_metrics": {"f1_macro": 0.7},
                }
            ],
        }

        def fake_comparison(
            task_name: str,
            splits: object,
            project_config: dict[str, object],
            model_config: dict[str, object],
            artifact_directory: Path,
            evaluate_test: bool,
        ) -> dict[str, object]:
            self.assertEqual(task_name, "binary_detection")
            self.assertIsNotNone(splits)
            self.assertEqual(project_config, {})
            self.assertEqual(model_config, {})
            self.assertTrue(evaluate_test)
            bundle = artifact_directory / task_name / "random_forest"
            bundle.mkdir(parents=True)
            bundle_path = bundle / "model_bundle.joblib"
            bundle_path.write_bytes(b"test model")
            (bundle / "metadata.json").write_text("{}", encoding="utf-8")
            (artifact_directory / task_name / "comparison.json").write_text(
                "{}", encoding="utf-8"
            )
            expected["models"][0]["artifact_path"] = str(bundle_path)
            return expected

        context = Context()
        with (
            patch(
                "pipelines.mlrun_components.prepare_task_data_component",
                return_value=object(),
            ),
            patch(
                "pipelines.mlrun_components.run_task_model_comparison",
                side_effect=fake_comparison,
            ),
        ):
            result = train_model_comparison_component(
                context,
                dataset=pd.DataFrame({"value": [1]}),
                task_name="binary_detection",
                project_config_json="{}",
                model_config_json="{}",
            )

        self.assertEqual(result["selected_model"], "random_forest")
        self.assertEqual(context.results["cv_f1_macro_mean"], 0.8)
        self.assertEqual(context.results["validation_f1_macro"], 0.75)
        self.assertEqual(context.results["test_f1_macro"], 0.7)
        self.assertEqual(
            context.results["random_forest_cv_f1_macro_mean"],
            0.8,
        )
        self.assertEqual(
            context.results["random_forest_parameters"],
            '{"n_estimators": 10}',
        )
        self.assertEqual(len(context.artifacts), 3)
        self.assertEqual(context.models[0]["key"], "selected_model")
        self.assertEqual(
            result["selected_model_uri"],
            "store://test/selected_model",
        )
        self.assertEqual(
            context.comparison_payload["models"][0]["artifact_path"],
            "store://test/random_forest_bundle",
        )
        self.assertEqual(
            context.models[0]["parameters"],
            {"n_estimators": 10},
        )

    def test_mlrun_source_rejects_embedded_credentials(self) -> None:
        _validate_source("git://github.com/example/project.git")
        with self.assertRaisesRegex(ValueError, "credentials"):
            _validate_source("https://user:secret@example.com/project.git")
        with self.assertRaisesRegex(ValueError, "query parameters"):
            _validate_source("https://example.com/project.zip?token=secret")

    def test_mlrun_dataset_uses_only_kpis_for_requested_task_branches(self) -> None:
        config = {
            "preprocessing": {
                "selected_kpis": {
                    "binary_detection": ["RSRP", "UL_SNR"],
                    "anomaly_type": ["RSRP", "DL_MCS"],
                }
            }
        }
        self.assertEqual(
            _selected_kpis(config, ["anomaly_type", "binary_detection"]),
            ["DL_MCS", "RSRP", "UL_SNR"],
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            _selected_kpis(config, ["anomaly_type", "anomaly_type"])

    def test_mlrun_launcher_registers_functions_and_submits_task_parameters(
        self,
    ) -> None:
        project = unittest.mock.Mock()
        project.run.return_value = SimpleNamespace(uid="run-123")
        mlrun_module = SimpleNamespace(
            get_or_create_project=unittest.mock.Mock(return_value=project)
        )
        project_config = {
            "preprocessing": {
                "selected_kpis": {"binary_detection": ["RSRP", "UL_SNR"]}
            }
        }

        with patch.dict("sys.modules", {"mlrun": mlrun_module}):
            run = register_and_run_pipeline(
                source="git://github.com/example/project.git",
                project_name="ina-test",
                workflow_name="training",
                base_image="mlrun/mlrun",
                image=None,
                dataset_name="example/dataset",
                tasks=["binary_detection"],
                project_config=project_config,
                model_config={},
                evaluate_test=False,
                watch=False,
            )

        self.assertEqual(run.uid, "run-123")
        self.assertEqual(project.set_function.call_count, 2)
        self.assertEqual(project.set_function.return_value.save.call_count, 2)
        project.build_image.assert_called_once_with()
        run_arguments = project.run.call_args.kwargs["arguments"]
        self.assertEqual(run_arguments["task_names"], ["binary_detection"])
        self.assertEqual(run_arguments["selected_kpis_json"], '["RSRP", "UL_SNR"]')
        self.assertFalse(run_arguments["evaluate_test"])
        self.assertFalse(project.run.call_args.kwargs["watch"])

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

    def test_window_summary_supports_robust_anomaly_type_statistics(self) -> None:
        values = np.array([[[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]]])
        selected_statistics = (
            "mean",
            "median",
            "q10",
            "q90",
            "first",
            "last",
            "delta",
            "iqr",
        )
        matrix, feature_names = summarize_windows(
            values,
            ("RSRP", "UL_SNR"),
            selected_statistics,
        )

        self.assertEqual(matrix.shape, (1, 16))
        self.assertEqual(feature_names[:2], ("RSRP__mean", "UL_SNR__mean"))
        self.assertEqual(feature_names[-2:], ("RSRP__iqr", "UL_SNR__iqr"))
        self.assertTrue(np.isfinite(matrix).all())

    def test_safe_binary_logistic_regression_handles_constant_label(self) -> None:
        features = np.array([[0.0], [1.0], [2.0]])
        for label in (0, 1):
            estimator = SafeBinaryLogisticRegression().fit(
                features,
                np.full(3, label),
            )
            np.testing.assert_array_equal(
                estimator.predict(features),
                np.full(3, label),
            )
            np.testing.assert_allclose(
                estimator.predict_proba(features)[:, label],
                np.ones(3),
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

    def test_group_split_has_no_time_overlap_between_partitions(self) -> None:
        rows = []
        base_time = pd.Timestamp("2025-01-01T00:00:00")
        for index in range(60):
            anomalous = index % 2 == 0
            rows.append(
                {
                    "start_time": base_time
                    + pd.Timedelta(seconds=(index // 2) * 50 + (index % 2) * 10),
                    "end_time": base_time
                    + pd.Timedelta(seconds=(index // 2) * 50 + (index % 2) * 10 + 12),
                    "KPIs": {"RSRP": [float(index)] * 4},
                    "anomalies": (
                        {
                            "exists": True,
                            "type": "type_1",
                            "anomaly_duration": {"start": 1, "end": 2},
                            "affected_kpis": ["RSRP"],
                        }
                        if anomalous
                        else {"exists": False}
                    ),
                }
            )
        frame = pd.DataFrame(rows)
        config = load_pipeline_config()
        config["preprocessing"]["sequence_length"] = 4
        config["preprocessing"]["selected_kpis"]["binary_detection"] = ["RSRP"]
        splits = prepare_task_data_component(frame, "binary_detection", config)

        self.assertIsNotNone(splits.groups_train)
        self.assertFalse(
            set(splits.groups_train) & set(splits.groups_validation)
        )
        self.assertFalse(set(splits.groups_train) & set(splits.groups_test))
        self.assertFalse(
            set(splits.groups_validation) & set(splits.groups_test)
        )
        self.assertTrue(len(np.unique(splits.groups_train)) < len(splits.y_train))

    def test_training_entrypoint_writes_comparison_results(self) -> None:
        rows = []
        base_time = pd.Timestamp("2025-01-01T00:00:00")
        for index in range(60):
            signal = float(index % 2) * 5 + index / 100
            rows.append(
                {
                    "start_time": base_time + pd.Timedelta(seconds=index * 20),
                    "end_time": base_time
                    + pd.Timedelta(seconds=index * 20 + 12),
                    "KPIs": {"RSRP": [signal, signal + 1, signal + 2, signal + 3]},
                    "anomalies": {"exists": bool(index % 2)},
                }
            )
        project_config = load_pipeline_config()
        project_config["preprocessing"]["sequence_length"] = 4
        project_config["preprocessing"]["selected_kpis"]["binary_detection"] = [
            "RSRP"
        ]
        model_config = {
            "cross_validation": {"folds": 2, "shuffle_seed": 42},
            "selection": {"primary_metric": "f1_macro"},
            "models": {
                "binary_detection": {
                    "logistic_regression": {
                        "estimator": "logistic_regression",
                        "parameter_grid": {
                            "C": [1.0],
                            "class_weight": ["balanced"],
                        },
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            previous_task_directory = (
                Path(temporary_directory) / "anomaly_type"
            )
            previous_task_directory.mkdir()
            (previous_task_directory / "comparison.json").write_text(
                json.dumps(
                    {
                        "task_name": "anomaly_type",
                        "selected_model": "svm",
                        "group_split": True,
                        "grouping_strategy": "overlapping_time_intervals",
                        "cross_validation_folds": 2,
                        "group_count": 12,
                        "models": [
                            {
                                "model_name": "svm",
                                "estimator": "svm",
                                "parameters": {},
                                "cv_metrics": {
                                    "f1_macro_mean": 0.5,
                                    "f1_macro_std": 0.0,
                                    "precision_macro_mean": 0.5,
                                    "recall_macro_mean": 0.5,
                                },
                                "validation_metrics": {
                                    "f1_macro": 0.5,
                                    "f1_micro": 0.5,
                                    "f1_weighted": 0.5,
                                    "precision_macro": 0.5,
                                    "recall_macro": 0.5,
                                },
                                "test_metrics": {"f1_macro": 0.5},
                                "decision_thresholds": [],
                                "selected": True,
                                "artifact_path": "anomaly_type/svm/model_bundle.joblib",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            results = run_model_comparison_pipeline(
                pd.DataFrame(rows),
                project_config,
                model_config,
                temporary_directory,
                ["binary_detection"],
            )
            comparison_path = Path(temporary_directory) / "comparison.csv"

            self.assertTrue(comparison_path.is_file())
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["task_name"], "binary_detection")
            self.assertTrue(results[0]["group_split"])
            comparison = pd.read_csv(comparison_path)
            self.assertIn("cv_precision_macro_mean", comparison.columns)
            self.assertIn("cv_recall_macro_mean", comparison.columns)
            self.assertGreaterEqual(
                float(comparison.loc[0, "cv_precision_macro_mean"]), 0
            )
            self.assertGreaterEqual(
                float(comparison.loc[0, "cv_recall_macro_mean"]), 0
            )
            self.assertEqual(
                set(comparison["task_name"]),
                {"binary_detection", "anomaly_type"},
            )
            consolidated = json.loads(
                (Path(temporary_directory) / "comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(consolidated), 2)

    def test_cv_model_selection_saves_test_metrics_only_for_selected_model(self) -> None:
        def make_task_splits(task_name: str) -> TaskSplits:
            n_train, n_holdout = 24, 8
            train_X = np.empty((n_train, 4, 1), dtype=object)
            holdout_X = np.empty((n_holdout, 4, 1), dtype=object)
            for index in range(n_train):
                signal = float(index % 2) * 4 + index / 100
                train_X[index, :, 0] = [signal, signal + 1, signal + 2, signal + 3]
            for index in range(n_holdout):
                signal = float(index % 2) * 4 + index / 100
                holdout_X[index, :, 0] = [signal, signal + 1, signal + 2, signal + 3]
            y_train = np.asarray([index % 2 for index in range(n_train)])
            y_holdout = np.asarray([index % 2 for index in range(n_holdout)])
            if task_name == "anomaly_type":
                y_train = np.asarray(
                    [f"type_{index % 3}" for index in range(n_train)]
                )
                y_holdout = np.asarray(
                    [f"type_{index % 3}" for index in range(n_holdout)]
                )
            elif task_name == "temporal_localization":
                y_train = np.tile(np.array([0, 0, 1, 1]), (n_train, 1))
                y_holdout = np.tile(np.array([0, 0, 1, 1]), (n_holdout, 1))
            elif task_name == "affected_kpis":
                labels = [
                    ("RSRP", "RX_Bytes", "TX_Bytes"),
                    ("DL_MCS", "RX_Bytes", "TX_Bytes"),
                    ("RSRP", "DL_MCS", "RX_Bytes", "TX_Bytes"),
                ]
                y_train = np.empty(n_train, dtype=object)
                y_holdout = np.empty(n_holdout, dtype=object)
                y_train[:] = [labels[index % 3] for index in range(n_train)]
                y_holdout[:] = [labels[index % 3] for index in range(n_holdout)]
            return TaskSplits(
                task_name=task_name,
                X_train=train_X,
                y_train=y_train,
                positions_train=np.arange(n_train),
                X_validation=holdout_X,
                y_validation=y_holdout,
                positions_validation=np.arange(n_train, n_train + n_holdout),
                X_test=holdout_X.copy(),
                y_test=y_holdout.copy(),
                positions_test=np.arange(n_train + n_holdout, n_train + 2 * n_holdout),
                feature_channels=("RSRP",),
                groups_train=np.arange(n_train),
                groups_validation=np.arange(n_train, n_train + n_holdout),
                groups_test=np.arange(n_train + n_holdout, n_train + 2 * n_holdout),
                stratify_groups_train=(
                    np.asarray([f"type_{i % 3}" for i in range(n_train)])
                    if task_name == "temporal_localization"
                    else None
                ),
            )

        project_config = load_pipeline_config()
        project_config["training"]["random_forest"][
            "max_temporal_training_timesteps"
        ] = 100
        model_config = {
            "cross_validation": {"folds": 2, "shuffle_seed": 42},
            "selection": {"primary_metric": "f1_macro"},
            "models": {},
        }
        algorithm_mapping = {
            "binary_detection": ["logistic_regression", "random_forest", "xgboost"],
            "anomaly_type": ["logistic_regression", "random_forest", "svm", "xgboost"],
            "temporal_localization": [
                "logistic_regression",
                "random_forest",
                "xgboost",
            ],
            "affected_kpis": [
                "one_vs_rest_logistic_regression",
                "random_forest",
                "classifier_chain",
            ],
        }
        estimator_mapping = {
            "one_vs_rest_logistic_regression": "one_vs_rest_logistic_regression",
            "classifier_chain": "classifier_chain",
        }
        for task_name, algorithms in algorithm_mapping.items():
            model_config["models"][task_name] = {}
            for algorithm in algorithms:
                estimator = estimator_mapping.get(algorithm, algorithm)
                if estimator == "logistic_regression" and algorithm.startswith(
                    "one_vs_rest"
                ):
                    estimator = "one_vs_rest_logistic_regression"
                parameters = (
                    {"n_estimators": [3], "max_depth": [3]}
                    if estimator in {"random_forest", "xgboost"}
                    else {"C": [1.0], "class_weight": ["balanced"]}
                )
                if estimator == "xgboost":
                    parameters["learning_rate"] = [0.2]
                if estimator == "svm":
                    parameters = {
                        "C": [1.0],
                        "kernel": ["linear"],
                        "class_weight": ["balanced"],
                    }
                model_config["models"][task_name][algorithm] = {
                    "estimator": estimator,
                    "parameter_grid": parameters,
                }

        with tempfile.TemporaryDirectory() as temporary_directory:
            for task_name in algorithm_mapping:
                result = run_task_model_comparison(
                    task_name,
                    make_task_splits(task_name),
                    project_config,
                    model_config,
                    temporary_directory,
                )
                self.assertEqual(len(result["models"]), len(algorithm_mapping[task_name]))
                self.assertEqual(
                    sum(model["selected"] for model in result["models"]), 1
                )
                self.assertTrue(result["test_metrics"])
                self.assertTrue(
                    all(
                        model["test_metrics"] is None
                        for model in result["models"]
                        if not model["selected"]
                    )
                )
                if task_name == "binary_detection":
                    selected_result = next(
                        model for model in result["models"] if model["selected"]
                    )
                    artifact = load_task_artifact(
                        selected_result["artifact_path"]
                    )
                    inference_frame = pd.DataFrame(
                        {
                            "KPIs": [
                                {"RSRP": [0.0, 1.0, 2.0, 3.0]},
                                {"RSRP": [4.0, 5.0, 6.0, 7.0]},
                            ]
                        }
                    )
                    predictions = predict_task_component(
                        inference_frame, artifact
                    )
                    self.assertEqual(predictions.predictions.shape, (2,))
                    self.assertTrue(artifact.decision_thresholds)


if __name__ == "__main__":
    unittest.main()

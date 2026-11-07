"""Focused tests for train-only preprocessing and artifact-based inference."""

import ast
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
from src.evaluation.holdout import (
    build_holdout_manifest,
    config_fingerprint,
    dataset_fingerprint,
    validate_holdout_manifest,
)
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
    _decode_json_value,
    _resolve_frame,
    batch_inference_component,
    evaluate_registered_model_component,
    register_model_component,
)
from pipelines.mlrun_project import configure_mlrun_project, validate_source
from pipelines.mlrun_registry import _selected_bundle, register_local_models
from pipelines.mlrun_batch import register_and_run_batch_pipeline


class PipelineComponentTests(unittest.TestCase):
    def test_mlrun_workflows_only_define_inference_and_evaluation(self) -> None:
        workflow_path = (
            Path(__file__).resolve().parents[1]
            / "pipelines"
            / "mlrun_workflow.py"
        )
        module = ast.parse(workflow_path.read_text(encoding="utf-8"))
        pipeline_definitions = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "pipeline"
                for decorator in node.decorator_list
            )
        ]
        self.assertEqual(
            {definition.name for definition in pipeline_definitions},
            {"inference_pipeline", "evaluation_pipeline"},
        )

    def test_mlrun_setup_registers_only_post_training_functions_and_workflows(self) -> None:
        project = unittest.mock.Mock()
        project.name = "ina-test"
        mlrun_module = SimpleNamespace(
            get_or_create_project=unittest.mock.Mock(return_value=project)
        )
        with patch.dict("sys.modules", {"mlrun": mlrun_module}):
            configured = configure_mlrun_project(
                source="git://github.com/example/project.git",
                project_name="ina-test",
                base_image="mlrun/mlrun",
                image=None,
            )

        self.assertIs(configured, project)
        self.assertEqual(
            [call.kwargs["handler"] for call in project.set_function.call_args_list],
            [
                "load_inference_dataset_component",
                "batch_inference_component",
                "evaluate_registered_model_component",
            ],
        )
        self.assertEqual(
            [call.kwargs["handler"] for call in project.set_workflow.call_args_list],
            ["inference_pipeline", "evaluation_pipeline"],
        )
        self.assertEqual(project.set_function.call_count, 3)
        project.build_image.assert_called_once_with()

    def test_holdout_manifest_rejects_dataset_config_and_split_changes(self) -> None:
        frame = pd.DataFrame({"KPIs": [[{"RSRP": [1.0]}], [{"RSRP": [2.0]}]]})
        config = {"seed": 7}
        splits = SimpleNamespace(
            positions_test=np.asarray([1]),
            y_test=np.asarray([1]),
            groups_test=np.asarray(["scenario-b"]),
        )
        with patch(
            "src.evaluation.holdout.prepare_task_data_component",
            return_value=splits,
        ):
            manifest = build_holdout_manifest(
                frame=frame,
                task_name="binary_detection",
                config=config,
                dataset_name="example/dataset",
                source_split="train",
                code_revision="a" * 40,
            )
        self.assertEqual(manifest["test_positions"], [1])
        with patch(
            "src.evaluation.holdout.prepare_task_data_component",
            return_value=splits,
        ):
            self.assertIs(
                validate_holdout_manifest(
                    frame=frame,
                    manifest=manifest,
                    task_name="binary_detection",
                    config=config,
                ),
                splits,
            )

        with self.assertRaisesRegex(ValueError, "source dataset"):
            validate_holdout_manifest(
                frame=frame.assign(KPIs=[[{"RSRP": [9.0]}], [{"RSRP": [2.0]}]]),
                manifest=manifest,
                task_name="binary_detection",
                config=config,
            )
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_holdout_manifest(
                frame=frame,
                manifest=manifest,
                task_name="binary_detection",
                config={"seed": 8},
            )
        with patch(
            "src.evaluation.holdout.prepare_task_data_component",
            return_value=SimpleNamespace(
                positions_test=np.asarray([0]),
                y_test=np.asarray([1]),
                groups_test=None,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "test split"):
                validate_holdout_manifest(
                    frame=frame,
                    manifest=manifest,
                    task_name="binary_detection",
                    config=config,
                )

    def test_mlrun_json_values_accept_serialized_and_decoded_values(self) -> None:
        self.assertEqual(_decode_json_value('["RSRP"]', "kpis"), ["RSRP"])
        self.assertEqual(_decode_json_value(["RSRP"], "kpis"), ["RSRP"])
        self.assertEqual(_decode_json_value('{"seed": 7}', "config"), {"seed": 7})
        self.assertEqual(_decode_json_value({"seed": 7}, "config"), {"seed": 7})
        with self.assertRaisesRegex(TypeError, "JSON string, object, or array"):
            _decode_json_value(7, "config")
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            _decode_json_value("{", "config")

    def test_mlrun_batch_inference_logs_json_predictions(self) -> None:
        class Context:
            def __init__(self) -> None:
                self.results = {}
                self.artifacts = []
                self.labels = {}

            def set_label(self, key: str, value: str) -> None:
                self.labels[key] = value

            def log_result(self, key: str, value: object) -> None:
                self.results[key] = value

            def log_artifact(self, item: str, **kwargs: object) -> SimpleNamespace:
                self.artifacts.append((item, kwargs))
                self.artifact_payload = Path(kwargs["local_path"]).read_text(
                    encoding="utf-8"
                )
                return SimpleNamespace(uri="store://test/predictions")

        model = SimpleNamespace(task_name="binary_detection", model_name="random_forest")
        prediction = SimpleNamespace(
            positions=np.asarray([0, 2]),
            predictions=np.asarray([1, 0]),
        )
        context = Context()
        with (
            patch(
                "pipelines.mlrun_components._resolve_model_artifact",
                return_value=model,
            ),
            patch(
                "pipelines.mlrun_components.predict_task_component",
                return_value=prediction,
            ),
        ):
            result = batch_inference_component(
                context,
                dataset=pd.DataFrame({"KPIs": [{}, {}, {}]}),
                model_uri="store://models/test/model",
            )

        self.assertEqual(result["prediction_count"], 2)
        self.assertEqual(result["predictions_uri"], "store://test/predictions")
        self.assertEqual(context.labels["task"], "binary_detection")
        item, _ = context.artifacts[0]
        self.assertEqual(item, "predictions")
        payload = json.loads(context.artifact_payload)
        self.assertEqual(
            payload["predictions"],
            [
                {"position": 0, "prediction": 1},
                {"position": 2, "prediction": 0},
            ],
        )

    def test_mlrun_registration_logs_selected_bundle_and_holdout_provenance(self) -> None:
        class Context:
            def __init__(self) -> None:
                self.results = {}
                self.artifacts = []
                self.models = []
                self.labels = {}

            def set_label(self, key: str, value: str) -> None:
                self.labels[key] = value

            def log_result(self, key: str, value: object) -> None:
                self.results[key] = value

            def log_model(self, **kwargs: object) -> SimpleNamespace:
                self.models.append(kwargs)
                return SimpleNamespace(uri="store://models/binary_detection:v1")

            def log_artifact(self, item: str, **kwargs: object) -> SimpleNamespace:
                self.artifacts.append((item, kwargs))
                return SimpleNamespace(uri=f"store://artifacts/{item}:v1")

        model = SimpleNamespace(
            selected=True,
            task_name="binary_detection",
            model_name="random_forest",
            cv_metrics={"f1_macro_mean": 0.8},
            validation_metrics={"f1_macro": 0.75},
            test_metrics=None,
            model_parameters={"n_estimators": 20},
            config={"project": {"seed": 7}, "model_selection": {}},
        )
        manifest = {
            "task_name": "binary_detection",
            "dataset_name": "example/dataset",
            "source_split": "train",
            "dataset_sha256": "abc123",
            "project_config_sha256": config_fingerprint({"seed": 7}),
            "code_revision": "a" * 40,
            "test_positions": [5, 6],
            "test_sample_count": 2,
            "grouping_strategy": "overlapping_time_intervals",
        }
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "model_bundle.joblib"
            model_path.write_bytes(b"trusted test file")
            comparison_path = Path(temporary) / "comparison.json"
            comparison_path.write_text(
                json.dumps(
                    {
                        "task_name": "binary_detection",
                        "models": [
                            {
                                "model_name": "random_forest",
                                "selected": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            context = Context()
            with patch(
                "pipelines.mlrun_components.load_task_artifact",
                return_value=model,
            ):
                result = register_model_component(
                    context,
                    model_path=str(model_path),
                    comparison_path=str(comparison_path),
                    holdout_manifest_json=manifest,
                    dataset_name="example/dataset",
                    source_split="train",
                    code_revision="a" * 40,
                    registration_tag="a" * 12 + "-20261107120000",
                )

        self.assertEqual(
            result["model_uri"],
            "store://models/binary_detection:v1",
        )
        self.assertEqual(
            result["holdout_manifest_uri"],
            "store://artifacts/binary_detection_holdout_manifest:v1",
        )
        self.assertEqual(context.models[0]["key"], "binary_detection")
        self.assertEqual(context.models[0]["algorithm"], "random_forest")
        self.assertEqual(context.models[0]["metrics"]["cv_f1_macro_mean"], 0.8)
        self.assertEqual(
            context.models[0]["extra_data"]["dataset_sha256"],
            "abc123",
        )
        self.assertEqual(
            [item for item, _ in context.artifacts],
            ["binary_detection_holdout_manifest", "binary_detection_comparison"],
        )

    def test_mlrun_evaluation_scores_registered_model_without_training(self) -> None:
        class Context:
            def __init__(self) -> None:
                self.results = {}
                self.artifacts = []
                self.labels = {}

            def set_label(self, key: str, value: str) -> None:
                self.labels[key] = value

            def log_result(self, key: str, value: object) -> None:
                self.results[key] = value

            def log_artifact(self, item: str, **kwargs: object) -> SimpleNamespace:
                self.artifacts.append((item, kwargs))
                return SimpleNamespace(uri="store://test/evaluation")

        groups_test = np.asarray(["scenario-a", "scenario-b"])
        splits = TaskSplits(
            task_name="binary_detection",
            X_train=np.empty((1, 1, 1), dtype=object),
            y_train=np.asarray([0]),
            positions_train=np.asarray([0]),
            X_validation=np.empty((1, 1, 1), dtype=object),
            y_validation=np.asarray([1]),
            positions_validation=np.asarray([1]),
            X_test=np.empty((2, 1, 1), dtype=object),
            y_test=np.asarray([0, 1]),
            positions_test=np.asarray([2, 3]),
            feature_channels=("RSRP",),
            groups_test=groups_test,
        )
        model = SimpleNamespace(
            task_name="binary_detection",
            model_name="random_forest",
            decision_thresholds=(0.5,),
        )
        prediction = SimpleNamespace(predictions=np.asarray([0, 1]))
        context = Context()
        frame = pd.DataFrame({"KPIs": [{}, {}, {}, {}]})
        with (
            patch(
                "pipelines.mlrun_components._resolve_model_artifact",
                return_value=model,
            ),
            patch(
                "pipelines.mlrun_components._resolve_json_artifact",
                return_value={
                    "dataset_name": "example/dataset",
                    "source_split": "train",
                    "grouping_strategy": "overlapping_time_intervals",
                    "dataset_sha256": "abc",
                    "code_revision": "revision",
                },
            ),
            patch(
                "pipelines.mlrun_components.validate_holdout_manifest",
                return_value=splits,
            ) as prepare,
            patch(
                "pipelines.mlrun_components.predict_task_component",
                return_value=prediction,
            ) as predict,
        ):
            result = evaluate_registered_model_component(
                context,
                dataset=frame,
                model_uri="store://models/test/model",
                holdout_manifest_uri="store://artifacts/holdout",
                dataset_name="example/dataset",
                source_split="train",
                project_config_json={},
            )

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["metrics"]["f1_macro"], 1.0)
        self.assertEqual(context.results["test_f1_macro"], 1.0)
        self.assertEqual(result["evaluation_uri"], "store://test/evaluation")
        self.assertEqual(prepare.call_count, 1)
        self.assertIs(prepare.call_args.kwargs["frame"], frame)
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(context.artifacts[0][0], "evaluation")

    def test_mlrun_source_rejects_embedded_credentials(self) -> None:
        validate_source("git://github.com/example/project.git")
        with self.assertRaisesRegex(ValueError, "credentials"):
            validate_source("https://user:secret@example.com/project.git")
        with self.assertRaisesRegex(ValueError, "query parameters"):
            validate_source("https://example.com/project.zip?token=secret")

    def test_mlrun_registry_runs_registration_for_selected_local_bundle(self) -> None:
        project = unittest.mock.Mock()
        project.run_function.return_value = SimpleNamespace(
            uid="registration-run",
            outputs={"model_uri": "store://models/binary_detection:v1"},
        )
        mlrun_module = SimpleNamespace(
            get_or_create_project=unittest.mock.Mock(return_value=project)
        )
        frame = pd.DataFrame({"KPIs": [{}]})
        manifest = {
            "task_name": "binary_detection",
            "dataset_name": "example/dataset",
            "source_split": "train",
            "dataset_sha256": dataset_fingerprint(frame),
            "project_config_sha256": config_fingerprint(
                load_pipeline_config()
            ),
            "test_positions": [0],
            "test_sample_count": 1,
            "grouping_strategy": "stratified_samples",
        }
        with tempfile.TemporaryDirectory() as temporary:
            comparison_path = Path(temporary) / "comparison.json"
            comparison_path.write_text(
                json.dumps(
                    {
                        "task_name": "binary_detection",
                        "holdout_provenance": {
                            "dataset_name": "example/dataset",
                            "source_split": "train",
                            **{
                                key: manifest[key]
                                for key in (
                                "dataset_sha256",
                                "project_config_sha256",
                                "test_positions",
                                "test_sample_count",
                                "grouping_strategy",
                                )
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.dict("sys.modules", {"mlrun": mlrun_module}),
                patch(
                    "pipelines.mlrun_registry.load_training_frame",
                    return_value=frame,
                ),
                patch(
                    "pipelines.mlrun_registry._git_revision",
                    return_value="a" * 40,
                ),
                patch(
                    "pipelines.mlrun_registry._selected_bundle",
                    return_value=(
                        Path("selected.joblib"),
                        comparison_path,
                        "random_forest",
                    ),
                ),
                patch(
                    "pipelines.mlrun_registry.build_holdout_manifest",
                    return_value=manifest,
                ) as build_manifest,
            ):
                result = register_local_models(
                    source="git://github.com/example/project.git",
                    project_name="ina-test",
                    dataset_name="example/dataset",
                    source_split="train",
                    artifact_directory=Path("models"),
                    tasks=["binary_detection"],
                )

        self.assertEqual(result[0]["run_id"], "registration-run")
        project.set_source.assert_called_once_with(
            source="git://github.com/example/project.git",
            pull_at_runtime=False,
        )
        self.assertEqual(
            project.set_function.call_args.kwargs["handler"],
            "register_model_component",
        )
        self.assertEqual(
            project.run_function.call_args.args[0],
            "ina-model-registration",
        )
        run_params = project.run_function.call_args.kwargs["params"]
        self.assertEqual(run_params["dataset_name"], "example/dataset")
        self.assertEqual(run_params["source_split"], "train")
        self.assertEqual(
            json.loads(run_params["holdout_manifest_json"]),
            manifest,
        )
        self.assertIs(build_manifest.call_args.kwargs["frame"], frame)

    def test_mlrun_batch_launcher_passes_registered_model_and_holdout(self) -> None:
        project = unittest.mock.Mock()
        project.run.return_value = SimpleNamespace(uid="evaluation-run")
        mlrun_module = SimpleNamespace(
            get_or_create_project=unittest.mock.Mock(return_value=project)
        )
        with patch.dict("sys.modules", {"mlrun": mlrun_module}):
            run = register_and_run_batch_pipeline(
                operation="evaluation",
                project_name="ina-test",
                dataset_name="example/dataset",
                split="train",
                model_uri="store://models/model:v1",
                project_config={"seed": 7},
                holdout_manifest_uri="store://artifacts/holdout:v1",
                watch=False,
            )

        self.assertEqual(run.uid, "evaluation-run")
        arguments = project.run.call_args.kwargs["arguments"]
        self.assertEqual(arguments["model_uri"], "store://models/model:v1")
        self.assertEqual(
            arguments["holdout_manifest_uri"],
            "store://artifacts/holdout:v1",
        )
        self.assertEqual(json.loads(arguments["project_config_json"]), {"seed": 7})
        self.assertEqual(
            project.run.call_args.kwargs["workflow_handler"],
            "evaluation_pipeline",
        )
        with patch.dict("sys.modules", {"mlrun": mlrun_module}):
            with self.assertRaisesRegex(ValueError, "holdout manifest"):
                register_and_run_batch_pipeline(
                    operation="evaluation",
                    project_name="ina-test",
                    dataset_name="example/dataset",
                    split="train",
                    model_uri="store://models/model:v1",
                    project_config={},
                    holdout_manifest_uri=None,
                    watch=False,
                )

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
                dataset_name="example/dataset",
                source_split="train",
            )
            comparison_path = Path(temporary_directory) / "comparison.csv"

            self.assertTrue(comparison_path.is_file())
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["task_name"], "binary_detection")
            self.assertTrue(results[0]["group_split"])
            holdout = results[0]["holdout_provenance"]
            self.assertEqual(holdout["dataset_name"], "example/dataset")
            self.assertEqual(holdout["source_split"], "train")
            self.assertEqual(
                len(holdout["test_positions"]),
                holdout["test_sample_count"],
            )
            task_comparison = json.loads(
                (
                    Path(temporary_directory)
                    / "binary_detection"
                    / "comparison.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                task_comparison["holdout_provenance"],
                holdout,
            )
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

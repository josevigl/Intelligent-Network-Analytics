"""Cross-validated model selection for the four TelecomTS objectives."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipelines.train import load_training_frame
from src.config import load_pipeline_config
from src.evaluation.holdout import (
    config_fingerprint,
    dataset_fingerprint,
)
from src.models.experiments import run_task_model_comparison
from src.preprocessing.data import TASK_NAMES
from src.preprocessing.pipeline import prepare_task_data_component


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    model_config_path = (
        Path(path) if path is not None else Path(__file__).resolve().parents[1] / "model_config.yaml"
    )
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Model configuration not found: {model_config_path}")
    with model_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Model configuration must be a YAML mapping: {model_config_path}")
    return config


def _comparison_rows(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for model in result["models"]:
            cv_metrics = model["cv_metrics"]
            validation_metrics = model["validation_metrics"]
            test_metrics = model["test_metrics"] or {}
            rows.append(
                {
                    "task_name": result["task_name"],
                    "model_name": model["model_name"],
                    "estimator": model["estimator"],
                    "parameters": json.dumps(
                        model["parameters"], sort_keys=True
                    ),
                    "cv_f1_macro_mean": cv_metrics.get("f1_macro_mean"),
                    "cv_f1_macro_std": cv_metrics.get("f1_macro_std"),
                    "cv_precision_macro_mean": cv_metrics.get(
                        "precision_macro_mean"
                    ),
                    "cv_recall_macro_mean": cv_metrics.get("recall_macro_mean"),
                    "validation_f1_macro": validation_metrics.get("f1_macro"),
                    "validation_f1_micro": validation_metrics.get("f1_micro"),
                    "validation_f1_weighted": validation_metrics.get(
                        "f1_weighted"
                    ),
                    "validation_precision_macro": validation_metrics.get(
                        "precision_macro"
                    ),
                    "validation_recall_macro": validation_metrics.get(
                        "recall_macro"
                    ),
                    "test_f1_macro": test_metrics.get("f1_macro"),
                    "test_f1_micro": test_metrics.get("f1_micro"),
                    "test_f1_weighted": test_metrics.get("f1_weighted"),
                    "decision_thresholds": json.dumps(
                        model["decision_thresholds"]
                    ),
                    "selected": model["selected"],
                    "artifact_path": model["artifact_path"],
                    "group_split": result["group_split"],
                    "grouping_strategy": result["grouping_strategy"],
                    "cross_validation_folds": result["cross_validation_folds"],
                    "group_count": result["group_count"],
                }
            )
    return rows


def run_model_comparison_pipeline(
    frame: pd.DataFrame,
    project_config: dict[str, Any],
    model_config: dict[str, Any],
    artifact_directory: str | Path,
    task_names: list[str] | tuple[str, ...] | None = None,
    evaluate_test: bool = True,
    dataset_name: str | None = None,
    source_split: str | None = None,
) -> list[dict[str, Any]]:
    selected_tasks = tuple(task_names or TASK_NAMES)
    unknown = set(selected_tasks) - set(TASK_NAMES)
    if unknown:
        raise ValueError(f"Unknown task names: {sorted(unknown)}")
    if len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError("Task names must not contain duplicates.")
    if not selected_tasks:
        raise ValueError("At least one task must be selected.")

    results = []
    dataset_sha256 = dataset_fingerprint(frame)
    project_config_sha256 = config_fingerprint(project_config)
    for task_name in selected_tasks:
        print(f"Preparing {task_name} data and leakage-safe splits...", flush=True)
        splits = prepare_task_data_component(frame, task_name, project_config)
        group_count = (
            len(set(splits.groups_train))
            if splits.groups_train is not None
            else 0
        )
        print(
            f"Training {task_name}: {len(splits.y_train)} train samples, "
            f"{group_count} train groups.",
            flush=True,
        )
        result = run_task_model_comparison(
            task_name,
            splits,
            project_config,
            model_config,
            artifact_directory,
            evaluate_test=evaluate_test,
        )
        result["holdout_provenance"] = {
            "dataset_name": dataset_name,
            "source_split": source_split,
            "dataset_sha256": dataset_sha256,
            "project_config_sha256": project_config_sha256,
            "test_positions": [
                int(position) for position in splits.positions_test
            ],
            "test_sample_count": int(len(splits.y_test)),
            "grouping_strategy": (
                "overlapping_time_intervals"
                if splits.groups_test is not None
                else "stratified_samples"
            ),
        }
        task_comparison_path = (
            Path(artifact_directory) / task_name / "comparison.json"
        )
        with task_comparison_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        results.append(result)
        print(
            f"Selected {result['selected_model']} for {task_name} "
            f"using {result['selection_source']}; "
            f"test metrics={result['test_metrics']}",
            flush=True,
        )

    output_directory = Path(artifact_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    persisted_results: dict[str, dict[str, Any]] = {}
    for existing_task in TASK_NAMES:
        comparison_path = output_directory / existing_task / "comparison.json"
        if comparison_path.is_file():
            with comparison_path.open("r", encoding="utf-8") as file:
                task_result = json.load(file)
            if not isinstance(task_result, dict):
                raise ValueError(
                    f"Task comparison must be a JSON object: {comparison_path}"
                )
            persisted_results[existing_task] = task_result
    persisted_results.update(
        {result["task_name"]: result for result in results}
    )
    consolidated_results = [
        persisted_results[task_name]
        for task_name in TASK_NAMES
        if task_name in persisted_results
    ]

    rows = _comparison_rows(consolidated_results)
    if rows:
        comparison_path = output_directory / "comparison.csv"
        with comparison_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output_directory / "comparison.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(consolidated_results, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models-config", type=Path, default=None)
    parser.add_argument("--dataset", default="AliMaatouk/TelecomTS")
    parser.add_argument(
        "--split",
        default="train",
        help="Source dataset split. Keep this at train; holdouts are created locally.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not evaluate the held-out test partition during experimentation.",
    )
    parser.add_argument("--task", action="append", choices=TASK_NAMES, dest="tasks")
    args = parser.parse_args()

    project_config = load_pipeline_config(args.config)
    model_config = load_model_config(args.models_config)
    frame = load_training_frame(args.dataset, args.split)
    output_directory = (
        args.output
        if args.output is not None
        else Path(project_config["output"]["model_artifacts_path"])
    )
    run_model_comparison_pipeline(
        frame,
        project_config,
        model_config,
        output_directory,
        args.tasks,
        evaluate_test=not args.skip_test,
        dataset_name=args.dataset,
        source_split=args.split,
    )


if __name__ == "__main__":
    main()

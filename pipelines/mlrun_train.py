"""Register and execute the TelecomTS model-training workflow in MLRun."""

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

from pipelines.train_models import load_model_config
from src.config import load_pipeline_config
from src.preprocessing.data import TASK_NAMES


RUNTIME_REQUIREMENTS = [
    "datasets>=2.14",
    "iterative-stratification>=0.1.9",
    "joblib>=1.3",
    "numpy>=1.24",
    "pandas>=2.0",
    "PyYAML>=6.0",
    "scikit-learn>=1.3",
    "xgboost>=2.0",
]


def _validate_source(source: str) -> None:
    parsed = urlparse(source)
    if parsed.scheme not in {"git", "https", "http", "file"}:
        raise ValueError(
            "Source must be a git, HTTP(S), or file archive URI that MLRun "
            "can access."
        )
    if (parsed.scheme == "file" and not parsed.path) or (
        parsed.scheme != "file" and not parsed.netloc
    ):
        raise ValueError("The MLRun source URI is missing its repository or file path.")
    if parsed.username or parsed.password:
        raise ValueError(
            "Do not embed repository credentials in the MLRun source URL; "
            "configure repository access in the MLRun environment."
        )
    if parse_qsl(parsed.query):
        raise ValueError(
            "Do not use query parameters in the MLRun source URL; configure "
            "repository access in the MLRun environment instead."
        )


def _selected_kpis(
    project_config: dict[str, Any],
    tasks: list[str],
) -> list[str]:
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Select one or more distinct MLRun task branches.")
    unknown_tasks = set(tasks) - set(TASK_NAMES)
    if unknown_tasks:
        raise ValueError(f"Unknown task names: {sorted(unknown_tasks)}")

    feature_map = project_config["preprocessing"]["selected_kpis"]
    missing_tasks = set(tasks) - set(feature_map)
    if missing_tasks:
        raise ValueError(
            "The project config has no selected KPIs for tasks: "
            f"{sorted(missing_tasks)}"
        )
    return sorted(
        {
            kpi
            for task_name in tasks
            for kpi in feature_map[task_name]
        }
    )


def register_and_run_pipeline(
    source: str,
    project_name: str,
    workflow_name: str,
    base_image: str,
    image: str | None,
    dataset_name: str,
    tasks: list[str],
    project_config: dict[str, Any],
    model_config: dict[str, Any],
    evaluate_test: bool,
    watch: bool,
) -> Any:
    """Create the MLRun project, functions, runtime image, and pipeline run."""
    try:
        import mlrun
    except ImportError as error:
        raise ImportError(
            "MLRun execution requires the optional dependencies. Install with "
            "`python -m pip install -e \".[mlrun,data,models]\"`."
        ) from error

    _validate_source(source)
    selected_kpis = _selected_kpis(project_config, tasks)
    repository_root = Path(__file__).resolve().parents[1]
    project = mlrun.get_or_create_project(
        project_name,
        context=str(repository_root),
    )
    project.set_source(source=source, pull_at_runtime=False)
    build_options: dict[str, Any] = {
        "base_image": base_image,
        "requirements": RUNTIME_REQUIREMENTS,
        "set_as_default": True,
    }
    if image is not None:
        build_options["image"] = image
    project.build_config(**build_options)

    component_path = str(Path("pipelines") / "mlrun_components.py")
    for name, handler in (
        ("ina-telecomts-loader", "load_telecomts_component"),
        ("ina-model-comparison", "train_model_comparison_component"),
    ):
        function_options: dict[str, Any] = {
            "name": name,
            "func": component_path,
            "kind": "job",
            "handler": handler,
            "with_repo": True,
        }
        if image is not None:
            function_options["image"] = image
        function = project.set_function(
            **function_options
        )
        function.save(tag="latest")

    workflow_path = str(Path("pipelines") / "mlrun_workflow.py")
    project.set_workflow(
        name=workflow_name,
        workflow_path=workflow_path,
        workflow_handler="training_pipeline",
    )
    project.save()
    project.build_image()

    return project.run(
        name=workflow_name,
        arguments={
            "dataset_name": dataset_name,
            "split": "train",
            "selected_kpis_json": json.dumps(selected_kpis),
            "task_names": tasks,
            "project_config_json": json.dumps(project_config, sort_keys=True),
            "model_config_json": json.dumps(model_config, sort_keys=True),
            "evaluate_test": evaluate_test,
        },
        watch=watch,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Repository/archive URI accessible to MLRun.",
    )
    parser.add_argument("--project", default="intelligent-network-analytics")
    parser.add_argument("--workflow", default="train-models")
    parser.add_argument("--base-image", default="mlrun/mlrun")
    parser.add_argument(
        "--image",
        default=None,
        help="Optional destination image name in a registry accessible by MLRun.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models-config", type=Path, default=None)
    parser.add_argument("--dataset", default="AliMaatouk/TelecomTS")
    parser.add_argument(
        "--task",
        action="append",
        choices=TASK_NAMES,
        dest="tasks",
        help="Task branch to run; repeat to select multiple. Defaults to all four.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not evaluate held-out test data during this run.",
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Submit the MLRun workflow without waiting for completion.",
    )
    args = parser.parse_args()

    project_config = load_pipeline_config(args.config)
    model_config = load_model_config(args.models_config)
    run = register_and_run_pipeline(
        source=args.source,
        project_name=args.project,
        workflow_name=args.workflow,
        base_image=args.base_image,
        image=args.image,
        dataset_name=args.dataset,
        tasks=args.tasks or list(TASK_NAMES),
        project_config=project_config,
        model_config=model_config,
        evaluate_test=not args.skip_test,
        watch=not args.no_watch,
    )
    run_id = getattr(run, "uid", None) or getattr(run, "id", None)
    print(f"MLRun workflow submitted: {run_id or run}")


if __name__ == "__main__":
    main()

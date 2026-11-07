"""Launch MLRun batch inference or final evaluation for registered models."""

import argparse
import json
from typing import Any

from src.config import load_pipeline_config


def register_and_run_batch_pipeline(
    operation: str,
    project_name: str,
    dataset_name: str,
    split: str,
    model_uri: str,
    project_config: dict[str, Any],
    holdout_manifest_uri: str | None,
    watch: bool,
) -> Any:
    if operation not in {"inference", "evaluation"}:
        raise ValueError(f"Unknown batch operation: {operation!r}.")
    if not isinstance(model_uri, str) or not model_uri.strip():
        raise ValueError("A registered model URI must be provided.")

    try:
        import mlrun
    except ImportError as error:
        raise ImportError(
            "MLRun execution requires the optional dependencies. Install with "
            "`python -m pip install -e \".[mlrun,data,models]\"`."
        ) from error

    project = mlrun.get_or_create_project(project_name)
    workflow_name = (
        "batch-inference" if operation == "inference" else "evaluate-model"
    )
    workflow_handler = (
        "inference_pipeline" if operation == "inference" else "evaluation_pipeline"
    )
    arguments: dict[str, Any] = {
        "dataset_name": dataset_name,
        "split": split,
        "model_uri": model_uri,
    }
    if operation == "evaluation":
        if not isinstance(holdout_manifest_uri, str) or not holdout_manifest_uri:
            raise ValueError("Evaluation requires a registered holdout manifest URI.")
        arguments["holdout_manifest_uri"] = holdout_manifest_uri
        arguments["project_config_json"] = json.dumps(
            project_config,
            sort_keys=True,
        )
    return project.run(
        name=workflow_name,
        workflow_handler=workflow_handler,
        arguments=arguments,
        watch=watch,
    )


def _main(operation: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="intelligent-network-analytics")
    parser.add_argument("--dataset", default="AliMaatouk/TelecomTS")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLRun model artifact URI, copied from the model artifact in the UI.",
    )
    if operation == "evaluation":
        parser.add_argument("--config", default=None)
        parser.add_argument("--holdout-manifest-uri", required=True)
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Submit the workflow and return without waiting for completion.",
    )
    args = parser.parse_args()
    project_config = (
        load_pipeline_config(args.config) if operation == "evaluation" else {}
    )
    run = register_and_run_batch_pipeline(
        operation=operation,
        project_name=args.project,
        dataset_name=args.dataset,
        split=args.split,
        model_uri=args.model_uri,
        project_config=project_config,
        holdout_manifest_uri=(
            args.holdout_manifest_uri
            if operation == "evaluation"
            else None
        ),
        watch=not args.no_watch,
    )
    run_id = getattr(run, "uid", None) or getattr(run, "id", None)
    print(f"MLRun {operation} workflow submitted: {run_id or run}")


def inference_main() -> None:
    _main("inference")


def evaluation_main() -> None:
    _main("evaluation")

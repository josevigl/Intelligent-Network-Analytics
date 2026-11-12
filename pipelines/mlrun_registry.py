"""Register locally trained selected model bundles in the MLRun model registry."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from pipelines.mlrun_project import validate_source
from pipelines.train import load_training_frame
from src.config import load_pipeline_config
from src.evaluation.holdout import build_holdout_manifest, dataset_fingerprint
from src.preprocessing.data import TASK_NAMES


def _git_revision(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if not revision:
        raise RuntimeError("Unable to determine the source Git revision.")
    return revision


def _selected_bundle(
    artifact_directory: Path,
    task_name: str,
) -> tuple[Path, Path, str]:
    comparison_path = artifact_directory / task_name / "comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(
            f"Missing comparison results for {task_name}: {comparison_path}"
        )
    with comparison_path.open("r", encoding="utf-8") as file:
        comparison = json.load(file)
    if not isinstance(comparison, dict) or comparison.get("task_name") != task_name:
        raise ValueError(f"Invalid comparison file for task {task_name!r}.")
    selected = [
        candidate
        for candidate in comparison.get("models", [])
        if candidate.get("selected")
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one selected model in {comparison_path}."
        )
    candidate = selected[0]
    model_name = candidate.get("model_name")
    artifact_path = candidate.get("artifact_path")
    if not isinstance(model_name, str) or not isinstance(artifact_path, str):
        raise ValueError(f"Selected model record is incomplete in {comparison_path}.")

    model_path = (
        artifact_directory / task_name / model_name / "model_bundle.joblib"
    )
    if model_path.is_file():
        return model_path.resolve(), comparison_path, model_name

    normalized_artifact_path = artifact_path.replace("\\", "/")
    model_path = Path(normalized_artifact_path)
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            "Selected model bundle not found. Checked the standard bundle "
            f"path {artifact_directory / task_name / model_name / 'model_bundle.joblib'} "
            f"and comparison path {model_path}."
        )
    return model_path, comparison_path, model_name


def register_local_models(
    source: str,
    project_name: str,
    dataset_name: str,
    source_split: str,
    artifact_directory: str | Path,
    tasks: list[str] | None = None,
    config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run a local MLRun registration function for each selected task bundle."""
    validate_source(source)
    selected_tasks = list(tasks or TASK_NAMES)
    if not selected_tasks or len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError("Select one or more distinct task names.")
    unknown_tasks = set(selected_tasks) - set(TASK_NAMES)
    if unknown_tasks:
        raise ValueError(f"Unknown task names: {sorted(unknown_tasks)}")

    try:
        import mlrun
    except ImportError as error:
        raise ImportError(
            "MLRun registration requires the optional dependencies. Install with "
            "`python -m pip install -e \".[mlrun,data,models]\"`."
        ) from error

    repository_root = Path(__file__).resolve().parents[1]
    artifact_root = Path(artifact_directory)
    if not artifact_root.is_absolute():
        artifact_root = (Path.cwd() / artifact_root).resolve()
    config = load_pipeline_config(config_path)
    frame = load_training_frame(dataset_name, source_split)
    revision = _git_revision(repository_root)
    dataset_sha256 = dataset_fingerprint(frame)
    registration_tag = (
        f"{revision[:12]}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )

    project = mlrun.get_or_create_project(
        project_name,
        context=str(repository_root),
    )
    project.set_source(source=source, pull_at_runtime=False)
    component_path = str(
        (repository_root / "pipelines" / "mlrun_components.py").resolve()
    )
    project.set_function(
        name="ina-model-registration",
        func=component_path,
        kind="job",
        handler="register_model_component",
    ).save(tag="latest")
    project.save()

    registered: list[dict[str, Any]] = []
    for task_name in selected_tasks:
        model_path, comparison_path, model_name = _selected_bundle(
            artifact_root,
            task_name,
        )
        manifest = build_holdout_manifest(
            frame=frame,
            task_name=task_name,
            config=config,
            dataset_name=dataset_name,
            source_split=source_split,
            code_revision=revision,
            dataset_sha256=dataset_sha256,
        )
        with comparison_path.open("r", encoding="utf-8") as file:
            comparison = json.load(file)
        provenance = comparison.get("holdout_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                f"Comparison results for {task_name} do not include the "
                "training holdout provenance. Rerun ina-train-models before "
                "registering this model."
            )
        expected_provenance = {
            "dataset_name": manifest["dataset_name"],
            "source_split": manifest["source_split"],
            "dataset_sha256": manifest["dataset_sha256"],
            "project_config_sha256": manifest["project_config_sha256"],
            "test_positions": manifest["test_positions"],
            "test_sample_count": manifest["test_sample_count"],
            "grouping_strategy": manifest["grouping_strategy"],
        }
        if provenance != expected_provenance:
            raise ValueError(
                f"Current data, configuration, or test partition differs "
                f"from the training provenance for {task_name}."
            )
        run = project.run_function(
            "ina-model-registration",
            local=True,
            params={
                "model_path": str(model_path),
                "comparison_path": str(comparison_path),
                "holdout_manifest_json": json.dumps(manifest, sort_keys=True),
                "dataset_name": dataset_name,
                "source_split": source_split,
                "code_revision": revision,
                "registration_tag": registration_tag,
            },
        )
        outputs = getattr(run, "outputs", None)
        registered.append(
            {
                "task_name": task_name,
                "model_name": model_name,
                "run_id": getattr(run, "uid", None) or getattr(run, "id", None),
                "outputs": outputs,
            }
        )
        print(
            f"Registered {task_name}/{model_name} in MLRun; "
            f"registration run={registered[-1]['run_id'] or run}",
            flush=True,
        )
    return registered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--project", default="intelligent-network-analytics")
    parser.add_argument("--dataset", default="AliMaatouk/TelecomTS")
    parser.add_argument("--split", default="train")
    parser.add_argument("--models-path", type=Path, default=Path("results/models"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--task",
        action="append",
        choices=TASK_NAMES,
        dest="tasks",
        help="Task to register; repeat to register multiple. Defaults to all four.",
    )
    args = parser.parse_args()
    register_local_models(
        source=args.source,
        project_name=args.project,
        dataset_name=args.dataset,
        source_split=args.split,
        artifact_directory=args.models_path,
        tasks=args.tasks,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()

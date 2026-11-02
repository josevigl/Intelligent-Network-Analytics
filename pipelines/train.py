"""Train all or selected TelecomTS task branches from the training split."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_pipeline_config, validate_pipeline_config
from src.preprocessing.data import TASK_NAMES
from pipelines.components import train_task_component


def load_training_frame(
    dataset_name: str = "AliMaatouk/TelecomTS",
    split: str = "train",
) -> pd.DataFrame:
    """Load only the requested source split, defaulting to the EDA train split."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "Loading TelecomTS requires the optional 'data' dependencies."
        ) from error
    dataset = load_dataset(dataset_name, split=split)
    return dataset.to_pandas()


def run_training_pipeline(
    frame: pd.DataFrame,
    config: dict[str, Any],
    artifact_directory: str | Path | None = None,
    task_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run independent task branches in parallel and return saved artifact paths."""
    validate_pipeline_config(config)
    selected_tasks = tuple(task_names or TASK_NAMES)
    unknown_tasks = set(selected_tasks) - set(TASK_NAMES)
    if unknown_tasks:
        raise ValueError(f"Unknown task names: {sorted(unknown_tasks)}")
    if len(set(selected_tasks)) != len(selected_tasks):
        raise ValueError("Task names must not contain duplicates.")
    if not selected_tasks:
        raise ValueError("At least one task must be selected.")

    root = Path(
        artifact_directory
        if artifact_directory is not None
        else config["output"]["model_artifacts_path"]
    )
    max_workers = int(config["training"].get("max_parallel_tasks", 1))
    if max_workers < 1:
        raise ValueError("training.max_parallel_tasks must be at least one.")
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(selected_tasks))
    ) as executor:
        futures = {
            executor.submit(
                train_task_component,
                frame,
                task_name,
                config,
                root / task_name,
            ): task_name
            for task_name in selected_tasks
        }
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                results[task_name] = future.result()
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f"Training branch failed for task {task_name!r}."
                ) from error
    return {task_name: results[task_name] for task_name in selected_tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", default="AliMaatouk/TelecomTS")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--task", action="append", choices=TASK_NAMES, dest="tasks")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    frame = load_training_frame(args.dataset, args.split)
    results = run_training_pipeline(
        frame,
        config,
        artifact_directory=args.output,
        task_names=args.tasks,
    )
    for task_name, result in results.items():
        metrics = ", ".join(
            f"{name}={value:.4f}"
            for name, value in result["validation_metrics"].items()
        )
        print(f"{task_name}: {metrics} | artifact={result['artifact_path']}")


if __name__ == "__main__":
    main()

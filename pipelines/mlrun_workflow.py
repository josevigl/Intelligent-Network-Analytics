"""MLRun DAG for dataset registration and parallel task model comparisons."""

from typing import List

import mlrun
from kfp import dsl


@dsl.pipeline(
    name="ina-model-training",
    description=(
        "Register TelecomTS train data, then compare and register one model "
        "for each selected analytics objective."
    ),
)
def pipeline(
    dataset_name: str,
    split: str,
    selected_kpis_json: str,
    task_names: List[str],
    project_config_json: str,
    model_config_json: str,
    evaluate_test: bool = True,
) -> None:
    dataset_run = mlrun.run_function(
        "ina-telecomts-loader",
        handler="load_telecomts_component",
        params={
            "dataset_name": dataset_name,
            "split": split,
            "selected_kpis_json": selected_kpis_json,
        },
        outputs=["train_dataset"],
    )

    with dsl.ParallelFor(task_names) as task_name:
        mlrun.run_function(
            "ina-model-comparison",
            handler="train_model_comparison_component",
            inputs={"dataset": dataset_run.outputs["train_dataset"]},
            params={
                "task_name": task_name,
                "project_config_json": project_config_json,
                "model_config_json": model_config_json,
                "evaluate_test": evaluate_test,
            },
            outputs=["comparison", "selected_model"],
        )

"""MLRun DAGs for model inference and held-out evaluation."""

import mlrun
from kfp import dsl

from pipelines.mlrun_project import (
    FUNCTION_MEMORY_LIMIT,
    FUNCTION_MEMORY_REQUEST,
)


@dsl.pipeline(
    name="ina-batch-inference",
    description="Run a registered task model over an input batch and log predictions.",
)
def inference_pipeline(
    dataset_name: str,
    split: str,
    model_uri: str,
) -> None:
    task = mlrun.run_function(
        "ina-batch-inference",
        handler="batch_inference_component",
        params={
            "dataset_name": dataset_name,
            "split": split,
            "model_uri": model_uri,
        },
        outputs=["predictions"],
    )
    task.set_memory_request(FUNCTION_MEMORY_REQUEST)
    task.set_memory_limit(FUNCTION_MEMORY_LIMIT)


@dsl.pipeline(
    name="ina-model-evaluation",
    description=(
        "Evaluate a registered model on its data-verified held-out partition "
        "without retraining."
    ),
)
def evaluation_pipeline(
    dataset_name: str,
    split: str,
    model_uri: str,
    holdout_manifest_uri: str,
    project_config_json: str,
) -> None:
    task = mlrun.run_function(
        "ina-model-evaluation",
        handler="evaluate_registered_model_component",
        params={
            "dataset_name": dataset_name,
            "split": split,
            "model_uri": model_uri,
            "holdout_manifest_uri": holdout_manifest_uri,
            "project_config_json": project_config_json,
        },
        outputs=["evaluation"],
    )
    task.set_memory_request(FUNCTION_MEMORY_REQUEST)
    task.set_memory_limit(FUNCTION_MEMORY_LIMIT)

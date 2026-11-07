"""MLRun DAGs for model inference and held-out evaluation."""

import mlrun
from kfp import dsl


@dsl.pipeline(
    name="ina-batch-inference",
    description="Run a registered task model over an input batch and log predictions.",
)
def inference_pipeline(
    dataset_name: str,
    split: str,
    model_uri: str,
) -> None:
    dataset_run = mlrun.run_function(
        "ina-telecomts-inference-loader",
        handler="load_inference_dataset_component",
        params={"dataset_name": dataset_name, "split": split},
        outputs=["inference_dataset"],
    )
    mlrun.run_function(
        "ina-batch-inference",
        handler="batch_inference_component",
        inputs={"dataset": dataset_run.outputs["inference_dataset"]},
        params={"model_uri": model_uri},
        outputs=["predictions"],
    )


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
    dataset_run = mlrun.run_function(
        "ina-telecomts-inference-loader",
        handler="load_inference_dataset_component",
        params={"dataset_name": dataset_name, "split": split},
        outputs=["inference_dataset"],
    )
    mlrun.run_function(
        "ina-model-evaluation",
        handler="evaluate_registered_model_component",
        inputs={"dataset": dataset_run.outputs["inference_dataset"]},
        params={
            "model_uri": model_uri,
            "holdout_manifest_uri": holdout_manifest_uri,
            "dataset_name": dataset_name,
            "source_split": split,
            "project_config_json": project_config_json,
        },
        outputs=["evaluation"],
    )

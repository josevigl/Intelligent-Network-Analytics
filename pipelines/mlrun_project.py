"""Shared MLRun project registration for post-training MLOps workflows."""

from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse


SCIKIT_LEARN_VERSION = "1.9.1"

RUNTIME_REQUIREMENTS = [
    "boto3>=1.26",
    "datasets>=2.14",
    "iterative-stratification>=0.1.9",
    "joblib>=1.3",
    "numpy>=1.24",
    "pandas>=2.0",
    "PyYAML>=6.0",
    f"scikit-learn=={SCIKIT_LEARN_VERSION}",
    "xgboost>=2.0",
]

FUNCTION_MEMORY_REQUEST = "1Gi"
FUNCTION_MEMORY_LIMIT = "4Gi"


def validate_source(source: str) -> None:
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


def configure_mlrun_project(
    source: str,
    project_name: str,
    base_image: str,
    image: str | None,
) -> Any:
    """Register inference/evaluation workflows and build their runtime image."""
    try:
        import mlrun
    except ImportError as error:
        raise ImportError(
            "MLRun execution requires the optional dependencies. Install with "
            "`python -m pip install -e \".[mlrun,data,models]\"`."
        ) from error

    validate_source(source)
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

    component_path = str(
        (repository_root / "pipelines" / "mlrun_components.py").resolve()
    )
    functions = (
        ("ina-batch-inference", "batch_inference_component"),
        ("ina-model-evaluation", "evaluate_registered_model_component"),
    )
    for name, handler in functions:
        options: dict[str, Any] = {
            "name": name,
            "func": component_path,
            "kind": "job",
            "handler": handler,
        }
        if image is not None:
            options["image"] = image
        function = project.set_function(**options)
        function.with_requests(mem=FUNCTION_MEMORY_REQUEST)
        function.with_limits(mem=FUNCTION_MEMORY_LIMIT)
        function.save(tag="latest")

    workflow_path = str(Path("pipelines") / "mlrun_workflow.py")
    for name, handler in (
        ("batch-inference", "inference_pipeline"),
        ("evaluate-model", "evaluation_pipeline"),
    ):
        project.set_workflow(
            name=name,
            workflow_path=workflow_path,
            handler=handler,
        )
    project.save()
    project.build_image()
    return project

"""Configure and build the MLRun project for post-training MLOps workflows."""

import argparse

from pipelines.mlrun_project import configure_mlrun_project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--project", default="intelligent-network-analytics")
    parser.add_argument("--base-image", default="mlrun/mlrun")
    parser.add_argument(
        "--image",
        default=None,
        help="Optional destination image name in a registry accessible by MLRun.",
    )
    args = parser.parse_args()
    project = configure_mlrun_project(
        source=args.source,
        project_name=args.project,
        base_image=args.base_image,
        image=args.image,
    )
    print(f"MLRun MLOps project configured: {project.name}")


if __name__ == "__main__":
    main()

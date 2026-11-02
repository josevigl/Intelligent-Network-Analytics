"""Batch inference from a persisted task model bundle."""

import argparse
import json
from pathlib import Path

from src.inference import load_and_predict_component
from pipelines.train import load_training_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", default="AliMaatouk/TelecomTS")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = load_training_frame(args.dataset, args.split)
    result = load_and_predict_component(frame, args.artifact)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "position": int(position),
            "prediction": prediction.tolist()
            if hasattr(prediction, "tolist")
            else prediction,
        }
        for position, prediction in zip(result.positions, result.predictions)
    ]
    with output.open("w", encoding="utf-8") as file:
        json.dump(
            {"task_name": result.task_name, "predictions": records},
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


if __name__ == "__main__":
    main()

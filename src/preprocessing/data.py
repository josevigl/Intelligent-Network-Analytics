"""Extract task-specific raw KPI sequences and targets from TelecomTS rows."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


KPI_COLUMNS = (
    "DL_BLER",
    "DL_MCS",
    "DL_NumberOfPackets",
    "DL_Protocol",
    "Estimated_UL_Buffer",
    "PRB_Utilization_DL",
    "PRB_Utilization_UL",
    "PRBs_DL_Current",
    "PRBs_UL_Current",
    "RSRP",
    "RX_Bytes",
    "TX_Bytes",
    "UL_BLER",
    "UL_MCS",
    "UL_NPRB",
    "UL_NumberOfPackets",
    "UL_Protocol",
    "UL_SNR",
)

CATEGORICAL_KPIS = ("DL_Protocol", "UL_Protocol")

TASK_FEATURES = {
    "binary_detection": (
        "RSRP",
        "UL_SNR",
        "UL_BLER",
        "DL_BLER",
        "RX_Bytes",
    ),
    "anomaly_type": (
        "RSRP",
        "UL_SNR",
        "DL_MCS",
        "UL_NPRB",
        "PRBs_DL_Current",
        "DL_BLER",
        "UL_NumberOfPackets",
        "Estimated_UL_Buffer",
        "PRB_Utilization_UL",
    ),
    "temporal_localization": (
        "RSRP",
        "TX_Bytes",
        "UL_NPRB",
        "DL_BLER",
        "UL_SNR",
        "RX_Bytes",
        "PRB_Utilization_DL",
        "DL_MCS",
        "UL_NumberOfPackets",
        "Estimated_UL_Buffer",
        "UL_BLER",
    ),
    "affected_kpis": (
        "RSRP",
        "DL_MCS",
        "DL_Protocol",
        "UL_SNR",
        "UL_MCS",
        "UL_NumberOfPackets",
        "PRB_Utilization_UL",
    ),
}

TASK_NAMES = tuple(TASK_FEATURES)


@dataclass(frozen=True)
class TaskDataset:
    """Raw selected KPI windows and their task-specific targets."""

    task_name: str
    feature_channels: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    positions: np.ndarray
    stratify_groups: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.y)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.isscalar(missing) else False


def _get_sequence(value: Any, sequence_length: int) -> list[Any] | None:
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    if len(value) != sequence_length:
        return None
    return list(value)


def _get_anomaly(row: pd.Series) -> dict[str, Any]:
    annotation = row.get("anomalies")
    return annotation if isinstance(annotation, dict) else {}


def _is_anomalous(annotation: dict[str, Any]) -> bool:
    value = annotation.get("exists")
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _temporal_target(
    annotation: dict[str, Any], sequence_length: int
) -> np.ndarray | None:
    duration = annotation.get("anomaly_duration")
    if not isinstance(duration, dict):
        return None

    start = duration.get("start")
    end = duration.get("end")
    if _is_missing(start) or _is_missing(end):
        return None

    try:
        start, end = int(start), int(end)
    except (TypeError, ValueError, OverflowError):
        return None

    if not 0 <= start <= end < sequence_length:
        return None

    target = np.zeros(sequence_length, dtype=np.int64)
    target[start : end + 1] = 1
    return target


def _affected_labels(annotation: dict[str, Any]) -> tuple[str, ...]:
    affected = annotation.get("affected_kpis")
    if not isinstance(affected, (list, tuple, set, np.ndarray)):
        return ()

    labels = {
        value.strip()
        for value in affected
        if isinstance(value, str) and value.strip() in KPI_COLUMNS
    }
    return tuple(sorted(labels))


def _target_for_task(
    task_name: str, annotation: dict[str, Any], sequence_length: int
) -> Any | None:
    anomalous = _is_anomalous(annotation)

    if task_name == "binary_detection":
        value = annotation.get("exists")
        return int(value) if isinstance(value, (bool, np.bool_)) else None

    if not anomalous:
        return None
    if task_name == "anomaly_type":
        value = annotation.get("type")
        return value.strip() if isinstance(value, str) and value.strip() else None
    if task_name == "temporal_localization":
        return _temporal_target(annotation, sequence_length)
    if task_name == "affected_kpis":
        return _affected_labels(annotation) or None
    raise ValueError(f"Unknown task: {task_name}")


def build_task_dataset(
    frame: pd.DataFrame,
    task_name: str,
    feature_channels: tuple[str, ...] | list[str] | None = None,
    sequence_length: int = 128,
) -> TaskDataset:
    """Build one task dataset without fitting any input or target encoders."""
    if task_name not in TASK_FEATURES:
        raise ValueError(
            f"Unknown task {task_name!r}; expected one of {TASK_NAMES}."
        )
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if "KPIs" not in frame.columns or "anomalies" not in frame.columns:
        raise ValueError("frame must contain 'KPIs' and 'anomalies' columns.")

    channels = tuple(
        TASK_FEATURES[task_name]
        if feature_channels is None
        else feature_channels
    )
    unknown_channels = set(channels) - set(KPI_COLUMNS)
    if not channels or unknown_channels:
        raise ValueError(
            f"Invalid KPI channels for {task_name}: {sorted(unknown_channels)}."
        )
    if len(set(channels)) != len(channels):
        raise ValueError("feature_channels must not contain duplicates.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero.")

    samples: list[list[list[Any]]] = []
    targets: list[Any] = []
    positions: list[int] = []
    stratify_groups: list[str] = []

    for position, (_, row) in enumerate(frame.iterrows()):
        kpis = row.get("KPIs")
        if not isinstance(kpis, dict):
            continue

        sample = {}
        for channel in channels:
            values = _get_sequence(kpis.get(channel), sequence_length)
            if values is None:
                break
            sample[channel] = values
        else:
            annotation = _get_anomaly(row)
            target = _target_for_task(task_name, annotation, sequence_length)
            if target is None:
                continue

            samples.append([sample[channel] for channel in channels])
            targets.append(target)
            positions.append(position)
            if task_name == "temporal_localization":
                stratify_groups.append(str(annotation["type"]))

    if not samples:
        raise ValueError(f"No valid samples found for task {task_name!r}.")

    if any(channel in CATEGORICAL_KPIS for channel in channels):
        X = np.empty(
            (len(samples), sequence_length, len(channels)),
            dtype=object,
        )
    else:
        X = np.empty(
            (len(samples), sequence_length, len(channels)),
            dtype=np.float32,
        )
    for channel_index, channel in enumerate(channels):
        channel_samples = [sample[channel_index] for sample in samples]
        if channel in CATEGORICAL_KPIS:
            X[:, :, channel_index] = channel_samples
        else:
            flattened_values = [
                value for sample in channel_samples for value in sample
            ]
            numeric_values = pd.to_numeric(
                pd.Series(flattened_values, dtype="object"),
                errors="coerce",
            ).to_numpy(dtype=np.float32, copy=True)
            X[:, :, channel_index] = numeric_values.reshape(
                len(samples), sequence_length
            )
    if task_name == "temporal_localization":
        y = np.stack(targets).astype(np.int64)
    elif task_name == "affected_kpis":
        y = np.empty(len(targets), dtype=object)
        y[:] = targets
    elif task_name == "binary_detection":
        y = np.asarray(targets, dtype=np.int64)
    else:
        y = np.asarray(targets, dtype=str)

    groups = (
        np.asarray(stratify_groups, dtype=str)
        if task_name == "temporal_localization"
        else None
    )
    return TaskDataset(
        task_name=task_name,
        feature_channels=channels,
        X=X,
        y=y,
        positions=np.asarray(positions, dtype=np.int64),
        stratify_groups=groups,
    )


def extract_inference_windows(
    frame: pd.DataFrame,
    feature_channels: tuple[str, ...] | list[str],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract inference windows, returning their original positional indices."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if "KPIs" not in frame.columns:
        raise ValueError("frame must contain a 'KPIs' column.")

    channels = tuple(feature_channels)
    samples: list[list[list[Any]]] = []
    positions: list[int] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        kpis = row.get("KPIs")
        if not isinstance(kpis, dict):
            continue
        sample = {}
        for channel in channels:
            values = _get_sequence(kpis.get(channel), sequence_length)
            if values is None:
                break
            sample[channel] = values
        else:
            samples.append([sample[channel] for channel in channels])
            positions.append(position)

    if not samples:
        raise ValueError("No valid inference KPI windows were found.")
    windows = np.asarray(samples, dtype=object).transpose(0, 2, 1)
    return windows, np.asarray(positions, dtype=np.int64)

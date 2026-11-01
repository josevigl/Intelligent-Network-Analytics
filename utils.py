from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 128

KPI_COLUMNS = [
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
]

CATEGORICAL_KPIS = [
    "DL_Protocol",
    "UL_Protocol",
]

NUMERICAL_KPIS = [
    kpi for kpi in KPI_COLUMNS
    if kpi not in CATEGORICAL_KPIS
]


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def is_missing_scalar(value):
    """
    Check whether a scalar value is missing.

    Handles None, NaN and pandas missing values.
    """

    if value is None:
        return True

    try:
        result = pd.isna(value)

        if np.isscalar(result):
            return bool(result)

        return False

    except (TypeError, ValueError):
        return False


def get_sequence(value, sequence_length=SEQUENCE_LENGTH):
    """
    Validate and return a KPI sequence.

    Returns None if the value is not a sequence of the
    expected length.
    """

    if not isinstance(value, (list, tuple, np.ndarray)):
        return None

    if len(value) != sequence_length:
        return None

    return list(value)


def get_anomaly_dict(row):
    """
    Safely retrieve the anomaly annotation dictionary.
    """

    anomaly = row.get("anomalies")

    if isinstance(anomaly, dict):
        return anomaly

    return {}


def is_anomalous(anomaly):
    """
    Check whether an anomaly annotation explicitly indicates
    the presence of an anomaly.
    """

    value = anomaly.get("exists")

    return isinstance(value, (bool, np.bool_)) and bool(value)

def load_config(config_path: str = "config.yaml") -> dict:
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"No hay archivo de configuración: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_dict_keys(df, column):
    """
    Returns all unique keys found in dictionaries within a DataFrame column.
    """
    keys = set()

    df[column].dropna().apply(
        lambda x: keys.update(x.keys()) if isinstance(x, dict) else None
    )

    return sorted(keys)



# ============================================================
# 3. KPI DATA PREPARATION
# ============================================================

def get_valid_kpi_data(
    df,
    sequence_length=SEQUENCE_LENGTH
):
    """
    Extract valid KPI sequences from the original DataFrame.

    A sample is considered valid when all 18 KPI channels
    exist and contain sequences of the expected length.

    Returns
    -------
    raw_samples : list
        Valid KPI dictionaries.

    valid_positions : list
        Original positional indices of valid samples.
    """

    raw_samples = []
    valid_positions = []

    for position, (_, row) in enumerate(df.iterrows()):

        kpi_data = row.get("KPIs")

        if not isinstance(kpi_data, dict):
            continue

        sample = {}
        valid_sample = True

        for kpi in KPI_COLUMNS:

            sequence = get_sequence(
                kpi_data.get(kpi),
                sequence_length
            )

            if sequence is None:
                valid_sample = False
                break

            sample[kpi] = sequence

        if valid_sample:
            raw_samples.append(sample)
            valid_positions.append(position)

    return raw_samples, valid_positions


def get_protocol_categories(
    raw_samples,
    categorical_kpis=CATEGORICAL_KPIS
):
    """
    Extract the unique categories for categorical KPI channels.

    Missing values are excluded.
    """

    categories = {}

    for kpi in categorical_kpis:

        unique_values = set()

        for sample in raw_samples:

            for value in sample[kpi]:

                if not is_missing_scalar(value):
                    unique_values.add(str(value))

        categories[kpi] = sorted(unique_values)

    return categories


def build_kpi_tensor(
    raw_samples,
    protocol_categories=None,
    sequence_length=SEQUENCE_LENGTH
):
    """
    Convert raw KPI sequences into a numerical 3D tensor.

    Numerical channels are converted to float values.
    Categorical protocol channels are one-hot encoded.

    Returns
    -------
    X : np.ndarray
        Shape: (n_samples, sequence_length, n_features).

    feature_names : list
        Names of the resulting numerical features.

    protocol_categories : dict
        Categories used for protocol encoding.
    """

    if protocol_categories is None:
        protocol_categories = get_protocol_categories(
            raw_samples
        )

    feature_names = list(NUMERICAL_KPIS)

    for kpi in CATEGORICAL_KPIS:

        for category in protocol_categories[kpi]:

            feature_names.append(
                f"{kpi}_{category}"
            )

    X = np.zeros(
        (
            len(raw_samples),
            sequence_length,
            len(feature_names)
        ),
        dtype=np.float32
    )

    numerical_indices = {
        kpi: index
        for index, kpi in enumerate(NUMERICAL_KPIS)
    }

    categorical_indices = {}

    current_index = len(NUMERICAL_KPIS)

    for kpi in CATEGORICAL_KPIS:

        categorical_indices[kpi] = {
            category: current_index + index
            for index, category in enumerate(
                protocol_categories[kpi]
            )
        }

        current_index += len(protocol_categories[kpi])

    for sample_index, sample in enumerate(raw_samples):

        # Numerical KPI channels
        for kpi in NUMERICAL_KPIS:

            feature_index = numerical_indices[kpi]

            values = pd.to_numeric(
                pd.Series(sample[kpi]),
                errors="coerce"
            ).to_numpy(dtype=np.float32)

            X[sample_index, :, feature_index] = values

        # Categorical KPI channels
        for kpi in CATEGORICAL_KPIS:

            category_mapping = categorical_indices[kpi]

            for timestep, value in enumerate(sample[kpi]):

                if is_missing_scalar(value):
                    continue

                category = str(value)

                feature_index = category_mapping.get(
                    category
                )

                if feature_index is not None:

                    X[
                        sample_index,
                        timestep,
                        feature_index
                    ] = 1.0

    return X, feature_names, protocol_categories


# ============================================================
# 4. ANOMALY ANNOTATION EXTRACTION
# ============================================================

def get_anomaly_info(df, positions):
    """
    Retrieve anomaly annotations for the valid KPI samples.

    The returned annotations preserve the alignment with
    the original DataFrame positional indices.
    """

    anomaly_info = []

    for position in positions:

        row = df.iloc[position]

        anomaly = get_anomaly_dict(row)

        anomaly_info.append(anomaly)

    return anomaly_info


# ============================================================
# 5. TASK 1 - BINARY ANOMALY DETECTION
# ============================================================

def create_binary_dataset(X, positions, anomaly_info):
    """
    Create a binary anomaly detection dataset.

    Target:
        0 = Normal
        1 = Anomalous

    Samples without an explicit boolean 'exists' annotation
    are excluded.
    """

    X_task = []
    y_task = []
    valid_positions = []

    for index, anomaly in enumerate(anomaly_info):

        exists = anomaly.get("exists")

        if not isinstance(exists, (bool, np.bool_)):
            continue

        X_task.append(X[index])

        y_task.append(
            int(bool(exists))
        )

        valid_positions.append(positions[index])

    if not X_task:
        raise ValueError(
            "No valid samples found for binary anomaly detection."
        )

    return {
        "X": np.asarray(X_task, dtype=np.float32),
        "y": np.asarray(y_task, dtype=np.int64),
        "positions": np.asarray(valid_positions),
    }


# ============================================================
# 6. TASK 2 - ANOMALY TYPE CLASSIFICATION
# ============================================================

def create_anomaly_type_dataset(X, positions, anomaly_info):
    """
    Create a multiclass anomaly classification dataset.

    Only anomalous samples with a valid string anomaly type
    are included.
    """

    X_task = []
    y_task = []
    valid_positions = []

    for index, anomaly in enumerate(anomaly_info):

        if not is_anomalous(anomaly):
            continue

        anomaly_type = anomaly.get("type")

        if not isinstance(anomaly_type, str):
            continue

        if not anomaly_type.strip():
            continue

        X_task.append(X[index])
        y_task.append(anomaly_type)
        valid_positions.append(positions[index])

    if not X_task:
        raise ValueError(
            "No valid samples found for anomaly type classification."
        )

    return {
        "X": np.asarray(X_task, dtype=np.float32),
        "y": np.asarray(y_task, dtype=str),
        "positions": np.asarray(valid_positions),
    }


# ============================================================
# 7. TASK 3 - TEMPORAL ANOMALY LOCALIZATION
# ============================================================

def create_temporal_dataset(
    X,
    positions,
    anomaly_info,
    sequence_length=SEQUENCE_LENGTH
):
    """
    Create a temporal localization dataset.

    Each target is a binary sequence with the same length
    as the input KPI sequence.

    Target:
        0 = Normal timestep
        1 = Anomalous timestep

    The start and end indices are treated as inclusive.
    Invalid intervals are excluded rather than clipped.
    """

    X_task = []
    y_task = []
    valid_positions = []

    for index, anomaly in enumerate(anomaly_info):

        if not is_anomalous(anomaly):
            continue

        duration = anomaly.get("anomaly_duration")

        if not isinstance(duration, dict):
            continue

        start = duration.get("start")
        end = duration.get("end")

        if is_missing_scalar(start) or is_missing_scalar(end):
            continue

        try:
            start = int(start)
            end = int(end)

        except (TypeError, ValueError, OverflowError):
            continue

        if not (0 <= start <= end < sequence_length):
            continue

        target = np.zeros(
            sequence_length,
            dtype=np.int64
        )

        target[start:end + 1] = 1

        X_task.append(X[index])
        y_task.append(target)
        valid_positions.append(positions[index])

    if not X_task:
        raise ValueError(
            "No valid samples found for temporal localization."
        )

    return {
        "X": np.asarray(X_task, dtype=np.float32),
        "y": np.asarray(y_task, dtype=np.int64),
        "positions": np.asarray(valid_positions),
    }


# ============================================================
# 8. TASK 4 - AFFECTED KPI IDENTIFICATION
# ============================================================

def create_affected_kpi_dataset(
    X,
    positions,
    anomaly_info
):
    """
    Create a multilabel classification dataset.

    Each sample can have one or more affected KPIs.

    MultiLabelBinarizer converts the KPI labels into a
    binary target matrix.
    """

    X_task = []
    labels_task = []
    valid_positions = []

    for index, anomaly in enumerate(anomaly_info):

        if not is_anomalous(anomaly):
            continue

        affected_kpis = anomaly.get("affected_kpis")

        # Accept lists, tuples, sets and NumPy arrays
        if not isinstance(
            affected_kpis,
            (list, tuple, set, np.ndarray)
        ):
            continue

        labels = [
            str(kpi)
            for kpi in affected_kpis
            if isinstance(kpi, str) and kpi.strip()
        ]

        if not labels:
            continue

        X_task.append(X[index])
        labels_task.append(labels)
        valid_positions.append(positions[index])

    if not X_task:
        raise ValueError(
            "No valid samples found for affected KPI identification."
        )

    mlb = MultiLabelBinarizer()

    y_task = mlb.fit_transform(labels_task)

    return {
        "X": np.asarray(X_task, dtype=np.float32),
        "y": np.asarray(y_task, dtype=np.int64),
        "positions": np.asarray(valid_positions),
        "label_classes": mlb.classes_,
        "label_binarizer": mlb,
    }


# ============================================================
# 9. COMPLETE DATASET PREPARATION
# ============================================================

def prepare_supervised_datasets(
    df,
    protocol_categories=None,
    sequence_length=SEQUENCE_LENGTH
):
    """
    Prepare the KPI tensor and all four supervised datasets.

    Returns
    -------
    results : dict
        Contains:
            - X_all
            - valid_positions
            - feature_names
            - protocol_categories
            - binary_detection
            - anomaly_type
            - temporal_localization
            - affected_kpis
    """

    # Extract valid KPI sequences
    raw_samples, valid_positions = get_valid_kpi_data(
        df,
        sequence_length=sequence_length
    )

    if not raw_samples:
        raise ValueError(
            "No valid KPI sequences were found in the DataFrame."
        )

    # Convert KPI sequences into a numerical tensor
    X_all, feature_names, protocol_categories = build_kpi_tensor(
        raw_samples,
        protocol_categories=protocol_categories,
        sequence_length=sequence_length
    )

    # Retrieve aligned anomaly annotations
    anomaly_info = get_anomaly_info(
        df,
        valid_positions
    )

    # Prepare the four supervised datasets
    binary_dataset = create_binary_dataset(
        X_all,
        valid_positions,
        anomaly_info
    )

    type_dataset = create_anomaly_type_dataset(
        X_all,
        valid_positions,
        anomaly_info
    )

    temporal_dataset = create_temporal_dataset(
        X_all,
        valid_positions,
        anomaly_info,
        sequence_length=sequence_length
    )

    affected_kpi_dataset = create_affected_kpi_dataset(
        X_all,
        valid_positions,
        anomaly_info
    )

    return {
        "X_all": X_all,
        "valid_positions": np.asarray(valid_positions),
        "feature_names": feature_names,
        "protocol_categories": protocol_categories,

        "binary_detection": binary_dataset,
        "anomaly_type": type_dataset,
        "temporal_localization": temporal_dataset,
        "affected_kpis": affected_kpi_dataset,
    }


# ============================================================
# 10. DATASET SUMMARY
# ============================================================

def print_dataset_summary(results):
    """
    Print a summary of the prepared datasets.
    """

    print("=" * 65)
    print("NEXUS 5G - SUPERVISED DATASET SUMMARY")
    print("=" * 65)

    print(
        f"\nValid KPI samples: "
        f"{len(results['valid_positions']):,}"
    )

    print(
        f"KPI tensor shape: "
        f"{results['X_all'].shape}"
    )

    print(
        f"Number of input features: "
        f"{len(results['feature_names'])}"
    )

    print("\n" + "-" * 65)
    print("SUPERVISED LEARNING TASKS")
    print("-" * 65)

    task_names = {
        "binary_detection": "Binary Anomaly Detection",
        "anomaly_type": "Anomaly Type Classification",
        "temporal_localization": "Temporal Anomaly Localization",
        "affected_kpis": "Affected KPI Identification",
    }

    for task_key, task_name in task_names.items():

        dataset = results[task_key]

        print(f"\n{task_name}")
        print(f"  X shape: {dataset['X'].shape}")
        print(f"  y shape: {dataset['y'].shape}")
        print(f"  Samples: {len(dataset['y']):,}")

    print("\n" + "=" * 65)


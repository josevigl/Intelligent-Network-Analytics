from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

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

    with open(config_file, "r", encoding="utf-8") as f:
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


# ============================================================
# 11. EXPLORATORY DATA ANALYSIS
# ============================================================

def configure_plotting(graphics):
    """Apply the shared chart style from the project configuration."""
    plt.style.use(graphics.get("style", "dark_background"))
    plt.rcParams.update({
        "figure.figsize": graphics.get("figsize", [8, 5]),
        "figure.facecolor": graphics.get("bg_color", "#000000"),
        "axes.facecolor": graphics.get("bg_color", "#000000"),
        "savefig.facecolor": graphics.get("bg_color", "#000000"),
        "text.color": graphics.get("font_color", "white"),
        "axes.labelcolor": graphics.get("font_color", "white"),
        "xtick.color": graphics.get("font_color", "white"),
        "ytick.color": graphics.get("font_color", "white"),
        "axes.edgecolor": graphics.get("edgecolor", "white"),
        "grid.color": graphics.get("grid_color", "#344050"),
        "axes.grid": True,
        "grid.alpha": 0.35,
    })
    sns.set_palette(graphics.get("palette", ["#004D98", "#A50044"]))


def _validate_kpi_tensor(X, feature_names):
    """Validate a KPI tensor and return it as a numeric NumPy array."""
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("X must have shape (samples, timesteps, features).")
    if values.shape[2] != len(feature_names):
        raise ValueError(
            "The number of feature names must match the tensor feature axis."
        )
    return values


def _window_feature_frame(X, feature_names):
    """Calculate six time-series summaries for every sample and feature."""
    values = _validate_kpi_tensor(X, feature_names)
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    safe_values = np.where(finite, values, 0.0)
    means = np.divide(
        safe_values.sum(axis=1),
        counts,
        out=np.full((len(values), values.shape[2]), np.nan),
        where=counts > 0,
    )

    centered_values = np.where(
        finite, values - means[:, np.newaxis, :], 0.0
    )
    standard_deviations = np.sqrt(
        np.divide(
            np.square(centered_values).sum(axis=1),
            counts,
            out=np.full_like(means, np.nan),
            where=counts > 0,
        )
    )
    minima = np.where(finite, values, np.inf).min(axis=1)
    maxima = np.where(finite, values, -np.inf).max(axis=1)
    minima[counts == 0] = np.nan
    maxima[counts == 0] = np.nan

    timesteps = np.broadcast_to(
        np.arange(values.shape[1], dtype=np.float64)[np.newaxis, :, np.newaxis],
        values.shape,
    )
    mean_timestep = np.divide(
        np.where(finite, timesteps, 0.0).sum(axis=1),
        counts,
        out=np.full_like(means, np.nan),
        where=counts > 0,
    )
    centered_timesteps = np.where(
        finite, timesteps - mean_timestep[:, np.newaxis, :], 0.0
    )
    slope_denominator = np.square(centered_timesteps).sum(axis=1)
    slopes = np.divide(
        (centered_timesteps * centered_values).sum(axis=1),
        slope_denominator,
        out=np.full_like(means, np.nan),
        where=slope_denominator > 0,
    )

    statistics = {
        "mean": means,
        "std": standard_deviations,
        "min": minima,
        "max": maxima,
        "range": maxima - minima,
        "slope": slopes,
    }
    columns = {
        f"{feature}__{statistic}": statistic_values[:, feature_index]
        for feature_index, feature in enumerate(feature_names)
        for statistic, statistic_values in statistics.items()
    }
    return pd.DataFrame(columns)


def summarize_kpi_training_data(X, feature_names):
    """Summarize KPI values per training window, including missingness."""
    values = _validate_kpi_tensor(X, feature_names)
    window_means = _window_feature_frame(X, feature_names)[
        [f"{feature}__mean" for feature in feature_names]
    ]
    summary = pd.DataFrame({
        "mean_window_mean": window_means.mean(),
        "std_window_mean": window_means.std(),
        "median_window_mean": window_means.median(),
        "p05_window_mean": window_means.quantile(0.05),
        "p95_window_mean": window_means.quantile(0.95),
        "missing_rate": 1.0 - np.isfinite(values).mean(axis=(0, 1)),
    })
    summary.index = feature_names
    summary.index.name = "feature"
    return summary


def plot_class_distribution(y, graphics, title, class_names=None):
    """Plot class counts and percentages for a training target."""
    labels = pd.Series(np.asarray(y), name="class")
    counts = labels.value_counts().sort_values(ascending=False)
    names = {
        value: class_names.get(value, str(value))
        if class_names is not None
        else str(value)
        for value in counts.index
    }
    percentages = counts / counts.sum() * 100
    colors = graphics.get(
        "categorical_palette", graphics.get("palette", ["#004D98"])
    )
    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.8), 5))
    bars = ax.bar(
        [names[value] for value in counts.index],
        counts.values,
        color=[colors[index % len(colors)] for index in range(len(counts))],
        edgecolor=graphics.get("edgecolor", "white"),
    )
    offset = max(counts.max() * 0.015, 0.5)
    for bar, count, percentage in zip(bars, counts.values, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + offset,
            f"{count:,}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(title)
    ax.set_ylabel("Training samples")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    return fig


def _selected_feature_names(feature_names, kpis):
    available = set(feature_names)
    selected = [kpi for kpi in kpis if kpi in available]
    if not selected:
        raise ValueError("None of the requested KPIs is present in feature_names.")
    return selected


def _window_means(X, feature_names, kpis):
    selected = _selected_feature_names(feature_names, kpis)
    summary = _window_feature_frame(X, feature_names)
    return summary[[f"{name}__mean" for name in selected]].rename(
        columns={f"{name}__mean": name for name in selected}
    )


def plot_kpi_correlation(X, feature_names, kpis, graphics):
    """Plot correlations of per-window KPI means using training windows."""
    means = _window_means(X, feature_names, kpis)
    correlation = means.corr()
    size = graphics.get("figsize", [8, 5])
    fig, ax = plt.subplots(
        figsize=(
            max(size[0], len(kpis) * 0.75),
            max(size[1], len(kpis) * 0.65),
        )
    )
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("3.2 KPI Correlation (Training Windows)")
    fig.tight_layout()
    return fig


def plot_kpi_distributions(X, feature_names, kpis, graphics):
    """Plot per-window mean distributions for selected KPIs."""
    means = _window_means(X, feature_names, kpis)
    n_columns = 3
    n_rows = int(np.ceil(len(means.columns) / n_columns))
    width, height = graphics.get("figsize", [8, 5])
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(width * 1.7, height * 1.35),
        squeeze=False,
    )
    for ax, feature in zip(axes.flat, means.columns):
        sns.histplot(
            means[feature].dropna(),
            bins=35,
            kde=True,
            color=graphics.get("palette", ["#004D98"])[0],
            ax=ax,
        )
        ax.set_title(feature)
        ax.set_xlabel("Mean value per training window")
    for ax in axes.flat[len(means.columns):]:
        ax.set_visible(False)
    fig.suptitle("3.3 KPI Distributions (Training Windows)", y=1.02)
    fig.tight_layout()
    return fig


def plot_kpi_distributions_by_class(
    X, y, feature_names, kpis, graphics, title, class_names=None
):
    """Compare per-window KPI means across training classes."""
    means = _window_means(X, feature_names, kpis)
    labels = np.asarray(y)
    if len(labels) != len(means):
        raise ValueError("y must have one class label per training window.")
    order = list(pd.Series(labels).value_counts().index)
    display_names = {
        value: class_names.get(value, str(value))
        if class_names is not None
        else str(value)
        for value in order
    }
    plot_data = means.copy()
    plot_data["class"] = [display_names[value] for value in labels]
    display_order = [display_names[value] for value in order]
    n_columns = 2
    n_rows = int(np.ceil(len(means.columns) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(16, max(5, n_rows * 4.5)),
        squeeze=False,
    )
    colors = graphics.get(
        "categorical_palette", graphics.get("palette", ["#004D98"])
    )
    palette = {
        name: colors[index % len(colors)]
        for index, name in enumerate(display_order)
    }
    for ax, feature in zip(axes.flat, means.columns):
        sns.boxplot(
            data=plot_data,
            x="class",
            y=feature,
            hue="class",
            order=display_order,
            palette=palette,
            dodge=False,
            legend=False,
            showfliers=False,
            ax=ax,
        )
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=55)
    for ax in axes.flat[len(means.columns):]:
        ax.set_visible(False)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def plot_kpi_temporal_profiles(
    X,
    feature_names,
    kpis,
    graphics,
    labels=None,
    class_order=None,
    title="3.4 Temporal KPI Profiles",
):
    """Plot mean KPI trajectories and standard-deviation bands."""
    values = _validate_kpi_tensor(X, feature_names)
    selected = _selected_feature_names(feature_names, kpis)
    feature_indices = [feature_names.index(name) for name in selected]
    if labels is None:
        labels = np.repeat("All training windows", len(values))
    labels = np.asarray(labels)
    if len(labels) != len(values):
        raise ValueError("labels must have one value per training window.")
    if class_order is None:
        class_order = list(pd.unique(labels))
    colors = graphics.get("categorical_palette", graphics.get("palette", []))
    color_by_class = dict(zip(class_order, colors))
    width, height = graphics.get("figsize", [8, 5])
    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(width * 1.5, height * max(1.0, len(selected) * 0.55)),
        squeeze=False,
    )
    timesteps = np.arange(values.shape[1])
    for ax, feature, feature_index in zip(
        axes.flat, selected, feature_indices
    ):
        for class_name in class_order:
            mask = labels == class_name
            group_values = values[mask, :, feature_index]
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(group_values, axis=0)
                std = np.nanstd(group_values, axis=0)
            color = color_by_class.get(class_name)
            ax.plot(timesteps, mean, label=str(class_name), color=color)
            if len(group_values) > 1:
                ax.fill_between(
                    timesteps,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=graphics.get("confidence_alpha", 0.16),
                    linewidth=0,
                )
        ax.set_title(feature)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("KPI value")
    if len(class_order) <= 12:
        axes.flat[0].legend(
            title="Training class", bbox_to_anchor=(1.02, 1), loc="upper left"
        )
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def _fit_importance_model(X, y, feature_names, seed, stratify=True):
    """Fit an exploratory forest using a holdout drawn only from training data."""
    features = _window_feature_frame(X, feature_names)
    features = features.loc[:, features.notna().any(axis=0)]
    if features.empty:
        raise ValueError("No non-missing window features are available.")

    labels = np.asarray(y)
    split_labels = labels if stratify and labels.ndim == 1 else None
    X_fit, X_eval, y_fit, y_eval = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=split_labels,
    )
    imputer = SimpleImputer(strategy="median")
    X_fit = imputer.fit_transform(X_fit)
    X_eval = imputer.transform(X_eval)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample" if labels.ndim == 1 else None,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit)
    return model, imputer, X_eval, y_eval, list(features.columns)


def plot_random_forest_importance(
    X, y, feature_names, graphics, title, seed
):
    """Compare forest impurity and permutation importance on an inner train holdout."""
    model, imputer, X_eval, y_eval, columns = _fit_importance_model(
        X, y, feature_names, seed
    )
    impurity = pd.Series(
        model.feature_importances_, index=columns, name="impurity"
    ).nlargest(20)
    rng = np.random.default_rng(seed)
    sample_size = min(2500, len(y_eval))
    sample = rng.choice(len(y_eval), size=sample_size, replace=False)
    permutation = permutation_importance(
        model,
        X_eval[sample],
        y_eval[sample],
        n_repeats=3,
        random_state=seed,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    perm_values = pd.Series(
        permutation.importances_mean,
        index=columns,
        name="permutation",
    ).reindex(impurity.index).sort_values()

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    impurity.sort_values().plot.barh(
        ax=axes[0], color=graphics.get("palette", ["#004D98"])[0]
    )
    axes[0].set_title("Mean decrease in impurity")
    axes[0].set_xlabel("Importance")
    perm_values.plot.barh(
        ax=axes[1], color=graphics.get("palette", ["#004D98", "#A50044"])[1]
    )
    axes[1].axvline(0, color=graphics.get("edgecolor", "white"), linewidth=0.8)
    axes[1].set_title("Permutation importance (inner training holdout)")
    axes[1].set_xlabel("Balanced-accuracy decrease")
    matrix = confusion_matrix(
        y_eval,
        model.predict(X_eval),
        labels=model.classes_,
        normalize="true",
    )
    sns.heatmap(
        matrix,
        annot=len(model.classes_) <= 12,
        fmt=".2f",
        cmap="mako",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=axes[2],
    )
    axes[2].set_title("Normalized confusion matrix (training-only holdout)")
    axes[2].set_xlabel("Predicted class")
    axes[2].set_ylabel("True class")
    axes[2].tick_params(axis="x", rotation=75)
    fig.suptitle(
        f"{title} — training data only",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_pca_projection(X, y, feature_names, graphics, title, seed):
    """Project training-window summaries to two PCA dimensions."""
    features = _window_feature_frame(X, feature_names)
    features = features.loc[:, features.notna().any(axis=0)]
    transformed = SimpleImputer(strategy="median").fit_transform(features)
    transformed = StandardScaler().fit_transform(transformed)
    embedding = PCA(n_components=2, random_state=seed).fit_transform(transformed)
    labels = np.asarray(y)
    classes = list(pd.unique(labels))
    colors = graphics.get(
        "categorical_palette", graphics.get("palette", ["#004D98"])
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    for index, class_name in enumerate(classes):
        mask = labels == class_name
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=str(class_name),
            color=colors[index % len(colors)],
            alpha=0.65,
            s=24,
            edgecolors="none",
        )
    ax.set_title(f"{title} — training windows only")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend(title="Training class", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def _validate_temporal_labels(y, anomaly_types):
    targets = np.asarray(y)
    types = np.asarray(anomaly_types)
    if targets.ndim != 2 or targets.shape[0] != len(types):
        raise ValueError("Temporal labels and anomaly types must align.")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Temporal targets must be binary.")
    return targets, types


def plot_anomaly_duration_distribution(y, anomaly_types, graphics):
    """Plot the distribution of labelled anomaly durations."""
    targets, _ = _validate_temporal_labels(y, anomaly_types)
    durations = targets.sum(axis=1)
    colors = graphics.get("categorical_palette", graphics.get("palette", []))
    fig, ax = plt.subplots()
    sns.histplot(
        durations,
        bins=np.arange(-0.5, targets.shape[1] + 1.5, 4),
        color=colors[0],
        ax=ax,
    )
    ax.set_title("6.1 Anomaly duration")
    ax.set_xlabel("Anomalous timesteps per training window")
    ax.set_ylabel("Training windows")
    fig.tight_layout()
    return fig


def plot_full_partial_distribution(y, anomaly_types, graphics):
    """Plot full and partial anomaly-window counts."""
    targets, _ = _validate_temporal_labels(y, anomaly_types)
    durations = targets.sum(axis=1)
    window_type = np.where(durations == targets.shape[1], "Full", "Partial")
    colors = graphics.get("categorical_palette", graphics.get("palette", []))
    fig, ax = plt.subplots()
    sns.countplot(
        x=window_type,
        hue=window_type,
        order=["Full", "Partial"],
        palette={
            "Full": graphics.get("anomalous_color", "#A50044"),
            "Partial": colors[0],
        },
        legend=False,
        ax=ax,
    )
    ax.set_title("6.2 Full vs partial windows")
    ax.set_xlabel("Window coverage")
    ax.set_ylabel("Training windows")
    fig.tight_layout()
    return fig


def plot_duration_by_anomaly_type(y, anomaly_types, graphics):
    """Compare labelled anomaly durations across training anomaly types."""
    targets, types = _validate_temporal_labels(y, anomaly_types)
    durations = targets.sum(axis=1)
    type_order = list(pd.unique(types))
    colors = graphics.get("categorical_palette", graphics.get("palette", []))
    fig, ax = plt.subplots(figsize=(max(8, len(type_order) * 0.9), 5))
    sns.boxplot(
        x=types,
        y=durations,
        order=type_order,
        color=colors[1 % len(colors)],
        showfliers=False,
        ax=ax,
    )
    ax.set_title("6.3 Duration by anomaly type")
    ax.tick_params(axis="x", rotation=75)
    ax.set_xlabel("Anomaly type")
    ax.set_ylabel("Anomalous timesteps")
    fig.tight_layout()
    return fig


def _anomaly_boundaries(y, anomaly_types):
    targets, _ = _validate_temporal_labels(y, anomaly_types)
    return (
        targets.argmax(axis=1),
        targets.shape[1] - 1 - targets[:, ::-1].argmax(axis=1),
        targets.shape[1],
    )


def plot_anomaly_start_distribution(y, anomaly_types, graphics):
    """Plot the distribution of labelled anomaly start timesteps."""
    starts, _, sequence_length = _anomaly_boundaries(y, anomaly_types)
    bins = np.arange(-0.5, sequence_length + 0.5, 1)
    colors = graphics.get("categorical_palette", graphics.get("palette", []))
    fig, ax = plt.subplots()
    ax.hist(starts, bins=bins, color=colors[0])
    ax.set_title("6.4 Anomaly start timestep")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Training windows")
    fig.tight_layout()
    return fig


def plot_anomaly_end_distribution(y, anomaly_types, graphics):
    """Plot the distribution of labelled anomaly end timesteps."""
    _, ends, sequence_length = _anomaly_boundaries(y, anomaly_types)
    bins = np.arange(-0.5, sequence_length + 0.5, 1)
    colors = graphics.get("categorical_palette", graphics.get("palette", []))
    fig, ax = plt.subplots()
    ax.hist(ends, bins=bins, color=colors[1 % len(colors)])
    ax.set_title("6.5 Anomaly end timestep")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Training windows")
    fig.tight_layout()
    return fig


def plot_anomaly_boundary_distributions(y, anomaly_types, graphics):
    """Return separate start and end boundary distributions."""
    return [
        plot_anomaly_start_distribution(y, anomaly_types, graphics),
        plot_anomaly_end_distribution(y, anomaly_types, graphics),
    ]


def plot_temporal_label_diagnostics(y, anomaly_types, graphics):
    """Plot temporal-label diagnostics as separate, independently usable figures."""
    figures = [
        plot_anomaly_duration_distribution(y, anomaly_types, graphics),
        plot_full_partial_distribution(y, anomaly_types, graphics),
        plot_duration_by_anomaly_type(y, anomaly_types, graphics),
    ]
    figures.extend(plot_anomaly_boundary_distributions(y, anomaly_types, graphics))
    return figures


def plot_anomaly_window_examples(X, y, feature_names, kpis, graphics):
    """Show training-only KPI traces and the annotated anomaly interval."""
    values = _validate_kpi_tensor(X, feature_names)
    targets = np.asarray(y)
    partial = np.flatnonzero(
        (targets.sum(axis=1) > 0) & (targets.sum(axis=1) < targets.shape[1])
    )
    sample_index = int(partial[0]) if len(partial) else int(np.argmax(targets.sum(axis=1)))
    start = int(targets[sample_index].argmax())
    end = int(targets.shape[1] - 1 - targets[sample_index, ::-1].argmax())
    selected = _selected_feature_names(feature_names, kpis)
    width, height = graphics.get("figsize", [8, 5])
    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(width * 1.5, height * len(selected) * 0.55),
        squeeze=False,
    )
    timesteps = np.arange(values.shape[1])
    for ax, feature in zip(axes.flat, selected):
        feature_index = feature_names.index(feature)
        ax.plot(
            timesteps,
            values[sample_index, :, feature_index],
            color=graphics.get("palette", ["#004D98"])[0],
        )
        ax.axvspan(
            start,
            end,
            color=graphics.get("anomalous_color", "#A50044"),
            alpha=graphics.get("confidence_alpha", 0.16) * 2,
            label="Annotated anomaly interval",
        )
        ax.set_title(feature)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("KPI value")
    axes.flat[0].legend()
    fig.suptitle(
        f"6.6 Example from a training window (sample {sample_index})",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_temporal_forest_importance(
    X, y, anomaly_types, feature_names, graphics, seed
):
    """Rank KPI inputs for timestep labels with window-level train-only holdout."""
    values = _validate_kpi_tensor(X, feature_names)
    targets = np.asarray(y)
    types = np.asarray(anomaly_types)
    if targets.shape[:1] != values.shape[:1] or targets.ndim != 2:
        raise ValueError("Temporal targets must align with KPI training windows.")
    if len(types) != len(values):
        raise ValueError("anomaly_types must have one value per training window.")

    fit_windows, eval_windows = train_test_split(
        np.arange(len(values)),
        test_size=0.2,
        random_state=seed,
        stratify=types,
    )
    rng = np.random.default_rng(seed)
    fit_values = values[fit_windows].reshape(-1, values.shape[2])
    fit_targets = targets[fit_windows].reshape(-1)
    max_fit_timesteps = min(40000, len(fit_targets))
    fit_rows = rng.choice(len(fit_targets), size=max_fit_timesteps, replace=False)

    eval_values = values[eval_windows].reshape(-1, values.shape[2])
    eval_targets = targets[eval_windows].reshape(-1)
    max_eval_timesteps = min(2500, len(eval_targets))
    eval_rows = rng.choice(len(eval_targets), size=max_eval_timesteps, replace=False)

    imputer = SimpleImputer(strategy="median")
    fit_values = imputer.fit_transform(fit_values[fit_rows])
    eval_values = imputer.transform(eval_values[eval_rows])
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(fit_values, fit_targets[fit_rows])
    impurity = pd.Series(
        model.feature_importances_, index=feature_names
    ).nlargest(15)
    permutation = permutation_importance(
        model,
        eval_values,
        eval_targets[eval_rows],
        n_repeats=3,
        random_state=seed,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    permutation_values = pd.Series(
        permutation.importances_mean, index=feature_names
    ).reindex(impurity.index).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    impurity.sort_values().plot.barh(
        ax=axes[0], color=graphics.get("palette", ["#004D98"])[0]
    )
    axes[0].set_title("Mean decrease in impurity")
    axes[0].set_xlabel("Importance")
    permutation_values.plot.barh(
        ax=axes[1], color=graphics.get("palette", ["#004D98", "#A50044"])[1]
    )
    axes[1].axvline(0, color=graphics.get("edgecolor", "white"), linewidth=0.8)
    axes[1].set_title("Permutation importance (training-window holdout)")
    axes[1].set_xlabel("Balanced-accuracy decrease")
    fig.suptitle("6.7 Timestep-level exploratory feature importance", y=1.02)
    fig.tight_layout()
    return fig


def _validate_multilabel_targets(y, label_names):
    targets = np.asarray(y)
    labels = list(label_names)
    if targets.ndim != 2 or targets.shape[1] != len(labels):
        raise ValueError("The multilabel matrix and label_names must align.")
    return targets, labels


def plot_affected_kpi_frequencies(y, label_names, graphics):
    """Plot affected-KPI label frequencies for the training partition."""
    targets, labels = _validate_multilabel_targets(y, label_names)
    counts = targets.sum(axis=0)
    colors = graphics.get("categorical_palette", graphics.get("palette", []))
    fig, ax = plt.subplots(figsize=(12, 6))
    order = np.argsort(counts)
    ax.barh(
        np.asarray(labels)[order],
        counts[order],
        color=colors[0],
        edgecolor=graphics.get("edgecolor", "white"),
    )
    ax.set_title("7.1 Affected-KPI frequency (training labels)")
    ax.set_xlabel("Training samples")
    fig.tight_layout()
    return fig


def plot_affected_kpi_cooccurrence(y, label_names, graphics):
    """Plot affected-KPI label co-occurrence counts for training samples."""
    targets, labels = _validate_multilabel_targets(y, label_names)
    cooccurrence = targets.T @ targets
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cooccurrence,
        xticklabels=labels,
        yticklabels=labels,
        cmap="rocket",
        annot=len(labels) <= 16,
        fmt="d",
        ax=ax,
    )
    ax.set_title("7.2 Affected-KPI co-occurrence count")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    return fig


def plot_label_cardinality(y, label_names, graphics):
    """Plot the number of affected KPIs per training sample."""
    targets, _ = _validate_multilabel_targets(y, label_names)
    cardinality = targets.sum(axis=1)
    colors = graphics.get("categorical_palette", graphics.get("palette", []))
    fig, ax = plt.subplots(figsize=(8, 5))
    cardinality_counts = pd.Series(cardinality).value_counts().sort_index()
    ax.bar(
        cardinality_counts.index.astype(str),
        cardinality_counts.values,
        color=colors[1 % len(colors)],
    )
    ax.set_title("7.3 Number of affected KPIs per training sample")
    ax.set_xlabel("Affected KPI count")
    ax.set_ylabel("Training samples")
    fig.tight_layout()
    return fig


def plot_multilabel_summary(y, label_names, graphics):
    """Return separate frequency, co-occurrence and cardinality figures."""
    return [
        plot_affected_kpi_frequencies(y, label_names, graphics),
        plot_affected_kpi_cooccurrence(y, label_names, graphics),
        plot_label_cardinality(y, label_names, graphics),
    ]


def plot_affected_kpis_by_type(y, anomaly_types, label_names, graphics):
    """Plot within-type affected-KPI rates using training labels only."""
    targets = np.asarray(y)
    types = np.asarray(anomaly_types)
    labels = list(label_names)
    if targets.ndim != 2 or targets.shape != (len(types), len(labels)):
        raise ValueError("Training labels, anomaly types and label_names must align.")
    table = pd.DataFrame(targets, columns=labels).assign(anomaly_type=types)
    rates = table.groupby("anomaly_type", sort=False)[labels].mean()
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        rates,
        cmap="mako",
        annot=True,
        fmt=".0%",
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("7.4 Affected-KPI rate within each training anomaly type")
    ax.set_xlabel("Affected KPI")
    ax.set_ylabel("Anomaly type")
    fig.tight_layout()
    return fig


def plot_multilabel_forest_importance(
    X, y, feature_names, graphics, seed
):
    """Rank input-window summaries for the multilabel training targets."""
    features = _window_feature_frame(X, feature_names)
    features = features.loc[:, features.notna().any(axis=0)]
    labels = np.asarray(y)
    if labels.ndim != 2:
        raise ValueError("A two-dimensional multilabel target is required.")
    imputer = SimpleImputer(strategy="median")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(imputer.fit_transform(features), labels)
    importance = pd.Series(
        model.feature_importances_, index=features.columns
    ).nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    importance.plot.barh(
        ax=ax, color=graphics.get("palette", ["#004D98"])[0]
    )
    ax.set_title("7.5 Exploratory multi-output forest feature importance")
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    return fig

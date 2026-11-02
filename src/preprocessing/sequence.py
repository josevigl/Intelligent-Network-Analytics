"""Train-only imputation, scaling, and one-hot encoding for KPI sequences."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.preprocessing.data import CATEGORICAL_KPIS


class SequencePreprocessor(TransformerMixin, BaseEstimator):
    """Fit KPI transforms on training windows and reuse them unchanged."""

    def __init__(
        self,
        feature_channels: tuple[str, ...] | list[str],
        categorical_missing_value: str = "__MISSING__",
    ) -> None:
        self.feature_channels = tuple(feature_channels)
        self.categorical_missing_value = categorical_missing_value
        if not self.feature_channels:
            raise ValueError("At least one input KPI channel is required.")
        if len(set(self.feature_channels)) != len(self.feature_channels):
            raise ValueError("Input KPI channels must be unique.")

        self.numeric_channels_ = tuple(
            channel
            for channel in self.feature_channels
            if channel not in CATEGORICAL_KPIS
        )
        self.categorical_channels_ = tuple(
            channel
            for channel in self.feature_channels
            if channel in CATEGORICAL_KPIS
        )
        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if self.numeric_channels_:
            numeric_pipeline = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            keep_empty_features=True,
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(
                ("numeric", numeric_pipeline, list(self.numeric_channels_))
            )
        if self.categorical_channels_:
            categorical_pipeline = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value=self.categorical_missing_value,
                            keep_empty_features=True,
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            )
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    list(self.categorical_channels_),
                )
            )

        self.transformer_ = ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=0,
            verbose_feature_names_out=True,
        )

    def _to_frame(self, X: Any) -> pd.DataFrame:
        windows = np.asarray(X, dtype=object)
        if windows.ndim != 3:
            raise ValueError("X must have shape (samples, timesteps, channels).")
        if windows.shape[2] != len(self.feature_channels):
            raise ValueError(
                f"Expected {len(self.feature_channels)} channels in order "
                f"{self.feature_channels}; got {windows.shape[2]}."
            )
        if windows.shape[1] == 0:
            raise ValueError("KPI windows must contain at least one timestep.")

        flattened = windows.reshape(-1, windows.shape[2])
        columns: dict[str, np.ndarray] = {}
        for index, channel in enumerate(self.feature_channels):
            values = flattened[:, index]
            if channel in self.categorical_channels_:
                columns[channel] = np.asarray(
                    [
                        self.categorical_missing_value
                        if self._is_missing(value)
                        else str(value)
                        for value in values
                    ],
                    dtype=object,
                )
            else:
                numeric = pd.to_numeric(
                    pd.Series(values, dtype="object"), errors="coerce"
                ).to_numpy(dtype=np.float64, copy=True)
                numeric[~np.isfinite(numeric)] = np.nan
                columns[channel] = numeric
        return pd.DataFrame(columns, columns=self.feature_channels)

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        return bool(missing) if np.isscalar(missing) else False

    def fit(self, X: Any) -> "SequencePreprocessor":
        windows = np.asarray(X, dtype=object)
        frame = self._to_frame(windows)
        self.sequence_length_ = windows.shape[1]
        self.n_input_channels_ = windows.shape[2]
        self.transformer_.fit(frame)
        self.feature_names_out_ = tuple(
            str(name) for name in self.transformer_.get_feature_names_out()
        )
        self.n_output_channels_ = len(self.feature_names_out_)
        self.is_fitted_ = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        windows = np.asarray(X, dtype=object)
        if windows.ndim != 3:
            raise ValueError("X must have shape (samples, timesteps, channels).")
        if windows.shape[1] != self.sequence_length_:
            raise ValueError(
                f"Expected sequences of length {self.sequence_length_}; "
                f"got {windows.shape[1]}."
            )
        frame = self._to_frame(windows)
        transformed = self.transformer_.transform(frame)
        return np.asarray(
            transformed.reshape(
                len(windows), self.sequence_length_, self.n_output_channels_
            ),
            dtype=np.float32,
        )

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        return np.asarray(self.feature_names_out_, dtype=str)

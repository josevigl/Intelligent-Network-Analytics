"""Estimators that handle labels with a single class in a training fold."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_is_fitted


class SafeBinaryLogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary logistic regression with a constant prediction for one-class folds."""

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | dict[int, float] | None = None,
        max_iter: int = 2000,
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SafeBinaryLogisticRegression":
        features = check_array(X)
        target = np.asarray(y).reshape(-1)
        if len(target) != len(features):
            raise ValueError("X and y have inconsistent sample counts.")
        classes = np.unique(target)
        if not np.isin(classes, (0, 1)).all():
            raise ValueError("SafeBinaryLogisticRegression expects binary targets.")

        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = features.shape[1]
        if len(classes) == 1:
            self.constant_class_ = int(classes[0])
            self.estimator_ = None
            return self

        self.constant_class_ = None
        self.estimator_ = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        ).fit(features, target)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "classes_")
        features = check_array(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features; got {features.shape[1]}."
            )
        if self.constant_class_ is not None:
            probabilities = np.zeros((len(features), 2), dtype=np.float64)
            probabilities[:, self.constant_class_] = 1.0
            return probabilities
        return self.estimator_.predict_proba(features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

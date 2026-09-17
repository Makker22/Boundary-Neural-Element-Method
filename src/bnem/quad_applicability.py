

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

def _as_feature_matrix(values: np.ndarray, feature_dim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("features must be a non-empty matrix")
    if feature_dim is not None and array.shape[1] != feature_dim:
        raise ValueError("features have incompatible dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError("features contain non-finite values")
    return array

def _conformal_upper(values: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        raise ValueError("cannot calibrate an empty error bin")
    rank = min(ordered.size, int(np.ceil((ordered.size + 1) * coverage))) - 1
    return float(ordered[max(rank, 0)])

@dataclass(frozen=True)
class QuadApplicabilityAssessment:
    distances: np.ndarray
    applicable: np.ndarray
    operator_error_upper_bounds: np.ndarray
    bin_indices: np.ndarray

    @property
    def accepted(self) -> bool:
        return bool(np.all(self.applicable))

    @property
    def rejected_indices(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.flatnonzero(~self.applicable))

@dataclass(frozen=True)
class FixedQuadApplicabilityEnvelope:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    normalized_training_features: np.ndarray
    maximum_distance: float
    distance_bin_edges: np.ndarray
    operator_error_upper_bounds: np.ndarray
    coverage: float

    @classmethod
    def fit(
        cls,
        training_features: np.ndarray,
        calibration_features: np.ndarray,
        calibration_operator_errors: np.ndarray,
        *,
        coverage: float = 0.99,
        bins: int = 4,
        distance_margin: float = 1.10,
    ) -> "FixedQuadApplicabilityEnvelope":
        train = _as_feature_matrix(training_features)
        calibration = _as_feature_matrix(calibration_features, train.shape[1])
        errors = np.asarray(calibration_operator_errors, dtype=np.float64)
        if errors.shape != (len(calibration),) or np.any(errors < 0.0) or not np.all(
            np.isfinite(errors)
        ):
            raise ValueError("calibration_operator_errors must be one finite non-negative value per row")
        if not (0.5 < coverage < 1.0):
            raise ValueError("coverage must lie in (0.5,1)")
        if bins < 1 or bins > len(calibration):
            raise ValueError("bins is outside the calibration sample count")
        if distance_margin < 1.0:
            raise ValueError("distance_margin must be at least one")
        mean = np.mean(train, axis=0)
        scale = np.std(train, axis=0)
        scale[scale < 1.0e-6] = 1.0
        normalized_train = (train - mean) / scale
        normalized_calibration = (calibration - mean) / scale
        distances, _ = cKDTree(normalized_train).query(normalized_calibration, k=1)
        quantiles = np.linspace(100.0 / bins, 100.0, bins)
        edges = np.asarray([np.percentile(distances, q) for q in quantiles], dtype=np.float64)
        edges[-1] = max(edges[-1] * distance_margin, 1.0e-12)
        bin_indices = np.minimum(np.searchsorted(edges, distances, side="left"), bins - 1)
        bounds = np.empty(bins, dtype=np.float64)
        previous = 0.0
        for index in range(bins):
            selected = errors[bin_indices == index]
            if selected.size == 0:
                selected = errors[distances <= edges[index]]
            bound = _conformal_upper(selected, coverage)
            previous = max(previous, bound)
            bounds[index] = previous
        return cls(
            feature_mean=mean,
            feature_scale=scale,
            normalized_training_features=normalized_train,
            maximum_distance=float(edges[-1]),
            distance_bin_edges=edges,
            operator_error_upper_bounds=bounds,
            coverage=float(coverage),
        )

    def assess(self, features: np.ndarray) -> QuadApplicabilityAssessment:
        values = _as_feature_matrix(features, self.feature_mean.size)
        normalized = (values - self.feature_mean) / self.feature_scale
        distances, _ = cKDTree(self.normalized_training_features).query(normalized, k=1)
        bins = np.minimum(
            np.searchsorted(self.distance_bin_edges, distances, side="left"),
            len(self.operator_error_upper_bounds) - 1,
        )
        applicable = distances <= self.maximum_distance
        bounds = self.operator_error_upper_bounds[bins].copy()
        bounds[~applicable] = np.inf
        return QuadApplicabilityAssessment(
            distances=np.asarray(distances, dtype=np.float64),
            applicable=np.asarray(applicable, dtype=bool),
            operator_error_upper_bounds=bounds,
            bin_indices=np.asarray(bins, dtype=np.int64),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "bnem.fixed_quad_applicability.v1",
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "normalized_training_features": self.normalized_training_features.tolist(),
            "maximum_distance": self.maximum_distance,
            "distance_bin_edges": self.distance_bin_edges.tolist(),
            "operator_error_upper_bounds": self.operator_error_upper_bounds.tolist(),
            "coverage": self.coverage,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FixedQuadApplicabilityEnvelope":
        if payload.get("schema") != "bnem.fixed_quad_applicability.v1":
            raise ValueError("unsupported fixed-quad applicability schema")
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
            normalized_training_features=np.asarray(
                payload["normalized_training_features"], dtype=np.float64
            ),
            maximum_distance=float(payload["maximum_distance"]),
            distance_bin_edges=np.asarray(payload["distance_bin_edges"], dtype=np.float64),
            operator_error_upper_bounds=np.asarray(
                payload["operator_error_upper_bounds"], dtype=np.float64
            ),
            coverage=float(payload["coverage"]),
        )

__all__ = [
    "FixedQuadApplicabilityEnvelope",
    "QuadApplicabilityAssessment",
]

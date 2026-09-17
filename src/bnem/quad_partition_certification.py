

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.bnem.quad_mesh_quality import QuadMeshQualitySummary, QuadQualityGate

CERTIFIED_QUAD_SHAPE_GATE = QuadQualityGate(
    max_edge_aspect=2.0,
    min_angle_deg=45.0,
    max_angle_deg=135.0,
    max_edge_ratio_std=0.30,
    min_scaled_jacobian=0.50,
)

@dataclass(frozen=True)
class PartitionCertificationPolicy:


    target_relative_error: float = 0.01
    max_geometry_indicator: float = 0.003
    max_energy_weighted_operator_indicator: float = 0.005
    max_cell_operator_error_bound: float = 0.02
    hard_reject_cell_operator_error_bound: float = 0.05
    max_port_indicator: float = 0.005
    max_total_indicator: float = 0.008
    port_saturation_factor: float = 1.0
    max_curvature_size_product: float = 0.25
    max_load_size_product: float = 0.25
    max_neighbor_size_ratio: float = 2.0
    min_speedup: float = 50.0

    def __post_init__(self) -> None:
        positive = {
            "target_relative_error": self.target_relative_error,
            "max_geometry_indicator": self.max_geometry_indicator,
            "max_energy_weighted_operator_indicator": (
                self.max_energy_weighted_operator_indicator
            ),
            "max_cell_operator_error_bound": self.max_cell_operator_error_bound,
            "hard_reject_cell_operator_error_bound": (
                self.hard_reject_cell_operator_error_bound
            ),
            "max_port_indicator": self.max_port_indicator,
            "max_total_indicator": self.max_total_indicator,
            "port_saturation_factor": self.port_saturation_factor,
            "max_curvature_size_product": self.max_curvature_size_product,
            "max_load_size_product": self.max_load_size_product,
            "max_neighbor_size_ratio": self.max_neighbor_size_ratio,
            "min_speedup": self.min_speedup,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_cell_operator_error_bound > self.hard_reject_cell_operator_error_bound:
            raise ValueError(
                "max_cell_operator_error_bound cannot exceed "
                "hard_reject_cell_operator_error_bound"
            )
        if self.max_total_indicator > self.target_relative_error:
            raise ValueError("max_total_indicator cannot exceed target_relative_error")

@dataclass(frozen=True)
class PartitionGeometryMetrics:


    boundary_approximation_relative: float = 0.0
    max_curvature_size_product: float = 0.0
    max_load_size_product: float = 0.0
    max_neighbor_size_ratio: float = 1.0
    topology_aligned: bool = True
    load_breaks_aligned: bool = True
    material_interfaces_aligned: bool = True

    def __post_init__(self) -> None:
        values = {
            "boundary_approximation_relative": self.boundary_approximation_relative,
            "max_curvature_size_product": self.max_curvature_size_product,
            "max_load_size_product": self.max_load_size_product,
            "max_neighbor_size_ratio": self.max_neighbor_size_ratio,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_neighbor_size_ratio < 1.0:
            raise ValueError("max_neighbor_size_ratio must be at least one")

@dataclass(frozen=True)
class HierarchicalPortIndicator:


    value: float
    residual_energy: float
    reference_energy: float
    mode_contributions: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": float(self.value),
            "residual_energy": float(self.residual_energy),
            "reference_energy": float(self.reference_energy),
            "mode_contributions": [float(value) for value in self.mode_contributions],
        }

def hierarchical_port_indicator(
    residual: np.ndarray | Sequence[float],
    inverse_metric: np.ndarray | Sequence[float],
    *,
    reference_energy: float,
) -> HierarchicalPortIndicator:







    vector = np.asarray(residual, dtype=np.float64)
    metric = np.asarray(inverse_metric, dtype=np.float64)
    energy = float(reference_energy)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("residual must be a non-empty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("residual contains non-finite values")
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("reference_energy must be finite and positive")

    if metric.ndim == 1:
        if metric.shape != vector.shape:
            raise ValueError("diagonal inverse_metric must match residual shape")
        if not np.all(np.isfinite(metric)) or np.any(metric < 0.0):
            raise ValueError("diagonal inverse_metric must be finite and non-negative")
        contributions = metric * vector * vector
        residual_energy = float(np.sum(contributions))
    elif metric.ndim == 2:
        if metric.shape != (vector.size, vector.size):
            raise ValueError("matrix inverse_metric must be square and match residual")
        if not np.all(np.isfinite(metric)):
            raise ValueError("matrix inverse_metric contains non-finite values")
        symmetric = 0.5 * (metric + metric.T)
        scale = max(float(np.linalg.norm(symmetric, ord=2)), 1.0)
        if np.linalg.norm(metric - metric.T, ord=np.inf) > 1.0e-10 * scale:
            raise ValueError("matrix inverse_metric must be symmetric")
        minimum = float(np.linalg.eigvalsh(symmetric)[0])
        if minimum < -1.0e-10 * scale:
            raise ValueError("matrix inverse_metric must be positive semidefinite")
        projected = symmetric @ vector
        contributions = vector * projected
        residual_energy = float(vector @ projected)
    else:
        raise ValueError("inverse_metric must be a vector or matrix")

    residual_energy = max(residual_energy, 0.0)
    return HierarchicalPortIndicator(
        value=float(np.sqrt(residual_energy / energy)),
        residual_energy=residual_energy,
        reference_energy=energy,
        mode_contributions=tuple(float(value) for value in contributions),
    )

def energy_weighted_operator_indicator(
    cell_error_bounds: np.ndarray | Sequence[float],
    cell_energy_weights: np.ndarray | Sequence[float],
) -> float:


    bounds = np.asarray(cell_error_bounds, dtype=np.float64)
    weights = np.asarray(cell_energy_weights, dtype=np.float64)
    if bounds.ndim != 1 or bounds.size == 0 or weights.shape != bounds.shape:
        raise ValueError("cell_error_bounds and cell_energy_weights must be equal non-empty vectors")
    if not np.all(np.isfinite(bounds)) or np.any(bounds < 0.0):
        raise ValueError("cell_error_bounds must be finite and non-negative")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("cell_energy_weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("cell_energy_weights must have positive sum")
    return float(np.sqrt(np.dot(weights, bounds * bounds) / total))

@dataclass(frozen=True)
class QuadPartitionCertificate:


    passed: bool
    reasons: tuple[str, ...]
    gates: dict[str, bool]
    indicators: dict[str, float]
    rejected_cell_indices: tuple[int, ...]
    quality: QuadMeshQualitySummary
    geometry: PartitionGeometryMetrics
    hierarchical_port: HierarchicalPortIndicator
    policy: PartitionCertificationPolicy
    online_seconds: float
    fem_reference_seconds: float
    speedup: float

    def to_dict(self, *, include_quality_cells: bool = True) -> dict[str, Any]:
        return {
            "schema": "bnem.quad_partition_certificate.v1",
            "passed": self.passed,
            "reasons": list(self.reasons),
            "gates": dict(self.gates),
            "indicators": {key: float(value) for key, value in self.indicators.items()},
            "rejected_cell_indices": list(self.rejected_cell_indices),
            "quality": self.quality.to_dict(include_cells=include_quality_cells),
            "geometry": {
                key: value
                for key, value in self.geometry.__dict__.items()
            },
            "hierarchical_port": self.hierarchical_port.to_dict(),
            "policy": {key: value for key, value in self.policy.__dict__.items()},
            "timing": {
                "online_seconds": float(self.online_seconds),
                "fem_reference_seconds": float(self.fem_reference_seconds),
                "speedup": float(self.speedup),
            },
        }

def certify_quad_partition(
    *,
    quality: QuadMeshQualitySummary,
    geometry: PartitionGeometryMetrics,
    cell_operator_error_bounds: np.ndarray | Sequence[float],
    cell_energy_weights: np.ndarray | Sequence[float],
    model_applicability: np.ndarray | Sequence[bool],
    hierarchical_port: HierarchicalPortIndicator,
    online_seconds: float,
    fem_reference_seconds: float,
    policy: PartitionCertificationPolicy = PartitionCertificationPolicy(),
) -> QuadPartitionCertificate:


    bounds = np.asarray(cell_operator_error_bounds, dtype=np.float64)
    weights = np.asarray(cell_energy_weights, dtype=np.float64)
    applicable = np.asarray(model_applicability, dtype=bool)
    expected = quality.cell_count
    if bounds.shape != (expected,) or weights.shape != (expected,) or applicable.shape != (expected,):
        raise ValueError(
            "operator bounds, energy weights, and applicability must contain one value per cell"
        )
    operator_indicator = energy_weighted_operator_indicator(bounds, weights)
    online = float(online_seconds)
    fem = float(fem_reference_seconds)
    if not np.isfinite(online) or online <= 0.0:
        raise ValueError("online_seconds must be finite and positive")
    if not np.isfinite(fem) or fem <= 0.0:
        raise ValueError("fem_reference_seconds must be finite and positive")
    speedup = fem / online

    geometry_indicator = float(geometry.boundary_approximation_relative)
    effective_port_indicator = float(
        policy.port_saturation_factor * hierarchical_port.value
    )
    total_indicator = float(
        np.sqrt(
            geometry_indicator * geometry_indicator
            + operator_indicator * operator_indicator
            + effective_port_indicator**2
        )
    )
    geometry_gate = bool(
        geometry.topology_aligned
        and geometry.load_breaks_aligned
        and geometry.material_interfaces_aligned
        and geometry.boundary_approximation_relative <= policy.max_geometry_indicator
        and geometry.max_curvature_size_product <= policy.max_curvature_size_product
        and geometry.max_load_size_product <= policy.max_load_size_product
        and geometry.max_neighbor_size_ratio <= policy.max_neighbor_size_ratio
    )
    hard_operator_reject = bounds > policy.hard_reject_cell_operator_error_bound
    local_operator_gate = bool(
        np.all(bounds <= policy.max_cell_operator_error_bound)
        and not np.any(hard_operator_reject)
    )
    applicability_gate = bool(np.all(applicable))
    gates = {
        "shape": bool(quality.accepted),
        "geometry": geometry_gate,
        "model_applicability": applicability_gate,
        "local_operator": local_operator_gate,
        "energy_weighted_operator": bool(
            operator_indicator <= policy.max_energy_weighted_operator_indicator
        ),
        "hierarchical_port_resolution": bool(
            effective_port_indicator <= policy.max_port_indicator
        ),
        "total_error_budget": bool(total_indicator <= policy.max_total_indicator),
        "speed": bool(speedup >= policy.min_speedup),
    }
    reason_names = {
        "shape": "shape_quality_failed",
        "geometry": "geometry_resolution_failed",
        "model_applicability": "model_applicability_failed",
        "local_operator": "local_operator_bound_failed",
        "energy_weighted_operator": "global_operator_budget_failed",
        "hierarchical_port_resolution": "port_resolution_failed",
        "total_error_budget": "total_error_budget_failed",
        "speed": "speed_budget_failed",
    }
    reasons = tuple(reason_names[name] for name, passed in gates.items() if not passed)
    rejected = set(int(cell.cell_index) for cell in quality.rejected_cells)
    rejected.update(int(index) for index in np.flatnonzero(~applicable))
    rejected.update(
        int(index)
        for index in np.flatnonzero(bounds > policy.max_cell_operator_error_bound)
    )
    return QuadPartitionCertificate(
        passed=all(gates.values()),
        reasons=reasons,
        gates=gates,
        indicators={
            "geometry": geometry_indicator,
            "energy_weighted_operator": operator_indicator,
            "max_cell_operator_error_bound": float(np.max(bounds, initial=0.0)),
            "hierarchical_port": float(hierarchical_port.value),
            "effective_hierarchical_port": effective_port_indicator,
            "total": total_indicator,
        },
        rejected_cell_indices=tuple(sorted(rejected)),
        quality=quality,
        geometry=geometry,
        hierarchical_port=hierarchical_port,
        policy=policy,
        online_seconds=online,
        fem_reference_seconds=fem,
        speedup=float(speedup),
    )

__all__ = [
    "CERTIFIED_QUAD_SHAPE_GATE",
    "HierarchicalPortIndicator",
    "PartitionCertificationPolicy",
    "PartitionGeometryMetrics",
    "QuadPartitionCertificate",
    "certify_quad_partition",
    "energy_weighted_operator_indicator",
    "hierarchical_port_indicator",
]

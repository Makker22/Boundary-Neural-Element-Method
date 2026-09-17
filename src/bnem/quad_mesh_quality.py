

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Literal, Sequence

import numpy as np

GateMode = Literal["strict", "soft"]

_NATURAL_CORNERS: tuple[tuple[float, float], ...] = (
    (-1.0, -1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
)
_GAUSS_COORDINATE = 1.0 / sqrt(3.0)
_GAUSS_POINTS: tuple[tuple[float, float], ...] = (
    (-_GAUSS_COORDINATE, -_GAUSS_COORDINATE),
    (_GAUSS_COORDINATE, -_GAUSS_COORDINATE),
    (_GAUSS_COORDINATE, _GAUSS_COORDINATE),
    (-_GAUSS_COORDINATE, _GAUSS_COORDINATE),
)

_SOFT_SHAPE_CODES = frozenset(
    {
        "edge_aspect_exceeds_limit",
        "edge_ratio_std_exceeds_limit",
        "angle_below_minimum",
        "angle_above_maximum",
        "scaled_jacobian_below_minimum",
    }
)

@dataclass(frozen=True)
class QuadQualityGate:









    max_edge_aspect: float | None = 1.6
    min_angle_deg: float | None = 50.0
    max_angle_deg: float | None = 130.0
    max_edge_ratio_std: float | None = None
    min_scaled_jacobian: float | None = None
    min_det_jacobian: float = 0.0
    require_simple: bool = True
    require_convex: bool = True
    require_counter_clockwise: bool = True
    check_corner_jacobians: bool = True
    check_gauss_jacobians: bool = True

    def __post_init__(self) -> None:
        if self.max_edge_aspect is not None and self.max_edge_aspect < 1.0:
            raise ValueError("max_edge_aspect must be at least 1")
        if self.min_angle_deg is not None and not 0.0 <= self.min_angle_deg <= 360.0:
            raise ValueError("min_angle_deg must lie in [0, 360]")
        if self.max_angle_deg is not None and not 0.0 <= self.max_angle_deg <= 360.0:
            raise ValueError("max_angle_deg must lie in [0, 360]")
        if (
            self.min_angle_deg is not None
            and self.max_angle_deg is not None
            and self.min_angle_deg > self.max_angle_deg
        ):
            raise ValueError("min_angle_deg cannot exceed max_angle_deg")
        if self.max_edge_ratio_std is not None and self.max_edge_ratio_std < 0.0:
            raise ValueError("max_edge_ratio_std must be non-negative")
        if self.min_scaled_jacobian is not None and not -1.0 <= self.min_scaled_jacobian <= 1.0:
            raise ValueError("min_scaled_jacobian must lie in [-1, 1]")
        if not np.isfinite(self.min_det_jacobian):
            raise ValueError("min_det_jacobian must be finite")

DEFAULT_PRODUCTION_GATE = QuadQualityGate()

@dataclass(frozen=True)
class QuadQualityMetrics:


    signed_area: float
    interior_angles_deg: tuple[float, float, float, float]
    edge_lengths: tuple[float, float, float, float]
    edge_ratios: tuple[float, float, float, float]
    edge_aspect: float
    edge_ratio_std: float
    is_simple: bool
    is_convex: bool
    corner_det_jacobians: tuple[float, float, float, float]
    gauss_det_jacobians: tuple[float, float, float, float]
    corner_scaled_jacobians: tuple[float, float, float, float]
    gauss_scaled_jacobians: tuple[float, float, float, float]

    @property
    def min_angle_deg(self) -> float:
        return _finite_extreme(self.interior_angles_deg, minimum=True)

    @property
    def max_angle_deg(self) -> float:
        return _finite_extreme(self.interior_angles_deg, minimum=False)

    @property
    def min_corner_det_jacobian(self) -> float:
        return _finite_extreme(self.corner_det_jacobians, minimum=True)

    @property
    def min_gauss_det_jacobian(self) -> float:
        return _finite_extreme(self.gauss_det_jacobians, minimum=True)

    @property
    def min_scaled_jacobian(self) -> float:
        return _finite_extreme(
            self.corner_scaled_jacobians + self.gauss_scaled_jacobians,
            minimum=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "signed_area": _json_float(self.signed_area),
            "interior_angles_deg": [_json_float(value) for value in self.interior_angles_deg],
            "min_angle_deg": _json_float(self.min_angle_deg),
            "max_angle_deg": _json_float(self.max_angle_deg),
            "edge_lengths": [_json_float(value) for value in self.edge_lengths],
            "edge_ratios": [_json_float(value) for value in self.edge_ratios],
            "edge_aspect": _json_float(self.edge_aspect),
            "edge_ratio_std": _json_float(self.edge_ratio_std),
            "is_simple": self.is_simple,
            "is_convex": self.is_convex,
            "corner_det_jacobians": [_json_float(value) for value in self.corner_det_jacobians],
            "gauss_det_jacobians": [_json_float(value) for value in self.gauss_det_jacobians],
            "corner_scaled_jacobians": [
                _json_float(value) for value in self.corner_scaled_jacobians
            ],
            "gauss_scaled_jacobians": [
                _json_float(value) for value in self.gauss_scaled_jacobians
            ],
            "min_corner_det_jacobian": _json_float(self.min_corner_det_jacobian),
            "min_gauss_det_jacobian": _json_float(self.min_gauss_det_jacobian),
            "min_scaled_jacobian": _json_float(self.min_scaled_jacobian),
        }

@dataclass(frozen=True)
class QuadQualityIssue:


    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}

@dataclass(frozen=True)
class QuadQualityAssessment:


    metrics: QuadQualityMetrics
    mode: GateMode
    failures: tuple[QuadQualityIssue, ...]
    warnings: tuple[QuadQualityIssue, ...]

    @property
    def accepted(self) -> bool:
        return not self.failures

    @property
    def production_compliant(self) -> bool:
        return not self.failures and not self.warnings

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.failures)

    @property
    def warning_reasons(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "production_compliant": self.production_compliant,
            "mode": self.mode,
            "failure_reasons": list(self.failure_reasons),
            "warning_reasons": list(self.warning_reasons),
            "failures": [issue.to_dict() for issue in self.failures],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "metrics": self.metrics.to_dict(),
        }

@dataclass(frozen=True)
class QuadCellQualityAssessment:


    cell_index: int
    vertex_indices: tuple[int, ...]
    metrics: QuadQualityMetrics | None
    mode: GateMode
    failures: tuple[QuadQualityIssue, ...]
    warnings: tuple[QuadQualityIssue, ...] = ()

    @property
    def accepted(self) -> bool:
        return not self.failures

    @property
    def production_compliant(self) -> bool:
        return not self.failures and not self.warnings

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.failures)

    @property
    def warning_reasons(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_index": self.cell_index,
            "vertex_indices": list(self.vertex_indices),
            "accepted": self.accepted,
            "production_compliant": self.production_compliant,
            "mode": self.mode,
            "failure_reasons": list(self.failure_reasons),
            "warning_reasons": list(self.warning_reasons),
            "failures": [issue.to_dict() for issue in self.failures],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
        }

@dataclass(frozen=True)
class QuadMeshQualitySummary:


    mode: GateMode
    cell_count: int
    accepted_cell_count: int
    rejected_cell_count: int
    warning_cell_count: int
    production_compliant_cell_count: int
    failure_reason_counts: dict[str, int]
    warning_reason_counts: dict[str, int]
    metric_summary: dict[str, float | None]
    cells: tuple[QuadCellQualityAssessment, ...]

    @property
    def accepted(self) -> bool:
        return self.rejected_cell_count == 0

    @property
    def production_compliant(self) -> bool:
        return self.production_compliant_cell_count == self.cell_count

    @property
    def rejected_cells(self) -> tuple[QuadCellQualityAssessment, ...]:
        return tuple(cell for cell in self.cells if not cell.accepted)

    @property
    def warning_cells(self) -> tuple[QuadCellQualityAssessment, ...]:
        return tuple(cell for cell in self.cells if cell.warnings)

    @property
    def cell_assessments(self) -> tuple[QuadCellQualityAssessment, ...]:


        return self.cells

    def to_dict(self, *, include_cells: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "accepted": self.accepted,
            "production_compliant": self.production_compliant,
            "mode": self.mode,
            "cell_count": self.cell_count,
            "accepted_cell_count": self.accepted_cell_count,
            "rejected_cell_count": self.rejected_cell_count,
            "warning_cell_count": self.warning_cell_count,
            "production_compliant_cell_count": self.production_compliant_cell_count,
            "failure_reason_counts": dict(self.failure_reason_counts),
            "warning_reason_counts": dict(self.warning_reason_counts),
            "metric_summary": dict(self.metric_summary),
        }
        if include_cells:
            payload["cells"] = [cell.to_dict() for cell in self.cells]
        return payload

class QuadMeshQualityError(ValueError):


    def __init__(self, summary: QuadMeshQualitySummary):
        self.summary = summary
        reasons = ", ".join(
            f"{code}={count}" for code, count in sorted(summary.failure_reason_counts.items())
        )
        super().__init__(
            f"Quadrilateral mesh quality gate rejected "
            f"{summary.rejected_cell_count}/{summary.cell_count} cells: {reasons}"
        )

def _finite_extreme(values: Sequence[float], *, minimum: bool) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite) if minimum else np.max(finite))

def _json_float(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None

def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])

def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return _cross(b - a, c - a)

def _on_segment(a: np.ndarray, b: np.ndarray, point: np.ndarray, tolerance: float) -> bool:
    return bool(
        abs(_orientation(a, b, point)) <= tolerance
        and np.all(point >= np.minimum(a, b) - tolerance)
        and np.all(point <= np.maximum(a, b) + tolerance)
    )

def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tolerance: float,
) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    if ab_c * ab_d < -(tolerance * tolerance) and cd_a * cd_b < -(tolerance * tolerance):
        return True
    return (
        _on_segment(a, b, c, tolerance)
        or _on_segment(a, b, d, tolerance)
        or _on_segment(c, d, a, tolerance)
        or _on_segment(c, d, b, tolerance)
    )

def _q4_jacobian(vertices: np.ndarray, xi: float, eta: float) -> np.ndarray:
    dshape_dxi = 0.25 * np.asarray(
        [-(1.0 - eta), 1.0 - eta, 1.0 + eta, -(1.0 + eta)],
        dtype=np.float64,
    )
    dshape_deta = 0.25 * np.asarray(
        [-(1.0 - xi), -(1.0 + xi), 1.0 + xi, 1.0 - xi],
        dtype=np.float64,
    )
    return np.column_stack((dshape_dxi @ vertices, dshape_deta @ vertices))

def _jacobian_samples(
    vertices: np.ndarray,
    points: Sequence[tuple[float, float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    determinants: list[float] = []
    scaled: list[float] = []
    for xi, eta in points:
        jacobian = _q4_jacobian(vertices, xi, eta)
        determinant = float(np.linalg.det(jacobian))
        denominator = float(np.linalg.norm(jacobian[:, 0]) * np.linalg.norm(jacobian[:, 1]))
        determinants.append(determinant)
        scaled.append(determinant / denominator if denominator > 0.0 else 0.0)
    return tuple(determinants), tuple(scaled)

def quad_quality_metrics(vertices: np.ndarray | Sequence[Sequence[float]]) -> QuadQualityMetrics:


    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (4, 2):
        raise ValueError(f"Expected Q4 vertices with shape [4,2], got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Q4 vertices must contain only finite coordinates")

    shifted = np.roll(points, -1, axis=0)
    edge_vectors = shifted - points
    edge_lengths_array = np.linalg.norm(edge_vectors, axis=1)
    edge_lengths = tuple(float(value) for value in edge_lengths_array)
    mean_edge = float(np.mean(edge_lengths_array))
    minimum_edge = float(np.min(edge_lengths_array))
    maximum_edge = float(np.max(edge_lengths_array))
    edge_aspect = maximum_edge / minimum_edge if minimum_edge > 0.0 else float("inf")
    edge_ratio_std = (
        float(np.std(edge_lengths_array / mean_edge)) if mean_edge > 0.0 else float("inf")
    )
    edge_ratios_array = (
        edge_lengths_array / mean_edge
        if mean_edge > 0.0
        else np.full(4, float("nan"), dtype=np.float64)
    )

    signed_area = 0.5 * float(
        np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1])
    )
    length_scale = max(maximum_edge, float(np.ptp(points, axis=0).max()), 1.0)
    length_tolerance = 64.0 * np.finfo(np.float64).eps * length_scale
    area_tolerance = 64.0 * np.finfo(np.float64).eps * length_scale * length_scale
    has_degenerate_edge = minimum_edge <= length_tolerance

    is_simple = not has_degenerate_edge and not (
        _segments_intersect(points[0], points[1], points[2], points[3], area_tolerance)
        or _segments_intersect(points[1], points[2], points[3], points[0], area_tolerance)
    )
    turns = np.asarray(
        [
            _cross(points[(index + 1) % 4] - points[index], points[(index + 2) % 4] - points[(index + 1) % 4])
            for index in range(4)
        ],
        dtype=np.float64,
    )
    is_convex = bool(
        is_simple
        and (np.all(turns > area_tolerance) or np.all(turns < -area_tolerance))
    )

    orientation_sign = 1.0 if signed_area >= 0.0 else -1.0
    angles: list[float] = []
    for index in range(4):
        incoming = points[index] - points[(index - 1) % 4]
        outgoing = points[(index + 1) % 4] - points[index]
        denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if denominator <= 0.0:
            angles.append(float("nan"))
            continue
        cosine = float(np.clip(np.dot(-incoming, outgoing) / denominator, -1.0, 1.0))
        base_angle = float(np.degrees(np.arccos(cosine)))
        turn = _cross(incoming, outgoing)
        angles.append(base_angle if turn * orientation_sign >= 0.0 else 360.0 - base_angle)

    corner_det, corner_scaled = _jacobian_samples(points, _NATURAL_CORNERS)
    gauss_det, gauss_scaled = _jacobian_samples(points, _GAUSS_POINTS)
    return QuadQualityMetrics(
        signed_area=signed_area,
        interior_angles_deg=tuple(angles),
        edge_lengths=edge_lengths,
        edge_ratios=tuple(float(value) for value in edge_ratios_array),
        edge_aspect=float(edge_aspect),
        edge_ratio_std=float(edge_ratio_std),
        is_simple=is_simple,
        is_convex=is_convex,
        corner_det_jacobians=corner_det,
        gauss_det_jacobians=gauss_det,
        corner_scaled_jacobians=corner_scaled,
        gauss_scaled_jacobians=gauss_scaled,
    )

def _orientation_batch(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    first = b - a
    second = c - a
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]

def _on_segment_batch(
    a: np.ndarray,
    b: np.ndarray,
    point: np.ndarray,
    tolerance: np.ndarray,
) -> np.ndarray:
    return (
        (np.abs(_orientation_batch(a, b, point)) <= tolerance)
        & np.all(point >= np.minimum(a, b) - tolerance[:, None], axis=1)
        & np.all(point <= np.maximum(a, b) + tolerance[:, None], axis=1)
    )

def _segments_intersect_batch(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tolerance: np.ndarray,
) -> np.ndarray:
    ab_c = _orientation_batch(a, b, c)
    ab_d = _orientation_batch(a, b, d)
    cd_a = _orientation_batch(c, d, a)
    cd_b = _orientation_batch(c, d, b)
    strict = (ab_c * ab_d < -(tolerance * tolerance)) & (
        cd_a * cd_b < -(tolerance * tolerance)
    )
    return (
        strict
        | _on_segment_batch(a, b, c, tolerance)
        | _on_segment_batch(a, b, d, tolerance)
        | _on_segment_batch(c, d, a, tolerance)
        | _on_segment_batch(c, d, b, tolerance)
    )

def quad_quality_metrics_batch(
    vertices: np.ndarray | Sequence[Sequence[Sequence[float]]],
) -> tuple[QuadQualityMetrics, ...]:


    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (4, 2):
        raise ValueError(f"Expected Q4 batch with shape [M,4,2], got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Q4 vertices must contain only finite coordinates")
    count = int(points.shape[0])
    if count == 0:
        return ()

    shifted = np.roll(points, -1, axis=1)
    edges = shifted - points
    lengths = np.linalg.norm(edges, axis=2)
    means = np.mean(lengths, axis=1)
    minima = np.min(lengths, axis=1)
    maxima = np.max(lengths, axis=1)
    aspects = np.divide(
        maxima,
        minima,
        out=np.full(count, np.inf, dtype=np.float64),
        where=minima > 0.0,
    )
    ratios = np.divide(
        lengths,
        means[:, None],
        out=np.full_like(lengths, np.nan),
        where=means[:, None] > 0.0,
    )
    ratio_stds = np.where(means > 0.0, np.std(ratios, axis=1), np.inf)
    signed_areas = 0.5 * np.sum(
        points[:, :, 0] * shifted[:, :, 1]
        - shifted[:, :, 0] * points[:, :, 1],
        axis=1,
    )
    point_ranges = np.ptp(points, axis=1)
    length_scales = np.maximum.reduce(
        (maxima, np.max(point_ranges, axis=1), np.ones(count, dtype=np.float64))
    )
    length_tolerances = 64.0 * np.finfo(np.float64).eps * length_scales
    area_tolerances = 64.0 * np.finfo(np.float64).eps * length_scales * length_scales
    degenerate = minima <= length_tolerances
    crossings = _segments_intersect_batch(
        points[:, 0], points[:, 1], points[:, 2], points[:, 3], area_tolerances
    ) | _segments_intersect_batch(
        points[:, 1], points[:, 2], points[:, 3], points[:, 0], area_tolerances
    )
    simple = ~degenerate & ~crossings
    turns = edges[:, :, 0] * np.roll(edges, -1, axis=1)[:, :, 1] - edges[:, :, 1] * np.roll(
        edges, -1, axis=1
    )[:, :, 0]
    convex = simple & (
        np.all(turns > area_tolerances[:, None], axis=1)
        | np.all(turns < -area_tolerances[:, None], axis=1)
    )

    incoming = points - np.roll(points, 1, axis=1)
    outgoing = shifted - points
    denominators = np.linalg.norm(incoming, axis=2) * np.linalg.norm(outgoing, axis=2)
    cosine = np.divide(
        np.sum(-incoming * outgoing, axis=2),
        denominators,
        out=np.zeros_like(denominators),
        where=denominators > 0.0,
    )
    base_angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    angle_turns = incoming[:, :, 0] * outgoing[:, :, 1] - incoming[:, :, 1] * outgoing[:, :, 0]
    orientation_sign = np.where(signed_areas >= 0.0, 1.0, -1.0)
    angles = np.where(
        angle_turns * orientation_sign[:, None] >= 0.0,
        base_angles,
        360.0 - base_angles,
    )
    angles[denominators <= 0.0] = np.nan

    samples = np.asarray(_NATURAL_CORNERS + _GAUSS_POINTS, dtype=np.float64)
    xi = samples[:, 0]
    eta = samples[:, 1]
    dshape_dxi = 0.25 * np.stack(
        (-(1.0 - eta), 1.0 - eta, 1.0 + eta, -(1.0 + eta)), axis=1
    )
    dshape_deta = 0.25 * np.stack(
        (-(1.0 - xi), -(1.0 + xi), 1.0 + xi, 1.0 - xi), axis=1
    )
    jacobian_xi = np.einsum("sj,mjk->msk", dshape_dxi, points)
    jacobian_eta = np.einsum("sj,mjk->msk", dshape_deta, points)
    determinants = (
        jacobian_xi[:, :, 0] * jacobian_eta[:, :, 1]
        - jacobian_xi[:, :, 1] * jacobian_eta[:, :, 0]
    )
    jacobian_denominators = np.linalg.norm(jacobian_xi, axis=2) * np.linalg.norm(
        jacobian_eta, axis=2
    )
    scaled = np.divide(
        determinants,
        jacobian_denominators,
        out=np.zeros_like(determinants),
        where=jacobian_denominators > 0.0,
    )

    return tuple(
        QuadQualityMetrics(
            signed_area=float(signed_areas[index]),
            interior_angles_deg=tuple(float(value) for value in angles[index]),
            edge_lengths=tuple(float(value) for value in lengths[index]),
            edge_ratios=tuple(float(value) for value in ratios[index]),
            edge_aspect=float(aspects[index]),
            edge_ratio_std=float(ratio_stds[index]),
            is_simple=bool(simple[index]),
            is_convex=bool(convex[index]),
            corner_det_jacobians=tuple(float(value) for value in determinants[index, :4]),
            gauss_det_jacobians=tuple(float(value) for value in determinants[index, 4:]),
            corner_scaled_jacobians=tuple(float(value) for value in scaled[index, :4]),
            gauss_scaled_jacobians=tuple(float(value) for value in scaled[index, 4:]),
        )
        for index in range(count)
    )

def _issue(code: str, message: str) -> QuadQualityIssue:
    return QuadQualityIssue(code=code, message=message)

def _quality_issues(metrics: QuadQualityMetrics, gate: QuadQualityGate) -> list[QuadQualityIssue]:
    issues: list[QuadQualityIssue] = []
    edge_lengths = np.asarray(metrics.edge_lengths, dtype=np.float64)
    if np.any(edge_lengths <= 0.0) or not np.all(np.isfinite(edge_lengths)):
        issues.append(_issue("degenerate_edge", "At least one Q4 edge has zero or invalid length"))
    if gate.require_simple and not metrics.is_simple:
        issues.append(_issue("non_simple", "The quadrilateral boundary self-intersects or degenerates"))
    if gate.require_convex and not metrics.is_convex:
        issues.append(_issue("non_convex", "The quadrilateral is not strictly convex"))
    if gate.require_counter_clockwise and not metrics.signed_area > 0.0:
        issues.append(
            _issue(
                "nonpositive_signed_area",
                f"Signed area must be positive, got {metrics.signed_area:.6g}",
            )
        )
    if gate.max_edge_aspect is not None and not metrics.edge_aspect <= gate.max_edge_aspect:
        issues.append(
            _issue(
                "edge_aspect_exceeds_limit",
                f"Edge aspect {metrics.edge_aspect:.6g} exceeds {gate.max_edge_aspect:.6g}",
            )
        )
    if (
        gate.max_edge_ratio_std is not None
        and not metrics.edge_ratio_std <= gate.max_edge_ratio_std
    ):
        issues.append(
            _issue(
                "edge_ratio_std_exceeds_limit",
                f"Edge-ratio standard deviation {metrics.edge_ratio_std:.6g} "
                f"exceeds {gate.max_edge_ratio_std:.6g}",
            )
        )

    angles = np.asarray(metrics.interior_angles_deg, dtype=np.float64)
    if not np.all(np.isfinite(angles)):
        issues.append(_issue("invalid_interior_angle", "At least one interior angle is undefined"))
    else:
        if gate.min_angle_deg is not None and float(np.min(angles)) < gate.min_angle_deg:
            issues.append(
                _issue(
                    "angle_below_minimum",
                    f"Minimum angle {np.min(angles):.6g} is below {gate.min_angle_deg:.6g} degrees",
                )
            )
        if gate.max_angle_deg is not None and float(np.max(angles)) > gate.max_angle_deg:
            issues.append(
                _issue(
                    "angle_above_maximum",
                    f"Maximum angle {np.max(angles):.6g} exceeds {gate.max_angle_deg:.6g} degrees",
                )
            )

    if gate.check_corner_jacobians:
        values = np.asarray(metrics.corner_det_jacobians, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= gate.min_det_jacobian):
            issues.append(
                _issue(
                    "nonpositive_corner_jacobian",
                    f"Corner detJ minimum {np.min(values):.6g} must exceed "
                    f"{gate.min_det_jacobian:.6g}",
                )
            )
    if gate.check_gauss_jacobians:
        values = np.asarray(metrics.gauss_det_jacobians, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= gate.min_det_jacobian):
            issues.append(
                _issue(
                    "nonpositive_gauss_jacobian",
                    f"Gauss-point detJ minimum {np.min(values):.6g} must exceed "
                    f"{gate.min_det_jacobian:.6g}",
                )
            )
    if gate.min_scaled_jacobian is not None:
        values = np.asarray(
            metrics.corner_scaled_jacobians + metrics.gauss_scaled_jacobians,
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values < gate.min_scaled_jacobian):
            issues.append(
                _issue(
                    "scaled_jacobian_below_minimum",
                    f"Scaled Jacobian minimum {np.min(values):.6g} is below "
                    f"{gate.min_scaled_jacobian:.6g}",
                )
            )
    return issues

def assess_quad_quality(
    vertices: np.ndarray | Sequence[Sequence[float]],
    *,
    gate: QuadQualityGate = DEFAULT_PRODUCTION_GATE,
    mode: GateMode = "strict",
) -> QuadQualityAssessment:


    if mode not in {"strict", "soft"}:
        raise ValueError("mode must be either 'strict' or 'soft'")
    metrics = quad_quality_metrics(vertices)
    issues = _quality_issues(metrics, gate)
    if mode == "soft":
        warnings = tuple(issue for issue in issues if issue.code in _SOFT_SHAPE_CODES)
        failures = tuple(issue for issue in issues if issue.code not in _SOFT_SHAPE_CODES)
    else:
        failures = tuple(issues)
        warnings = ()
    return QuadQualityAssessment(
        metrics=metrics,
        mode=mode,
        failures=failures,
        warnings=warnings,
    )

def _assessment_from_metrics(
    metrics: QuadQualityMetrics,
    *,
    gate: QuadQualityGate,
    mode: GateMode,
) -> QuadQualityAssessment:
    issues = _quality_issues(metrics, gate)
    if mode == "soft":
        warnings = tuple(issue for issue in issues if issue.code in _SOFT_SHAPE_CODES)
        failures = tuple(issue for issue in issues if issue.code not in _SOFT_SHAPE_CODES)
    else:
        failures = tuple(issues)
        warnings = ()
    return QuadQualityAssessment(
        metrics=metrics,
        mode=mode,
        failures=failures,
        warnings=warnings,
    )

def _invalid_cell(
    cell_index: int,
    indices: tuple[int, ...],
    mode: GateMode,
    code: str,
    message: str,
) -> QuadCellQualityAssessment:
    return QuadCellQualityAssessment(
        cell_index=cell_index,
        vertex_indices=indices,
        metrics=None,
        mode=mode,
        failures=(_issue(code, message),),
    )

def _percentile(values: Sequence[float], percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, percentile))

def _metric_summary(cells: Sequence[QuadCellQualityAssessment]) -> dict[str, float | None]:
    metrics = [cell.metrics for cell in cells if cell.metrics is not None]
    if not metrics:
        return {
            key: None
            for key in (
                "min_signed_area",
                "max_signed_area",
                "min_angle_deg",
                "max_angle_deg",
                "mean_edge_aspect",
                "p50_edge_aspect",
                "p90_edge_aspect",
                "p95_edge_aspect",
                "max_edge_aspect",
                "max_edge_ratio_std",
                "min_corner_det_jacobian",
                "min_gauss_det_jacobian",
                "min_scaled_jacobian",
            )
        }
    areas = [metric.signed_area for metric in metrics]
    aspects = [metric.edge_aspect for metric in metrics]
    edge_stds = [metric.edge_ratio_std for metric in metrics]
    finite_aspects = np.asarray(aspects, dtype=np.float64)
    finite_aspects = finite_aspects[np.isfinite(finite_aspects)]
    return {
        "min_signed_area": _percentile(areas, 0.0),
        "max_signed_area": _percentile(areas, 100.0),
        "min_angle_deg": _percentile([metric.min_angle_deg for metric in metrics], 0.0),
        "max_angle_deg": _percentile([metric.max_angle_deg for metric in metrics], 100.0),
        "mean_edge_aspect": (
            float(np.mean(finite_aspects)) if finite_aspects.size else None
        ),
        "p50_edge_aspect": _percentile(aspects, 50.0),
        "p90_edge_aspect": _percentile(aspects, 90.0),
        "p95_edge_aspect": _percentile(aspects, 95.0),
        "max_edge_aspect": _percentile(aspects, 100.0),
        "max_edge_ratio_std": _percentile(edge_stds, 100.0),
        "min_corner_det_jacobian": _percentile(
            [metric.min_corner_det_jacobian for metric in metrics], 0.0
        ),
        "min_gauss_det_jacobian": _percentile(
            [metric.min_gauss_det_jacobian for metric in metrics], 0.0
        ),
        "min_scaled_jacobian": _percentile(
            [metric.min_scaled_jacobian for metric in metrics], 0.0
        ),
    }

def _build_mesh_summary(
    results: Sequence[QuadCellQualityAssessment],
    *,
    mode: GateMode,
) -> QuadMeshQualitySummary:
    failure_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    for result in results:
        failure_counts.update(result.failure_reasons)
        warning_counts.update(result.warning_reasons)
    rejected_count = sum(not result.accepted for result in results)
    warning_count = sum(bool(result.warnings) for result in results)
    compliant_count = sum(result.production_compliant for result in results)
    return QuadMeshQualitySummary(
        mode=mode,
        cell_count=len(results),
        accepted_cell_count=len(results) - rejected_count,
        rejected_cell_count=rejected_count,
        warning_cell_count=warning_count,
        production_compliant_cell_count=compliant_count,
        failure_reason_counts=dict(sorted(failure_counts.items())),
        warning_reason_counts=dict(sorted(warning_counts.items())),
        metric_summary=_metric_summary(results),
        cells=tuple(results),
    )

def assess_quad_mesh_quality(
    vertices: np.ndarray | Sequence[Sequence[float]],
    cells: Iterable[Iterable[int]],
    *,
    gate: QuadQualityGate = DEFAULT_PRODUCTION_GATE,
    mode: GateMode = "strict",
) -> QuadMeshQualitySummary:


    if mode not in {"strict", "soft"}:
        raise ValueError("mode must be either 'strict' or 'soft'")
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected mesh vertices with shape [N,2], got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Mesh vertices must contain only finite coordinates")

    raw_cells = list(cells)
    results: list[QuadCellQualityAssessment] = []

    dense_connectivity: np.ndarray | None = None
    try:
        candidate = np.asarray(raw_cells)
        numeric_candidate = candidate.astype(np.float64)
        if (
            candidate.ndim == 2
            and candidate.shape[1] == 4
            and np.all(np.isfinite(numeric_candidate))
            and np.all(numeric_candidate == np.rint(numeric_candidate))
        ):
            indices_candidate = numeric_candidate.astype(np.int64)
            if (
                np.all(indices_candidate >= 0)
                and np.all(indices_candidate < len(points))
                and np.all(
                    np.asarray(
                        [np.unique(row).size == 4 for row in indices_candidate],
                        dtype=bool,
                    )
                )
            ):
                dense_connectivity = indices_candidate
    except (TypeError, ValueError):
        dense_connectivity = None

    if dense_connectivity is not None:
        batched_metrics = quad_quality_metrics_batch(points[dense_connectivity])
        for cell_index, (indices_array, metrics) in enumerate(
            zip(dense_connectivity, batched_metrics)
        ):
            assessment = _assessment_from_metrics(metrics, gate=gate, mode=mode)
            results.append(
                QuadCellQualityAssessment(
                    cell_index=cell_index,
                    vertex_indices=tuple(int(value) for value in indices_array),
                    metrics=metrics,
                    mode=mode,
                    failures=assessment.failures,
                    warnings=assessment.warnings,
                )
            )
    else:
        for cell_index, cell in enumerate(raw_cells):
            try:
                raw = np.asarray(list(cell))
            except TypeError:
                results.append(
                    _invalid_cell(
                        cell_index,
                        (),
                        mode,
                        "invalid_cell_connectivity",
                        "Q4 connectivity must be an iterable of four integer indices",
                    )
                )
                continue
            if raw.ndim != 1 or raw.size != 4:
                preview: list[int] = []
                for value in raw.reshape(-1):
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(numeric_value) and numeric_value == round(numeric_value):
                        preview.append(int(numeric_value))
                indices = tuple(preview)
                results.append(
                    _invalid_cell(
                        cell_index,
                        indices,
                        mode,
                        "invalid_cell_connectivity",
                        f"Q4 connectivity must contain exactly four indices, got shape {raw.shape}",
                    )
                )
                continue
            try:
                numeric = raw.astype(np.float64)
            except (TypeError, ValueError):
                results.append(
                    _invalid_cell(
                        cell_index,
                        (),
                        mode,
                        "invalid_cell_connectivity",
                        "Q4 connectivity contains a non-numeric index",
                    )
                )
                continue
            if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.rint(numeric)):
                results.append(
                    _invalid_cell(
                        cell_index,
                        (),
                        mode,
                        "invalid_cell_connectivity",
                        "Q4 connectivity indices must be finite integers",
                    )
                )
                continue
            indices_array = numeric.astype(np.int64)
            indices = tuple(int(value) for value in indices_array)
            if np.any(indices_array < 0) or np.any(indices_array >= len(points)):
                results.append(
                    _invalid_cell(
                        cell_index,
                        indices,
                        mode,
                        "cell_index_out_of_range",
                        "Q4 connectivity references a vertex outside the mesh array",
                    )
                )
                continue
            if np.unique(indices_array).size != 4:
                results.append(
                    _invalid_cell(
                        cell_index,
                        indices,
                        mode,
                        "repeated_cell_vertex",
                        "Q4 connectivity repeats at least one vertex index",
                    )
                )
                continue
            assessment = assess_quad_quality(points[indices_array], gate=gate, mode=mode)
            results.append(
                QuadCellQualityAssessment(
                    cell_index=cell_index,
                    vertex_indices=indices,
                    metrics=assessment.metrics,
                    mode=mode,
                    failures=assessment.failures,
                    warnings=assessment.warnings,
                )
            )

    return _build_mesh_summary(results, mode=mode)

def assess_quad_mesh_quality_gates(
    vertices: np.ndarray | Sequence[Sequence[float]],
    cells: Iterable[Iterable[int]],
    gates: dict[str, QuadQualityGate],
    *,
    mode: GateMode = "strict",
) -> dict[str, QuadMeshQualitySummary]:


    if not gates:
        raise ValueError("gates must contain at least one named quality gate")
    if mode not in {"strict", "soft"}:
        raise ValueError("mode must be either 'strict' or 'soft'")
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected mesh vertices with shape [N,2], got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Mesh vertices must contain only finite coordinates")
    raw_cells = list(cells)
    try:
        numeric = np.asarray(raw_cells).astype(np.float64)
        dense = numeric.astype(np.int64)
        valid_dense = bool(
            numeric.ndim == 2
            and numeric.shape[1] == 4
            and np.all(np.isfinite(numeric))
            and np.all(numeric == dense)
            and np.all(dense >= 0)
            and np.all(dense < len(points))
            and np.all([np.unique(row).size == 4 for row in dense])
        )
    except (TypeError, ValueError):
        valid_dense = False
        dense = np.empty((0, 4), dtype=np.int64)
    if not valid_dense:
        return {
            name: assess_quad_mesh_quality(points, raw_cells, gate=gate, mode=mode)
            for name, gate in gates.items()
        }

    metrics = quad_quality_metrics_batch(points[dense])
    summaries: dict[str, QuadMeshQualitySummary] = {}
    for name, gate in gates.items():
        results: list[QuadCellQualityAssessment] = []
        for cell_index, (indices, cell_metrics) in enumerate(zip(dense, metrics)):
            assessment = _assessment_from_metrics(cell_metrics, gate=gate, mode=mode)
            results.append(
                QuadCellQualityAssessment(
                    cell_index=cell_index,
                    vertex_indices=tuple(int(value) for value in indices),
                    metrics=cell_metrics,
                    mode=mode,
                    failures=assessment.failures,
                    warnings=assessment.warnings,
                )
            )
        summaries[name] = _build_mesh_summary(results, mode=mode)
    return summaries

def require_quad_mesh_quality(
    vertices: np.ndarray | Sequence[Sequence[float]],
    cells: Iterable[Iterable[int]],
    *,
    gate: QuadQualityGate = DEFAULT_PRODUCTION_GATE,
    mode: GateMode = "strict",
) -> QuadMeshQualitySummary:


    summary = assess_quad_mesh_quality(vertices, cells, gate=gate, mode=mode)
    if not summary.accepted:
        raise QuadMeshQualityError(summary)
    return summary

__all__ = [
    "DEFAULT_PRODUCTION_GATE",
    "GateMode",
    "QuadCellQualityAssessment",
    "QuadMeshQualityError",
    "QuadMeshQualitySummary",
    "QuadQualityAssessment",
    "QuadQualityGate",
    "QuadQualityIssue",
    "QuadQualityMetrics",
    "assess_quad_mesh_quality",
    "assess_quad_mesh_quality_gates",
    "assess_quad_quality",
    "quad_quality_metrics",
    "quad_quality_metrics_batch",
    "require_quad_mesh_quality",
]

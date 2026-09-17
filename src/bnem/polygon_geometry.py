from __future__ import annotations

from collections.abc import Callable

import numpy as np

LEGACY_VERTEX_RMS_SCHEMA = "vertex-rms-v1"
BOUNDARY_ARCLENGTH_SCHEMA = "boundary-arclength-v2"
GEOMETRY_NORMALIZATION_SCHEMAS = {
    LEGACY_VERTEX_RMS_SCHEMA,
    BOUNDARY_ARCLENGTH_SCHEMA,
}

EdgeEvaluator = Callable[[int, np.ndarray], tuple[np.ndarray, np.ndarray]]

def validate_geometry_normalization_schema(schema: str) -> str:
    value = str(schema)
    if value not in GEOMETRY_NORMALIZATION_SCHEMAS:
        raise ValueError(
            "geometry normalization schema must be one of "
            + ", ".join(sorted(GEOMETRY_NORMALIZATION_SCHEMAS))
        )
    return value

def boundary_arclength_centroid_rms(
    vertices: np.ndarray,
    *,
    edge_evaluator: EdgeEvaluator | None = None,
    quadrature_order: int = 8,
) -> tuple[np.ndarray, float]:








    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError(f"Expected at least three 2D polygon vertices, got {points.shape}")
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least two")
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    parameters = 0.5 * (nodes + 1.0)
    gauss_weights = 0.5 * weights
    sample_bank: list[np.ndarray] = []
    weight_bank: list[np.ndarray] = []
    for edge in range(points.shape[0]):
        if edge_evaluator is None:
            start = points[edge]
            end = points[(edge + 1) % points.shape[0]]
            direction = end - start
            length = float(np.linalg.norm(direction))
            if length <= 1.0e-14:
                raise ValueError(f"Polygon edge {edge} is degenerate")
            samples = (1.0 - parameters[:, None]) * start + parameters[:, None] * end
            derivatives = np.broadcast_to(direction, samples.shape)
        else:
            samples, derivatives = edge_evaluator(edge, parameters.copy())
            samples = np.asarray(samples, dtype=np.float64)
            derivatives = np.asarray(derivatives, dtype=np.float64)
            if samples.shape != (parameters.size, 2) or derivatives.shape != samples.shape:
                raise ValueError(
                    "edge_evaluator must return point and derivative arrays with shape "
                    f"({parameters.size}, 2)"
                )
        differential = np.linalg.norm(derivatives, axis=1)
        if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(differential)):
            raise ValueError("Boundary quadrature contains non-finite values")
        if np.any(differential <= 1.0e-14):
            raise ValueError(f"Polygon edge {edge} has a degenerate boundary derivative")
        sample_bank.append(samples)
        weight_bank.append(gauss_weights * differential)
    samples = np.vstack(sample_bank)
    physical_weights = np.concatenate(weight_bank)
    perimeter = float(np.sum(physical_weights))
    if not np.isfinite(perimeter) or perimeter <= 1.0e-14:
        raise ValueError("Polygon boundary has zero arclength")
    centroid = np.sum(physical_weights[:, None] * samples, axis=0) / perimeter
    radius_squared = float(
        np.sum(physical_weights * np.sum((samples - centroid) ** 2, axis=1))
        / perimeter
    )
    rms = float(np.sqrt(max(radius_squared, 0.0)))
    if not np.all(np.isfinite(centroid)) or not np.isfinite(rms) or rms <= 1.0e-14:
        raise ValueError("Polygon boundary RMS radius is degenerate")
    return np.asarray(centroid, dtype=np.float64), rms

def stable_unit_tangent_average(
    incoming: np.ndarray,
    outgoing: np.ndarray,
    *,
    fallback_strength: float = 1.0e-2,
) -> np.ndarray:


    incoming = np.asarray(incoming, dtype=np.float64)
    outgoing = np.asarray(outgoing, dtype=np.float64)
    incoming_norm = float(np.linalg.norm(incoming))
    outgoing_norm = float(np.linalg.norm(outgoing))
    if incoming.shape != (2,) or outgoing.shape != (2,):
        raise ValueError("incoming and outgoing tangents must be 2D vectors")
    if incoming_norm <= 1.0e-14 or outgoing_norm <= 1.0e-14:
        raise ValueError("Cannot form a vertex tangent from a zero-length edge")
    if fallback_strength <= 0.0:
        raise ValueError("fallback_strength must be positive")
    incoming_unit = incoming / incoming_norm
    outgoing_unit = outgoing / outgoing_norm
    tangent_sum = incoming_unit + outgoing_unit
    sum_squared = float(np.dot(tangent_sum, tangent_sum))
    fallback_weight = float(fallback_strength**2 / (sum_squared + fallback_strength**2))
    candidate = tangent_sum + fallback_weight * outgoing_unit
    candidate_norm = float(np.linalg.norm(candidate))
    if candidate_norm <= 1.0e-14:
        return outgoing_unit
    return candidate / candidate_norm

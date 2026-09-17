

from __future__ import annotations

import numpy as np

from src.bnem.quad_affine_schur import quad_boundary_mass

def _validated_quad(vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError("vertices must be finite with shape (4,2)")
    shifted = np.roll(points, -1, axis=0)
    area = 0.5 * float(
        np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1])
    )
    if area <= 1.0e-14:
        raise ValueError("vertices must be counter-clockwise and nondegenerate")
    return points

def heat_affine_trace(
    vertices: np.ndarray, *, bubbles_per_edge: int = 2
) -> np.ndarray:


    points = _validated_quad(vertices)
    bubbles = int(bubbles_per_edge)
    if bubbles < 0:
        raise ValueError("bubbles_per_edge must be nonnegative")
    scalar_dofs = 4 * (1 + bubbles)
    trace = np.zeros((scalar_dofs, 3), dtype=np.float64)
    trace[:4, 0] = 1.0
    trace[:4, 1:] = points
    return trace

def heat_affine_generalized_fluxes(
    vertices: np.ndarray, *, bubbles_per_edge: int = 2
) -> np.ndarray:


    points = _validated_quad(vertices)
    bubbles = int(bubbles_per_edge)
    if bubbles < 0:
        raise ValueError("bubbles_per_edge must be nonnegative")
    scalar_dofs = 4 * (1 + bubbles)
    forces = np.zeros((scalar_dofs, 3), dtype=np.float64)
    order = max(4, bubbles + 3)
    xi, weights = np.polynomial.legendre.leggauss(order)
    for edge in range(4):
        direction = points[(edge + 1) % 4] - points[edge]
        length = float(np.linalg.norm(direction))
        normal = np.asarray([direction[1], -direction[0]], dtype=np.float64) / length
        for coordinate, weight in zip(xi, weights, strict=True):
            t = 0.5 * (float(coordinate) + 1.0)
            shape = np.zeros(scalar_dofs, dtype=np.float64)
            shape[edge] = 1.0 - t
            shape[(edge + 1) % 4] = t
            if bubbles:
                from src.bnem.basis import legendre_values

                start = 4 + edge * bubbles
                shape[start : start + bubbles] = (
                    (1.0 - coordinate * coordinate)
                    * np.asarray(legendre_values(float(coordinate), bubbles))
                )
            integral = 0.5 * length * float(weight) * shape
            forces[:, 1] += integral * normal[0]
            forces[:, 2] += integral * normal[1]
    return forces

def heat_affine_consistency_operator(
    vertices: np.ndarray, *, bubbles_per_edge: int = 2, rcond: float = 1.0e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


    trace = heat_affine_trace(vertices, bubbles_per_edge=bubbles_per_edge)
    forces = heat_affine_generalized_fluxes(
        vertices, bubbles_per_edge=bubbles_per_edge
    )
    affine_energy = 0.5 * (trace.T @ forces + forces.T @ trace)
    values, vectors = np.linalg.eigh(affine_energy)
    scale = max(float(np.max(np.abs(values))), 1.0)
    inverse = np.zeros_like(values)
    np.divide(1.0, values, out=inverse, where=values > rcond * scale)
    pseudoinverse = (vectors * inverse[None, :]) @ vectors.T
    operator = forces @ pseudoinverse @ forces.T
    return 0.5 * (operator + operator.T), trace, forces

def heat_affine_complement_projector(
    vertices: np.ndarray, *, bubbles_per_edge: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


    trace = heat_affine_trace(vertices, bubbles_per_edge=bubbles_per_edge)
    vector_mass = quad_boundary_mass(
        vertices, bubbles_per_edge=bubbles_per_edge
    )
    scalar_dofs = trace.shape[0]
    mass = vector_mass[:scalar_dofs, :scalar_dofs]
    gram = trace.T @ mass @ trace
    projector = np.eye(scalar_dofs) - trace @ np.linalg.solve(
        gram, trace.T @ mass
    )
    return projector, mass, trace

def add_heat_psd_complement(
    affine: np.ndarray, projector: np.ndarray, factor: np.ndarray
) -> np.ndarray:
    projected = np.asarray(projector).T @ np.asarray(factor)
    result = np.asarray(affine) + projected @ projected.T
    return 0.5 * (result + result.T)

__all__ = [
    "add_heat_psd_complement",
    "heat_affine_complement_projector",
    "heat_affine_consistency_operator",
    "heat_affine_generalized_fluxes",
    "heat_affine_trace",
]

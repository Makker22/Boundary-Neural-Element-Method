














from __future__ import annotations

from functools import lru_cache
import numpy as np

from src.bnem.basis import legendre_values
from src.bnem.quad_elastic_fem_oracle import plane_strain_matrix_unit

def _validated_quad(vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError(f"vertices must be finite with shape (4,2), got {points.shape}")
    shifted = np.roll(points, -1, axis=0)
    area = 0.5 * float(
        np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1])
    )
    if area <= 1.0e-14:
        raise ValueError("vertices must be nondegenerate and counter-clockwise")
    return points

@lru_cache(maxsize=8)
def _edge_reference_integrals(bubbles: int) -> tuple[np.ndarray, np.ndarray]:
    order = max(4, int(bubbles) + 3)
    gauss, weights = np.polynomial.legendre.leggauss(order)
    mass = np.zeros((2 + bubbles, 2 + bubbles), dtype=np.float64)
    load = np.zeros(2 + bubbles, dtype=np.float64)
    for xi, weight in zip(gauss, weights, strict=True):
        shape = np.empty(2 + bubbles, dtype=np.float64)
        shape[0] = 0.5 * (1.0 - xi)
        shape[1] = 0.5 * (1.0 + xi)
        if bubbles:
            shape[2:] = (1.0 - xi * xi) * np.asarray(
                legendre_values(float(xi), bubbles), dtype=np.float64
            )
        mass += 0.5 * float(weight) * np.outer(shape, shape)
        load += 0.5 * float(weight) * shape
    return mass, load

def quad_affine_trace(vertices: np.ndarray, *, bubbles_per_edge: int = 1) -> np.ndarray:






    points = _validated_quad(vertices)
    bubbles = int(bubbles_per_edge)
    if bubbles < 0:
        raise ValueError("bubbles_per_edge must be non-negative")
    scalar_dofs = 4 * (1 + bubbles)
    trace = np.zeros((2 * scalar_dofs, 6), dtype=np.float64)
    trace[:4, 0] = 1.0
    trace[scalar_dofs : scalar_dofs + 4, 1] = 1.0
    trace[:4, 2] = points[:, 0]
    trace[:4, 3] = points[:, 1]
    trace[scalar_dofs : scalar_dofs + 4, 4] = points[:, 0]
    trace[scalar_dofs : scalar_dofs + 4, 5] = points[:, 1]
    return trace

def quad_boundary_mass(
    vertices: np.ndarray,
    *,
    bubbles_per_edge: int = 1,
    quadrature_order: int | None = None,
) -> np.ndarray:


    points = _validated_quad(vertices)
    bubbles = int(bubbles_per_edge)
    if bubbles < 0:
        raise ValueError("bubbles_per_edge must be non-negative")
    scalar_dofs = 4 * (1 + bubbles)
    mass = np.zeros((scalar_dofs, scalar_dofs), dtype=np.float64)
    edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if bubbles == 1 and quadrature_order is None:

        reference = np.asarray(
            [[1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0],
             [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0],
             [1.0 / 3.0, 1.0 / 3.0, 8.0 / 15.0]],
            dtype=np.float64,
        )
        for edge, length in enumerate(edge_lengths):
            indices = np.asarray([edge, (edge + 1) % 4, 4 + edge], dtype=np.int64)
            mass[np.ix_(indices, indices)] += float(length) * reference
        vector_mass = np.zeros((2 * scalar_dofs, 2 * scalar_dofs), dtype=np.float64)
        vector_mass[:scalar_dofs, :scalar_dofs] = mass
        vector_mass[scalar_dofs:, scalar_dofs:] = mass
        return vector_mass
    if quadrature_order is None:
        reference, _load = _edge_reference_integrals(bubbles)
        for edge, length in enumerate(edge_lengths):
            indices = np.asarray(
                [edge, (edge + 1) % 4]
                + [4 + edge * bubbles + mode for mode in range(bubbles)],
                dtype=np.int64,
            )
            mass[np.ix_(indices, indices)] += float(length) * reference
        vector_mass = np.zeros((2 * scalar_dofs, 2 * scalar_dofs), dtype=np.float64)
        vector_mass[:scalar_dofs, :scalar_dofs] = mass
        vector_mass[scalar_dofs:, scalar_dofs:] = mass
        return vector_mass

    order = int(quadrature_order or max(4, bubbles + 3))
    if order < 1:
        raise ValueError("quadrature_order must be positive")
    gauss, weights = np.polynomial.legendre.leggauss(order)
    for edge, length in enumerate(edge_lengths):
        for xi, weight in zip(gauss, weights, strict=True):
            t = 0.5 * (float(xi) + 1.0)
            shape = np.zeros(scalar_dofs, dtype=np.float64)
            shape[edge] = 1.0 - t
            shape[(edge + 1) % 4] = t
            if bubbles:
                start = 4 + edge * bubbles
                shape[start : start + bubbles] = (1.0 - xi * xi) * np.asarray(
                    legendre_values(float(xi), bubbles), dtype=np.float64
                )
            mass += 0.5 * float(length) * float(weight) * np.outer(shape, shape)
    vector_mass = np.zeros((2 * scalar_dofs, 2 * scalar_dofs), dtype=np.float64)
    vector_mass[:scalar_dofs, :scalar_dofs] = mass
    vector_mass[scalar_dofs:, scalar_dofs:] = mass
    return 0.5 * (vector_mass + vector_mass.T)

def quad_affine_generalized_forces(
    vertices: np.ndarray,
    *,
    poisson: float,
    bubbles_per_edge: int = 1,
    thickness: float = 1.0,
    quadrature_order: int | None = None,
) -> np.ndarray:


    points = _validated_quad(vertices)
    bubbles = int(bubbles_per_edge)
    if bubbles < 0:
        raise ValueError("bubbles_per_edge must be non-negative")
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness must be finite and positive")
    scalar_dofs = 4 * (1 + bubbles)
    strain = np.zeros((3, 6), dtype=np.float64)
    strain[0, 2] = 1.0
    strain[2, 3] = 1.0
    strain[2, 4] = 1.0
    strain[1, 5] = 1.0
    stress = plane_strain_matrix_unit(float(poisson)) @ strain
    forces = np.zeros((2 * scalar_dofs, 6), dtype=np.float64)
    if bubbles == 1 and quadrature_order is None:
        for edge in range(4):
            direction = points[(edge + 1) % 4] - points[edge]
            length = float(np.linalg.norm(direction))
            normal = np.asarray([direction[1], -direction[0]], dtype=np.float64) / length
            traction = np.vstack(
                (
                    stress[0] * normal[0] + stress[2] * normal[1],
                    stress[2] * normal[0] + stress[1] * normal[1],
                )
            )
            indices = np.asarray([edge, (edge + 1) % 4, 4 + edge], dtype=np.int64)
            integrals = float(thickness) * length * np.asarray([0.5, 0.5, 2.0 / 3.0])
            forces[indices] += integrals[:, None] * traction[0][None, :]
            forces[scalar_dofs + indices] += integrals[:, None] * traction[1][None, :]
        return forces
    if quadrature_order is None:
        _mass, reference_load = _edge_reference_integrals(bubbles)
        for edge in range(4):
            direction = points[(edge + 1) % 4] - points[edge]
            length = float(np.linalg.norm(direction))
            normal = np.asarray([direction[1], -direction[0]], dtype=np.float64) / length
            traction = np.vstack(
                (
                    stress[0] * normal[0] + stress[2] * normal[1],
                    stress[2] * normal[0] + stress[1] * normal[1],
                )
            )
            indices = np.asarray(
                [edge, (edge + 1) % 4]
                + [4 + edge * bubbles + mode for mode in range(bubbles)],
                dtype=np.int64,
            )
            integrals = float(thickness) * length * reference_load
            forces[indices] += integrals[:, None] * traction[0][None, :]
            forces[scalar_dofs + indices] += integrals[:, None] * traction[1][None, :]
        return forces

    order = int(quadrature_order or max(4, bubbles + 3))
    gauss, weights = np.polynomial.legendre.leggauss(order)
    for edge in range(4):
        direction = points[(edge + 1) % 4] - points[edge]
        length = float(np.linalg.norm(direction))
        normal = np.asarray([direction[1], -direction[0]], dtype=np.float64) / length
        traction = np.vstack(
            (
                stress[0] * normal[0] + stress[2] * normal[1],
                stress[2] * normal[0] + stress[1] * normal[1],
            )
        )
        for xi, weight in zip(gauss, weights, strict=True):
            t = 0.5 * (float(xi) + 1.0)
            shape = np.zeros(scalar_dofs, dtype=np.float64)
            shape[edge] = 1.0 - t
            shape[(edge + 1) % 4] = t
            if bubbles:
                start = 4 + edge * bubbles
                shape[start : start + bubbles] = (1.0 - xi * xi) * np.asarray(
                    legendre_values(float(xi), bubbles), dtype=np.float64
                )
            scale = 0.5 * length * float(weight) * float(thickness)
            forces[:scalar_dofs] += scale * shape[:, None] * traction[0][None, :]
            forces[scalar_dofs:] += scale * shape[:, None] * traction[1][None, :]
    return forces

def quad_affine_consistency_operator(
    vertices: np.ndarray,
    *,
    poisson: float,
    bubbles_per_edge: int = 1,
    thickness: float = 1.0,
    rcond: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


    trace = quad_affine_trace(vertices, bubbles_per_edge=bubbles_per_edge)
    forces = quad_affine_generalized_forces(
        vertices,
        poisson=poisson,
        bubbles_per_edge=bubbles_per_edge,
        thickness=thickness,
    )
    affine_energy = 0.5 * (trace.T @ forces + forces.T @ trace)
    eigenvalues, eigenvectors = np.linalg.eigh(affine_energy)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    keep = eigenvalues > float(rcond) * scale
    inverse = np.zeros_like(eigenvalues)
    np.divide(1.0, eigenvalues, out=inverse, where=keep)
    pseudoinverse = (eigenvectors * inverse[None, :]) @ eigenvectors.T
    operator = forces @ pseudoinverse @ forces.T
    return 0.5 * (operator + operator.T), trace, forces

def mass_affine_complement_projector(
    vertices: np.ndarray,
    *,
    bubbles_per_edge: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


    trace = quad_affine_trace(vertices, bubbles_per_edge=bubbles_per_edge)
    mass = quad_boundary_mass(vertices, bubbles_per_edge=bubbles_per_edge)
    gram = trace.T @ mass @ trace
    projector = np.eye(trace.shape[0], dtype=np.float64) - trace @ np.linalg.solve(
        gram, trace.T @ mass
    )
    return projector, mass, trace

def add_psd_affine_complement_stabilization(
    affine_operator: np.ndarray,
    complement_projector: np.ndarray,
    factor: np.ndarray,
) -> np.ndarray:


    base = np.asarray(affine_operator, dtype=np.float64)
    projector = np.asarray(complement_projector, dtype=np.float64)
    learned_factor = np.asarray(factor, dtype=np.float64)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("affine_operator must be square")
    if projector.shape != base.shape:
        raise ValueError("complement_projector must match affine_operator")
    if learned_factor.ndim != 2 or learned_factor.shape[0] != base.shape[0]:
        raise ValueError("factor must have one row per port degree of freedom")
    projected_factor = projector.T @ learned_factor
    result = base + projected_factor @ projected_factor.T
    return 0.5 * (result + result.T)

__all__ = [
    "add_psd_affine_complement_stabilization",
    "mass_affine_complement_projector",
    "quad_affine_consistency_operator",
    "quad_affine_generalized_forces",
    "quad_affine_trace",
    "quad_boundary_mass",
]

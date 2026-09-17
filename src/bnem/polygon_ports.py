from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bnem.basis import legendre_values

def counter_clockwise_vertices(vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError(f"Expected at least three 2D polygon vertices, got {points.shape}")
    shifted = np.roll(points, -1, axis=0)
    signed_area = 0.5 * float(np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1]))
    if abs(signed_area) <= 1.0e-14:
        raise ValueError("Polygon vertices are degenerate")
    return points if signed_area > 0.0 else points[::-1].copy()

@dataclass(frozen=True)
class PolygonVertexBubblePort:
    vertices: np.ndarray
    bubbles_per_edge: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", counter_clockwise_vertices(self.vertices))
        if self.bubbles_per_edge < 0:
            raise ValueError("bubbles_per_edge must be nonnegative")

    @property
    def edge_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def scalar_dofs(self) -> int:
        return self.edge_count * (1 + self.bubbles_per_edge)

    @property
    def vector_dofs(self) -> int:
        return 2 * self.scalar_dofs

    def shape_on_edge(self, edge_index: int, t: float) -> np.ndarray:
        edge = int(edge_index) % self.edge_count
        parameter = float(np.clip(t, 0.0, 1.0))
        values = np.zeros(self.scalar_dofs, dtype=np.float64)
        left = edge
        right = (edge + 1) % self.edge_count
        values[left] = 1.0 - parameter
        values[right] = parameter
        xi = 2.0 * parameter - 1.0
        if self.bubbles_per_edge:
            bubble = (1.0 - xi * xi) * np.asarray(
                legendre_values(xi, self.bubbles_per_edge), dtype=np.float64
            )
            start = self.edge_count + edge * self.bubbles_per_edge
            values[start : start + self.bubbles_per_edge] = bubble
        return values

    def locate_boundary_point(self, point: np.ndarray, *, tol: float | None = None) -> tuple[int, float]:
        xy = np.asarray(point, dtype=np.float64)
        span = np.ptp(self.vertices, axis=0)
        tolerance = float(tol if tol is not None else 1.0e-7 * max(float(np.max(span)), 1.0))
        vertex_distances = np.linalg.norm(self.vertices - xy, axis=1)
        vertex_index = int(np.argmin(vertex_distances))
        if float(vertex_distances[vertex_index]) <= tolerance:
            return vertex_index, 0.0

        best: tuple[float, int, float] | None = None
        for edge in range(self.edge_count):
            start = self.vertices[edge]
            end = self.vertices[(edge + 1) % self.edge_count]
            direction = end - start
            denom = float(np.dot(direction, direction))
            if denom <= 1.0e-30:
                continue
            parameter = float(np.clip(np.dot(xy - start, direction) / denom, 0.0, 1.0))
            projected = start + parameter * direction
            distance = float(np.linalg.norm(xy - projected))
            candidate = (distance, edge, parameter)
            if best is None or candidate < best:
                best = candidate
        if best is None or best[0] > tolerance:
            raise ValueError(f"Point {xy.tolist()} is not on the polygon boundary (distance={None if best is None else best[0]})")
        return int(best[1]), float(best[2])

    def shape_at_boundary_point(self, point: np.ndarray, *, tol: float | None = None) -> np.ndarray:
        edge, parameter = self.locate_boundary_point(point, tol=tol)
        if parameter == 0.0:
            values = np.zeros(self.scalar_dofs, dtype=np.float64)
            values[edge] = 1.0
            return values
        return self.shape_on_edge(edge, parameter)

    def vector_shapes(self, scalar_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scalar = np.asarray(scalar_shape, dtype=np.float64)
        if scalar.shape != (self.scalar_dofs,):
            raise ValueError(f"Expected scalar shape ({self.scalar_dofs},), got {scalar.shape}")
        shape_u1 = np.zeros(self.vector_dofs, dtype=np.float64)
        shape_u2 = np.zeros(self.vector_dofs, dtype=np.float64)
        shape_u1[: self.scalar_dofs] = scalar
        shape_u2[self.scalar_dofs :] = scalar
        return shape_u1, shape_u2

    def rigid_body_modes(self) -> np.ndarray:
        modes = np.zeros((self.vector_dofs, 3), dtype=np.float64)
        modes[: self.edge_count, 0] = 1.0
        modes[self.scalar_dofs : self.scalar_dofs + self.edge_count, 1] = 1.0
        modes[: self.edge_count, 2] = -self.vertices[:, 1]
        modes[self.scalar_dofs : self.scalar_dofs + self.edge_count, 2] = self.vertices[:, 0]
        return modes

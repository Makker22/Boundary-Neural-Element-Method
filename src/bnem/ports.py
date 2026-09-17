from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bnem.basis import legendre_values

@dataclass(frozen=True)
class VertexBubblePort:
    bubbles_per_edge: int = 2

    @property
    def port_dofs(self) -> int:
        return 4 + 4 * self.bubbles_per_edge

    def bubble_values(self, xi: float) -> list[float]:
        if self.bubbles_per_edge == 0:
            return []
        legendre = legendre_values(xi, self.bubbles_per_edge)
        scale = 1.0 - xi * xi
        return [scale * value for value in legendre]

    def shape_vector(self, i: int, j: int, nx: int, ny: int) -> list[float]:
        values = [0.0] * self.port_dofs

        if i == 0 and j == 0:
            values[0] = 1.0
            return values
        if i == nx and j == 0:
            values[1] = 1.0
            return values
        if i == nx and j == ny:
            values[2] = 1.0
            return values
        if i == 0 and j == ny:
            values[3] = 1.0
            return values

        if j == 0:
            xi = 2.0 * i / nx - 1.0
            self._fill_edge(values, edge_id=0, start_vertex=0, end_vertex=1, xi=xi)
        elif i == nx:
            xi = 2.0 * j / ny - 1.0
            self._fill_edge(values, edge_id=1, start_vertex=1, end_vertex=2, xi=xi)
        elif j == ny:
            xi = 1.0 - 2.0 * i / nx
            self._fill_edge(values, edge_id=2, start_vertex=2, end_vertex=3, xi=xi)
        elif i == 0:
            xi = 1.0 - 2.0 * j / ny
            self._fill_edge(values, edge_id=3, start_vertex=3, end_vertex=0, xi=xi)
        else:
            raise ValueError("shape_vector only accepts boundary nodes")
        return values

    def _fill_edge(
        self,
        values: list[float],
        *,
        edge_id: int,
        start_vertex: int,
        end_vertex: int,
        xi: float,
    ) -> None:
        values[start_vertex] = 0.5 * (1.0 - xi)
        values[end_vertex] = 0.5 * (1.0 + xi)
        offset = 4 + edge_id * self.bubbles_per_edge
        for mode, value in enumerate(self.bubble_values(xi)):
            values[offset + mode] = value

def vertex_bubble_matrix(points: np.ndarray, bubbles_per_edge: int) -> np.ndarray:
    port = VertexBubblePort(bubbles_per_edge=bubbles_per_edge)
    rows = []
    for xi in points:
        rows.append(port.bubble_values(float(xi)))
    return np.asarray(rows, dtype=np.float64)

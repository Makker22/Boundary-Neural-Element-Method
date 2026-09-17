

from __future__ import annotations

import numpy as np

def u1_port_to_q8_nodal(local_u1: np.ndarray) -> np.ndarray:






    values = np.asarray(local_u1, dtype=np.float64)
    if values.shape != (16,):
        raise ValueError("local_u1 must contain 16 component-major coefficients")
    x = values[:8]
    y = values[8:]
    nodal = np.empty((8, 2), dtype=np.float64)
    nodal[:4, 0] = x[:4]
    nodal[:4, 1] = y[:4]
    for edge in range(4):
        right = (edge + 1) % 4
        nodal[4 + edge, 0] = 0.5 * (x[edge] + x[right]) + x[4 + edge]
        nodal[4 + edge, 1] = 0.5 * (y[edge] + y[right]) + y[4 + edge]
    return nodal

def q8_shape_data_vectorized(
    xi: np.ndarray, eta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    shape = np.column_stack(
        [
            -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
            -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
            -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
            -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
            0.5 * (1 - xi * xi) * (1 - eta),
            0.5 * (1 + xi) * (1 - eta * eta),
            0.5 * (1 - xi * xi) * (1 + eta),
            0.5 * (1 - xi) * (1 - eta * eta),
        ]
    )
    dxi = np.column_stack(
        [
            0.25 * (1 - eta) * (2 * xi + eta),
            0.25 * (1 - eta) * (2 * xi - eta),
            0.25 * (1 + eta) * (2 * xi + eta),
            0.25 * (1 + eta) * (2 * xi - eta),
            -xi * (1 - eta),
            0.5 * (1 - eta * eta),
            -xi * (1 + eta),
            -0.5 * (1 - eta * eta),
        ]
    )
    deta = np.column_stack(
        [
            0.25 * (1 - xi) * (xi + 2 * eta),
            0.25 * (1 + xi) * (-xi + 2 * eta),
            0.25 * (1 + xi) * (xi + 2 * eta),
            0.25 * (1 - xi) * (-xi + 2 * eta),
            -0.5 * (1 - xi * xi),
            -(1 + xi) * eta,
            0.5 * (1 - xi * xi),
            -(1 - xi) * eta,
        ]
    )
    return shape, dxi, deta

def q8_geometry_nodes(corner_vertices: np.ndarray) -> np.ndarray:
    corners = np.asarray(corner_vertices, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError("corner_vertices must have shape (4,2)")
    return np.vstack((corners, 0.5 * (corners + np.roll(corners, -1, axis=0))))

__all__ = ["q8_geometry_nodes", "q8_shape_data_vectorized", "u1_port_to_q8_nodal"]

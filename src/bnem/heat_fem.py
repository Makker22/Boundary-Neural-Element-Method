

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from src.bnem.basis import legendre_values
from src.bnem.ports import VertexBubblePort

def inward_normal(start: Sequence[float], end: Sequence[float]) -> tuple[float, float]:
    tx = float(end[0]) - float(start[0])
    ty = float(end[1]) - float(start[1])
    length = math.hypot(tx, ty)
    if length <= 1.0e-12:
        raise ValueError("degenerate curved edge")
    return -ty / length, tx / length

def mapped_xy(
    i: int,
    j: int,
    nx: int,
    ny: int,
    vertices: Sequence[Sequence[float]],
    edge_bulges: Sequence[float] | None = None,
) -> tuple[float, float]:
    r = float(i) / float(nx)
    s = float(j) / float(ny)
    v0, v1, v2, v3 = vertices
    x = (
        (1.0 - r) * (1.0 - s) * v0[0]
        + r * (1.0 - s) * v1[0]
        + r * s * v2[0]
        + (1.0 - r) * s * v3[0]
    )
    y = (
        (1.0 - r) * (1.0 - s) * v0[1]
        + r * (1.0 - s) * v1[1]
        + r * s * v2[1]
        + (1.0 - r) * s * v3[1]
    )
    bulges = edge_bulges or [0.0, 0.0, 0.0, 0.0]
    edges = (
        (v0, v1, r, (1.0 - s) ** 2),
        (v1, v2, s, r**2),
        (v2, v3, 1.0 - r, s**2),
        (v3, v0, 1.0 - s, (1.0 - r) ** 2),
    )
    for bulge, (start, end, xi01, fade) in zip(bulges, edges):
        if float(bulge) == 0.0:
            continue
        nx_in, ny_in = inward_normal(start, end)
        weight = math.sin(math.pi * xi01) * fade
        x += float(bulge) * weight * nx_in
        y += float(bulge) * weight * ny_in
    return float(x), float(y)

def q4_heat_stiffness(coords: np.ndarray, conductivity: float = 1.0) -> np.ndarray:
    gauss = 1.0 / math.sqrt(3.0)
    points = ((-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss))
    element = np.zeros((4, 4), dtype=np.float64)
    xy = np.asarray(coords, dtype=np.float64)
    if xy.shape != (4, 2):
        raise ValueError("Q4 coordinates must have shape (4, 2)")
    for xi, eta in points:
        derivatives = np.asarray(
            [
                [-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)],
                [0.25 * (1.0 - eta), -0.25 * (1.0 + xi)],
                [0.25 * (1.0 + eta), 0.25 * (1.0 + xi)],
                [-0.25 * (1.0 + eta), 0.25 * (1.0 - xi)],
            ],
            dtype=np.float64,
        )
        jacobian = derivatives.T @ xy
        determinant = float(np.linalg.det(jacobian))
        if determinant <= 1.0e-14:
            raise ValueError(f"degenerate Q4 element with detJ={determinant}")
        gradient = derivatives @ np.linalg.inv(jacobian).T
        element += float(conductivity) * (gradient @ gradient.T) * determinant
    return element

def node_index(i: int, j: int, nx: int) -> int:
    return j * (nx + 1) + i

def build_structured_mesh(
    *,
    nx: int,
    ny: int,
    vertices: Sequence[Sequence[float]],
    edge_bulges: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    nodes = [
        mapped_xy(i, j, nx, ny, vertices, edge_bulges)
        for j in range(ny + 1)
        for i in range(nx + 1)
    ]
    elements = [
        [
            node_index(i, j, nx),
            node_index(i + 1, j, nx),
            node_index(i + 1, j + 1, nx),
            node_index(i, j + 1, nx),
        ]
        for j in range(ny)
        for i in range(nx)
    ]
    return np.asarray(nodes, dtype=np.float64), np.asarray(elements, dtype=np.int64)

def assemble_global_q4(
    nodes: np.ndarray,
    elements: np.ndarray,
    *,
    conductivity: float = 1.0,
) -> csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for connectivity in np.asarray(elements, dtype=np.int64):
        element = q4_heat_stiffness(nodes[connectivity], conductivity)
        for local_i, global_i in enumerate(connectivity):
            for local_j, global_j in enumerate(connectivity):
                rows.append(int(global_i))
                columns.append(int(global_j))
                values.append(float(element[local_i, local_j]))
    return coo_matrix(
        (values, (rows, columns)), shape=(nodes.shape[0], nodes.shape[0])
    ).tocsr()

def boundary_node_indices(nx: int, ny: int) -> list[int]:
    return [
        node_index(i, j, nx)
        for j in range(ny + 1)
        for i in range(nx + 1)
        if i in (0, nx) or j in (0, ny)
    ]

def boundary_edges_for_node(
    i: int, j: int, nx: int, ny: int
) -> list[tuple[int, float]]:
    edges: list[tuple[int, float]] = []
    if j == 0:
        edges.append((0, 2.0 * i / nx - 1.0))
    if i == nx:
        edges.append((1, 2.0 * j / ny - 1.0))
    if j == ny:
        edges.append((2, 1.0 - 2.0 * i / nx))
    if i == 0:
        edges.append((3, 1.0 - 2.0 * j / ny))
    return edges

def boundary_shape_vector(
    i: int,
    j: int,
    nx: int,
    ny: int,
    modes_per_edge: int,
    port_scheme: str,
    bubbles_per_edge: int,
) -> list[float]:
    if port_scheme == "vertex_bubble":
        return VertexBubblePort(bubbles_per_edge=bubbles_per_edge).shape_vector(
            i, j, nx, ny
        )
    values = [0.0] * (4 * modes_per_edge)
    for edge_id, coordinate in boundary_edges_for_node(i, j, nx, ny):
        offset = edge_id * modes_per_edge
        for mode, value in enumerate(legendre_values(coordinate, modes_per_edge)):
            values[offset + mode] += value
    return values

def boundary_shape_matrix(
    *,
    nx: int,
    ny: int,
    boundary: Sequence[int],
    modes_per_edge: int,
    bubbles_per_edge: int,
) -> np.ndarray:
    matrix = np.zeros((len(boundary), 4 + 4 * bubbles_per_edge), dtype=np.float64)
    for row, node in enumerate(boundary):
        j, i = divmod(int(node), nx + 1)
        matrix[row] = boundary_shape_vector(
            i,
            j,
            nx,
            ny,
            modes_per_edge,
            "vertex_bubble",
            bubbles_per_edge,
        )
    return matrix

def schur_to_boundary(matrix: csr_matrix, boundary: Sequence[int]) -> np.ndarray:
    boundary_array = np.asarray(boundary, dtype=np.int64)
    is_boundary = np.zeros(matrix.shape[0], dtype=bool)
    is_boundary[boundary_array] = True
    interior = np.nonzero(~is_boundary)[0]
    boundary_block = matrix[boundary_array, :][:, boundary_array].toarray()
    if interior.size == 0:
        return boundary_block
    boundary_interior = matrix[boundary_array, :][:, interior]
    interior_boundary = matrix[interior, :][:, boundary_array]
    interior_block = matrix[interior, :][:, interior]
    return np.asarray(
        boundary_block
        - boundary_interior @ spsolve(interior_block, interior_boundary.toarray()),
        dtype=np.float64,
    )

def port_schur_matrix(
    *,
    nx: int,
    ny: int,
    vertices: Sequence[Sequence[float]],
    edge_bulges: Sequence[float] | None = None,
    conductivity: float = 1.0,
    modes_per_edge: int = 4,
    bubbles_per_edge: int = 2,
) -> np.ndarray:
    nodes, elements = build_structured_mesh(
        nx=nx,
        ny=ny,
        vertices=vertices,
        edge_bulges=edge_bulges,
    )
    global_matrix = assemble_global_q4(
        nodes, elements, conductivity=conductivity
    )
    boundary = boundary_node_indices(nx, ny)
    condensed = schur_to_boundary(global_matrix, boundary)
    trace = boundary_shape_matrix(
        nx=nx,
        ny=ny,
        boundary=boundary,
        modes_per_edge=modes_per_edge,
        bubbles_per_edge=bubbles_per_edge,
    )
    operator = trace.T @ condensed @ trace
    return 0.5 * (operator + operator.T)

__all__ = [
    "assemble_global_q4",
    "boundary_node_indices",
    "boundary_shape_matrix",
    "build_structured_mesh",
    "mapped_xy",
    "port_schur_matrix",
    "q4_heat_stiffness",
    "schur_to_boundary",
]

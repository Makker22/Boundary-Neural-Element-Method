from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

from src.bnem.polygon_ports import PolygonVertexBubblePort

def plane_strain_matrix_unit(poisson: float) -> np.ndarray:


    nu = float(poisson)
    if not (-0.999 < nu < 0.499):
        raise ValueError("plane strain requires -0.999 < poisson < 0.499")
    return np.asarray(
        [
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)],
        ],
        dtype=np.float64,
    ) / ((1.0 + nu) * (1.0 - 2.0 * nu))

def q8_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


    n = np.asarray(
        [
            -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
            -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
            -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
            -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
            0.5 * (1 - xi * xi) * (1 - eta),
            0.5 * (1 + xi) * (1 - eta * eta),
            0.5 * (1 - xi * xi) * (1 + eta),
            0.5 * (1 - xi) * (1 - eta * eta),
        ],
        dtype=np.float64,
    )
    dxi = np.asarray(
        [
            0.25 * (1 - eta) * (2 * xi + eta),
            0.25 * (1 - eta) * (2 * xi - eta),
            0.25 * (1 + eta) * (2 * xi + eta),
            0.25 * (1 + eta) * (2 * xi - eta),
            -xi * (1 - eta),
            0.5 * (1 - eta * eta),
            -xi * (1 + eta),
            -0.5 * (1 - eta * eta),
        ],
        dtype=np.float64,
    )
    deta = np.asarray(
        [
            0.25 * (1 - xi) * (xi + 2 * eta),
            0.25 * (1 + xi) * (-xi + 2 * eta),
            0.25 * (1 + xi) * (xi + 2 * eta),
            0.25 * (1 - xi) * (-xi + 2 * eta),
            -0.5 * (1 - xi * xi),
            -(1 + xi) * eta,
            0.5 * (1 - xi * xi),
            -(1 - xi) * eta,
        ],
        dtype=np.float64,
    )
    return n, dxi, deta

def bilinear_quad_map(vertices: np.ndarray, xi: float, eta: float) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    shape = 0.25 * np.asarray(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=np.float64,
    )
    return shape @ points

@dataclass(frozen=True)
class Q8QuadMesh:
    nodes: np.ndarray
    elements: np.ndarray
    lattice_to_node: dict[tuple[int, int], int]
    subdivisions: int

@dataclass(frozen=True)
class CondensedQ8PortOracle:


    matrix: np.ndarray
    mesh: Q8QuadMesh
    boundary_dofs: np.ndarray
    interior_dofs: np.ndarray
    boundary_port_map: np.ndarray
    interior_port_map: np.ndarray

    def recover_displacement(self, port_coefficients: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(port_coefficients, dtype=np.float64)
        port_dofs = int(self.boundary_port_map.shape[1])
        if coefficients.shape != (port_dofs,):
            raise ValueError(
                f"expected {port_dofs} port coefficients, got {coefficients.shape}"
            )
        values = np.zeros(2 * int(self.mesh.nodes.shape[0]), dtype=np.float64)
        values[self.boundary_dofs] = self.boundary_port_map @ coefficients
        if self.interior_dofs.size:
            values[self.interior_dofs] = self.interior_port_map @ coefficients
        return values.reshape((-1, 2))

def structured_q8_quad_mesh(vertices: np.ndarray, subdivisions: int) -> Q8QuadMesh:
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (4, 2):
        raise ValueError(f"expected four 2D vertices, got {points.shape}")
    shifted = np.roll(points, -1, axis=0)
    signed_area = 0.5 * float(
        np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1])
    )
    if signed_area <= 1.0e-12:
        raise ValueError("vertices must be nondegenerate and counter-clockwise")
    n = int(subdivisions)
    if n < 1:
        raise ValueError("subdivisions must be positive")

    lattice_to_node: dict[tuple[int, int], int] = {}
    nodes: list[np.ndarray] = []
    extent = 2 * n
    for j in range(extent + 1):
        for i in range(extent + 1):
            if i % 2 == 1 and j % 2 == 1:
                continue
            xi = -1.0 + 2.0 * float(i) / float(extent)
            eta = -1.0 + 2.0 * float(j) / float(extent)
            lattice_to_node[(i, j)] = len(nodes)
            nodes.append(bilinear_quad_map(points, xi, eta))

    elements: list[list[int]] = []
    for ey in range(n):
        for ex in range(n):
            i = 2 * ex
            j = 2 * ey
            elements.append(
                [
                    lattice_to_node[(i, j)],
                    lattice_to_node[(i + 2, j)],
                    lattice_to_node[(i + 2, j + 2)],
                    lattice_to_node[(i, j + 2)],
                    lattice_to_node[(i + 1, j)],
                    lattice_to_node[(i + 2, j + 1)],
                    lattice_to_node[(i + 1, j + 2)],
                    lattice_to_node[(i, j + 1)],
                ]
            )
    return Q8QuadMesh(
        nodes=np.asarray(nodes, dtype=np.float64),
        elements=np.asarray(elements, dtype=np.int64),
        lattice_to_node=lattice_to_node,
        subdivisions=n,
    )

def q8_element_stiffness(
    coordinates: np.ndarray,
    *,
    poisson: float,
    thickness: float = 1.0,
) -> np.ndarray:
    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.shape != (8, 2):
        raise ValueError(f"expected Q8 coordinates (8,2), got {xy.shape}")
    gauss, weights = np.polynomial.legendre.leggauss(3)
    constitutive = plane_strain_matrix_unit(poisson)
    stiffness = np.zeros((16, 16), dtype=np.float64)
    for xi, wx in zip(gauss, weights, strict=True):
        for eta, wy in zip(gauss, weights, strict=True):
            _shape, dxi, deta = q8_shape(float(xi), float(eta))
            parametric = np.vstack([dxi, deta])
            jacobian = parametric @ xy
            determinant = float(np.linalg.det(jacobian))
            if determinant <= 1.0e-14:
                raise ValueError(f"nonpositive Q8 Jacobian determinant {determinant}")
            physical = np.linalg.solve(jacobian, parametric)
            d_dx = physical[0]
            d_dy = physical[1]
            bmat = np.zeros((3, 16), dtype=np.float64)
            for local in range(8):
                bmat[0, 2 * local] = d_dx[local]
                bmat[1, 2 * local + 1] = d_dy[local]
                bmat[2, 2 * local] = d_dy[local]
                bmat[2, 2 * local + 1] = d_dx[local]
            stiffness += (
                float(thickness)
                * float(wx)
                * float(wy)
                * determinant
                * (bmat.T @ constitutive @ bmat)
            )
    return 0.5 * (stiffness + stiffness.T)

def assemble_q8_stiffness(
    mesh: Q8QuadMesh,
    *,
    poisson: float,
    thickness: float = 1.0,
) -> csc_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for element in mesh.elements:
        local = q8_element_stiffness(
            mesh.nodes[element], poisson=poisson, thickness=thickness
        )
        dofs = np.asarray(
            [[2 * int(node), 2 * int(node) + 1] for node in element], dtype=np.int64
        ).reshape(-1)
        rows.extend(np.repeat(dofs, dofs.size).tolist())
        columns.extend(np.tile(dofs, dofs.size).tolist())
        values.extend(local.reshape(-1).tolist())
    size = 2 * int(mesh.nodes.shape[0])
    return coo_matrix((values, (rows, columns)), shape=(size, size)).tocsc()

def _boundary_node_shapes(
    vertices: np.ndarray,
    mesh: Q8QuadMesh,
    *,
    bubbles_per_edge: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(mesh.subdivisions)
    extent = 2 * n
    port = PolygonVertexBubblePort(vertices, bubbles_per_edge=bubbles_per_edge)
    boundary_rows: list[tuple[int, int, float]] = []
    for (i, j), node in mesh.lattice_to_node.items():
        if j == 0:
            edge, parameter = 0, float(i) / float(extent)
        elif i == extent:
            edge, parameter = 1, float(j) / float(extent)
        elif j == extent:
            edge, parameter = 2, float(extent - i) / float(extent)
        elif i == 0:
            edge, parameter = 3, float(extent - j) / float(extent)
        else:
            continue
        boundary_rows.append((int(node), edge, parameter))
    boundary_rows.sort(key=lambda row: row[0])
    boundary_nodes = np.asarray([row[0] for row in boundary_rows], dtype=np.int64)
    scalar_dofs = int(port.scalar_dofs)
    lift = np.zeros((2 * boundary_nodes.size, 2 * scalar_dofs), dtype=np.float64)
    for row, (_node, edge, parameter) in enumerate(boundary_rows):
        shape = port.shape_on_edge(edge, parameter)
        lift[2 * row, :scalar_dofs] = shape
        lift[2 * row + 1, scalar_dofs:] = shape
    return boundary_nodes, lift

def build_condensed_q8_port_oracle(
    vertices: np.ndarray,
    *,
    poisson: float,
    subdivisions: int = 8,
    bubbles_per_edge: int = 1,
    thickness: float = 1.0,
) -> CondensedQ8PortOracle:


    mesh = structured_q8_quad_mesh(vertices, subdivisions)
    stiffness = assemble_q8_stiffness(mesh, poisson=poisson, thickness=thickness)
    boundary_nodes, lift = _boundary_node_shapes(
        vertices, mesh, bubbles_per_edge=bubbles_per_edge
    )
    boundary_dofs = np.asarray(
        [[2 * int(node), 2 * int(node) + 1] for node in boundary_nodes], dtype=np.int64
    ).reshape(-1)
    all_dofs = np.arange(stiffness.shape[0], dtype=np.int64)
    interior_mask = np.ones(stiffness.shape[0], dtype=bool)
    interior_mask[boundary_dofs] = False
    interior_dofs = all_dofs[interior_mask]
    kbb = stiffness[boundary_dofs][:, boundary_dofs]
    direct = np.asarray(lift.T @ (kbb @ lift), dtype=np.float64)
    interior_port_map = np.empty((0, lift.shape[1]), dtype=np.float64)
    if interior_dofs.size:
        kib = stiffness[interior_dofs][:, boundary_dofs]
        coupling = np.asarray(kib @ lift, dtype=np.float64)
        kii = stiffness[interior_dofs][:, interior_dofs].tocsc()
        solved = splu(kii).solve(coupling)
        direct -= coupling.T @ solved
        interior_port_map = -np.asarray(solved, dtype=np.float64)
    return CondensedQ8PortOracle(
        matrix=0.5 * (direct + direct.T),
        mesh=mesh,
        boundary_dofs=boundary_dofs,
        interior_dofs=interior_dofs,
        boundary_port_map=lift,
        interior_port_map=interior_port_map,
    )

def condensed_q8_port_operator(
    vertices: np.ndarray,
    *,
    poisson: float,
    subdivisions: int = 8,
    bubbles_per_edge: int = 1,
    thickness: float = 1.0,
) -> np.ndarray:


    return build_condensed_q8_port_oracle(
        vertices,
        poisson=poisson,
        subdivisions=subdivisions,
        bubbles_per_edge=bubbles_per_edge,
        thickness=thickness,
    ).matrix

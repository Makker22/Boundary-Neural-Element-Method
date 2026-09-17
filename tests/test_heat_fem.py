from __future__ import annotations

import numpy as np

from src.bnem.heat_fem import port_schur_matrix, q4_heat_stiffness

def test_q4_element_is_symmetric_and_preserves_constants() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0], [1.2, 0.08], [1.05, 1.1], [-0.12, 0.92]],
        dtype=np.float64,
    )
    matrix = q4_heat_stiffness(coordinates, conductivity=1.7)
    np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-13)
    np.testing.assert_allclose(matrix @ np.ones(4), 0.0, atol=1.0e-13)
    assert np.linalg.eigvalsh(matrix).min() >= -1.0e-12

def test_condensed_port_operator_is_physical() -> None:
    vertices = [[0.0, 0.0], [1.2, 0.08], [1.05, 1.1], [-0.12, 0.92]]
    matrix = port_schur_matrix(
        nx=6,
        ny=6,
        vertices=vertices,
        edge_bulges=[0.0] * 4,
        conductivity=1.0,
        modes_per_edge=4,
        bubbles_per_edge=2,
    )
    assert matrix.shape == (12, 12)
    np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-12)
    np.testing.assert_allclose(matrix @ np.r_[np.ones(4), np.zeros(8)], 0.0, atol=2.0e-12)
    assert np.linalg.eigvalsh(matrix).min() >= -2.0e-12

def test_condensed_operator_scales_with_conductivity() -> None:
    arguments = {
        "nx": 4,
        "ny": 4,
        "vertices": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "edge_bulges": [0.0] * 4,
        "modes_per_edge": 4,
        "bubbles_per_edge": 2,
    }
    unit = port_schur_matrix(conductivity=1.0, **arguments)
    scaled = port_schur_matrix(conductivity=2.5, **arguments)
    np.testing.assert_allclose(scaled, 2.5 * unit, rtol=2.0e-12, atol=2.0e-12)

from pathlib import Path

import numpy as np

from src.bnem.compact_vertex_rank2 import (
    load_compact_vertex_rank2_operator,
    predict_compact_vertex_stiffnesses,
)


def test_released_elasticity_operator_is_symmetric_and_semidefinite() -> None:
    root = Path(__file__).resolve().parents[1]
    operator = load_compact_vertex_rank2_operator(
        root / "models" / "elasticity_compact_vertex_rank2.pt"
    )
    vertices = np.asarray(
        [[[0.0, 0.0], [1.1, 0.05], [1.0, 1.0], [-0.08, 0.9]]],
        dtype=np.float64,
    )
    matrices, _ = predict_compact_vertex_stiffnesses(
        operator, vertices, poisson=0.49, young=210_000.0
    )
    matrix = matrices[0]
    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-10)
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues[0] >= -1.0e-8 * eigenvalues[-1]
    assert np.count_nonzero(eigenvalues > 1.0e-7 * eigenvalues[-1]) == 5

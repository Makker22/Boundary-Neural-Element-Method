

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bnem.compact_vertex_rank2 import (
    load_compact_vertex_rank2_operator,
    predict_compact_vertex_stiffnesses,
)


def main() -> int:
    vertices = np.asarray(
        [[[0.0, 0.0], [1.15, 0.08], [1.02, 1.07], [-0.10, 0.91]]],
        dtype=np.float64,
    )
    operator = load_compact_vertex_rank2_operator(
        ROOT / "models" / "elasticity_compact_vertex_rank2.pt"
    )
    stiffness, timing = predict_compact_vertex_stiffnesses(
        operator,
        vertices,
        poisson=0.40,
        young=210_000.0,
        thickness=1.0,
    )
    matrix = stiffness[0]
    eigenvalues = np.linalg.eigvalsh(matrix)
    report = {
        "shape": list(matrix.shape),
        "symmetry_residual": float(
            np.linalg.norm(matrix - matrix.T) / np.linalg.norm(matrix)
        ),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "positive_eigenvalues": int(np.count_nonzero(eigenvalues > 1.0e-7 * eigenvalues[-1])),
        "timing_seconds": timing,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

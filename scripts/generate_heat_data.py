from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bnem.heat_fem import port_schur_matrix

def sample_vertices(rng: np.random.Generator) -> list[list[float]]:
    width = float(rng.uniform(0.75, 1.45))
    height = float(rng.uniform(0.75, 1.45))

    for _attempt in range(100):
        bottom_tilt = float(rng.uniform(-0.16, 0.18))
        right_top_x = width + float(rng.uniform(-0.24, 0.24))
        right_top_y = height + float(rng.uniform(-0.18, 0.18))
        left_top_x = float(rng.uniform(-0.28, 0.18))
        left_top_y = height + float(rng.uniform(-0.18, 0.18))
        vertices = [
            [0.0, 0.0],
            [width, bottom_tilt],
            [right_top_x, right_top_y],
            [left_top_x, left_top_y],
        ]
        points = np.asarray(vertices, dtype=np.float64)
        shifted = np.roll(points, -1, axis=0)
        area = 0.5 * float(np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1]))
        edge_lengths = np.linalg.norm(shifted - points, axis=1)
        if area > 0.25 and np.min(edge_lengths) > 0.35:
            return vertices
    raise RuntimeError("failed to sample a nondegenerate skew quadrilateral")

def sample_edge_bulges(rng: np.random.Generator, *, curved_fraction: float) -> list[float]:
    if float(rng.random()) > curved_fraction:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(v) for v in rng.uniform(-0.045, 0.045, size=4)]

def write_sample(
    out_dir: Path,
    *,
    role: str,
    index: int,
    nx: int,
    ny: int,
    vertices: list[list[float]],
    edge_bulges: list[float],
    conductivity: float,
    modes_per_edge: int,
    bubbles_per_edge: int,
) -> Path:
    matrix = port_schur_matrix(
        nx=nx,
        ny=ny,
        vertices=vertices,
        edge_bulges=edge_bulges,
        conductivity=conductivity,
        modes_per_edge=modes_per_edge,
        bubbles_per_edge=bubbles_per_edge,
    )
    role_dir = out_dir / role
    role_dir.mkdir(parents=True, exist_ok=True)
    path = role_dir / f"temperature_btsk_python_{role}_{index:04d}.npz"
    np.savez_compressed(
        path,
        eta_vertices=np.asarray(vertices, dtype=np.float64),
        eta_edge_bulges=np.asarray(edge_bulges, dtype=np.float64),
        eta_edge_types=np.zeros(4, dtype=np.int64),
        eta_k=np.array(conductivity, dtype=np.float64),
        S=matrix,
        S_raw=matrix,
        h=np.zeros(matrix.shape[0], dtype=np.float64),
        c=np.array(0.0, dtype=np.float64),
        basis_name=np.array("legendre"),
        port_scheme=np.array("vertex_bubble"),
        modes_per_edge=np.array(modes_per_edge, dtype=np.int64),
        bubbles_per_edge=np.array(bubbles_per_edge, dtype=np.int64),
        edges_per_patch=np.array(4, dtype=np.int64),
        physics=np.array("scalar_heat_conduction"),
        generator=np.array("self_contained_q4_fem_schur"),
    )
    diagnostics = {
        "path": str(path),
        "role": role,
        "nx": nx,
        "ny": ny,
        "vertices": vertices,
        "edge_bulges": edge_bulges,
        "conductivity": conductivity,
        "matrix_norm": float(np.linalg.norm(matrix)),
        "symmetry_error": float(np.linalg.norm(matrix - matrix.T) / (np.linalg.norm(matrix) + 1.0e-12)),
        "min_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
    }
    path.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return path

def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Q4 FEM heat-operator training bank.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "temperature_btsk_python_bank")
    parser.add_argument("--train-count", type=int, default=60)
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=12)
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--modes-per-edge", type=int, default=4)
    parser.add_argument("--bubbles-per-edge", type=int, default=2)
    parser.add_argument("--curved-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists() and not args.force:
        existing = list(args.out_dir.rglob("*.npz"))
        if existing:
            raise FileExistsError(f"{args.out_dir} already contains {len(existing)} npz files; pass --force")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    counts = {"train": args.train_count, "validation": args.validation_count, "test": args.test_count}
    paths: list[str] = []
    for role, count in counts.items():
        for index in range(count):
            vertices = sample_vertices(rng)
            edge_bulges = sample_edge_bulges(rng, curved_fraction=args.curved_fraction)
            conductivity = float(rng.uniform(0.7, 1.4))
            path = write_sample(
                args.out_dir,
                role=role,
                index=index,
                nx=args.nx,
                ny=args.ny,
                vertices=vertices,
                edge_bulges=edge_bulges,
                conductivity=conductivity,
                modes_per_edge=args.modes_per_edge,
                bubbles_per_edge=args.bubbles_per_edge,
            )
            paths.append(str(path))
    manifest = {
        "status": "self_contained_q4_fem_bank",
        "counts": counts,
        "nx": args.nx,
        "ny": args.ny,
        "modes_per_edge": args.modes_per_edge,
        "bubbles_per_edge": args.bubbles_per_edge,
        "curved_fraction": args.curved_fraction,
        "seed": args.seed,
        "paths": paths,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(json.dumps({k: len(list((args.out_dir / k).glob('*.npz'))) for k in counts}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

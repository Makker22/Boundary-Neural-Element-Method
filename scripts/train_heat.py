from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGED_DATASET = ROOT / "Data/training/heat.npz"
DEFAULT_DATASET = (
    PACKAGED_DATASET
    if PACKAGED_DATASET.is_file()
    else ROOT / "data/quad_heat_affine_v1.npz"
)

from src.bnem.heat_fem import port_schur_matrix
from src.bnem.quad_affine_model import AffineComplementCholeskyNet
from src.bnem.quad_canonical import (
    canonical_quad_features_from_canonical,
    canonicalize_quad,
)
from src.bnem.quad_heat_affine import (
    heat_affine_complement_projector,
    heat_affine_consistency_operator,
)
from src.bnem.quad_mesh_quality import assess_quad_quality
from src.bnem.quad_partition_certification import CERTIFIED_QUAD_SHAPE_GATE

def sample_certified_quad(rng: np.random.Generator) -> np.ndarray:
    for _ in range(1000):
        angles = np.asarray([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi])
        angles += rng.uniform(-0.28, 0.28, size=4)
        angles.sort()
        radius = np.exp(rng.uniform(math.log(0.72), math.log(1.35), size=4))
        points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        anisotropy = float(np.exp(rng.uniform(math.log(0.65), math.log(1.55))))
        shear = float(rng.uniform(-0.38, 0.38))
        transform = np.asarray([[anisotropy, shear], [0.0, 1.0 / anisotropy]])
        points = points @ transform.T
        area = 0.5 * float(
            np.sum(
                points[:, 0] * np.roll(points[:, 1], -1)
                - np.roll(points[:, 0], -1) * points[:, 1]
            )
        )
        if area < 0.0:
            points = points[::-1]
        quality = assess_quad_quality(
            points, gate=CERTIFIED_QUAD_SHAPE_GATE, mode="strict"
        )
        if quality.accepted:
            return points
    raise RuntimeError("failed to sample a certified quadrilateral")

def consistent_target(vertices: np.ndarray, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    affine, _trace, _forces = heat_affine_consistency_operator(vertices)
    projector, _mass, _trace = heat_affine_complement_projector(vertices)
    raw = port_schur_matrix(
        nx=subdivisions,
        ny=subdivisions,
        vertices=vertices.tolist(),
        edge_bulges=[0.0] * 4,
        conductivity=1.0,
        modes_per_edge=4,
        bubbles_per_edge=2,
    )
    residual = 0.5 * ((raw - affine) + (raw - affine).T)
    values, vectors = np.linalg.eigh(residual)
    keep = values > 1.0e-10 * max(float(np.max(np.abs(values))), 1.0)
    raw_factor = vectors[:, keep] * np.sqrt(values[keep])[None, :]
    projected = projector.T @ raw_factor
    target = affine + projected @ projected.T
    return 0.5 * (target + target.T), raw_factor

def build_dataset(path: Path, *, count: int, subdivisions: int, seed: int) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    rng = np.random.default_rng(seed)
    vertices = np.empty((count, 4, 2), dtype=np.float64)
    features = np.empty((count, 22), dtype=np.float32)
    affine = np.empty((count, 12, 12), dtype=np.float32)
    projectors = np.empty((count, 12, 12), dtype=np.float32)
    targets = np.empty((count, 12, 12), dtype=np.float32)
    base_factors: list[np.ndarray] = []
    started = time.perf_counter()
    for index in range(count):
        raw = sample_certified_quad(rng)
        canonical = canonicalize_quad(raw, bubbles_per_edge=2).vertices
        target, raw_factor = consistent_target(canonical, subdivisions)
        consistency, _trace, _forces = heat_affine_consistency_operator(canonical)
        projector, _mass, _trace = heat_affine_complement_projector(canonical)
        vertices[index] = canonical
        features[index] = canonical_quad_features_from_canonical(canonical, poisson=0.0)
        affine[index] = consistency
        projectors[index] = projector
        targets[index] = target
        base_factors.append(raw_factor)
        if (index + 1) % 500 == 0:
            print(json.dumps({"generated": index + 1, "seconds": time.perf_counter() - started}))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vertices=vertices,
        features=features,
        affine=affine,
        projectors=projectors,
        targets=targets,
    )
    return {
        "vertices": vertices,
        "features": features,
        "affine": affine,
        "projectors": projectors,
        "targets": targets,
    }

def relative_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = prediction - target
    matrix = torch.linalg.matrix_norm(difference) / torch.linalg.matrix_norm(target).clamp_min(1.0e-12)
    generator = torch.Generator(device=prediction.device).manual_seed(20260712)
    probes = torch.randn(
        target.shape[0], target.shape[1], 16, generator=generator, device=prediction.device
    )
    expected = target @ probes
    actual = prediction @ probes
    action = torch.linalg.vector_norm(actual - expected, dim=1) / torch.linalg.vector_norm(
        expected, dim=1
    ).clamp_min(1.0e-12)
    energy_expected = torch.sum(probes * expected, dim=1)
    energy_actual = torch.sum(probes * actual, dim=1)
    energy = torch.abs(energy_actual - energy_expected) / torch.abs(energy_expected).clamp_min(1.0e-12)
    return {
        "matrix_p95": float(torch.quantile(matrix, 0.95).cpu()),
        "action_p95": float(torch.quantile(action.reshape(-1), 0.95).cpu()),
        "energy_p95": float(torch.quantile(energy.reshape(-1), 0.95).cpu()),
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/models_quad_heat_affine_v1")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--train-count", type=int, default=3200)
    parser.add_argument("--validation-count", type=int, default=400)
    parser.add_argument("--subdivisions", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-7)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = build_dataset(
        args.dataset,
        count=args.count,
        subdivisions=args.subdivisions,
        seed=args.seed,
    )
    if args.train_count + args.validation_count >= args.count:
        raise ValueError("split leaves no test samples")
    device = args.device
    features_np = data["features"]
    feature_mean = torch.as_tensor(features_np[: args.train_count].mean(axis=0), dtype=torch.float32)
    feature_scale = torch.as_tensor(features_np[: args.train_count].std(axis=0) + 1.0e-6, dtype=torch.float32)

    square = canonicalize_quad(
        np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
        bubbles_per_edge=2,
    ).vertices
    _target, base_raw = consistent_target(square, args.subdivisions)
    base_factor = np.zeros((12, 9), dtype=np.float32)
    base_factor[:, : min(9, base_raw.shape[1])] = base_raw[:, :9]
    config = {
        "feature_dim": 22,
        "port_dofs": 12,
        "complement_rank": 9,
        "hidden_dim": 192,
        "hidden_layers": 4,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "base_factor": torch.as_tensor(base_factor),
        "correction_scale": 0.5,
    }
    model = AffineComplementCholeskyNet(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    tensors = {
        key: torch.as_tensor(data[key], dtype=torch.float32, device=device)
        for key in ("features", "affine", "projectors", "targets")
    }
    train = np.arange(args.train_count)
    validation = np.arange(args.train_count, args.train_count + args.validation_count)
    test = np.arange(args.train_count + args.validation_count, args.count)
    best = float("inf")
    best_state = None
    history = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(train)
        model.train()
        for first in range(0, len(train), args.batch_size):
            indices = torch.as_tensor(train[first : first + args.batch_size], device=device)
            prediction = model(
                tensors["features"][indices],
                tensors["affine"][indices],
                tensors["projectors"][indices],
            )
            target = tensors["targets"][indices]
            difference = prediction - target
            matrix_loss = torch.mean(
                torch.linalg.matrix_norm(difference) ** 2
                / torch.linalg.matrix_norm(target).clamp_min(1.0e-8) ** 2
            )
            probes = torch.randn(len(indices), 12, 12, device=device)
            expected = target @ probes
            actual = prediction @ probes
            action_loss = torch.mean(
                torch.linalg.vector_norm(actual - expected, dim=1) ** 2
                / torch.linalg.vector_norm(expected, dim=1).clamp_min(1.0e-8) ** 2
            )
            loss = matrix_loss + action_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            idx = torch.as_tensor(validation, device=device)
            val_prediction = model(
                tensors["features"][idx], tensors["affine"][idx], tensors["projectors"][idx]
            )
            val_target = tensors["targets"][idx]
            val_loss = float(
                torch.mean(
                    torch.linalg.matrix_norm(val_prediction - val_target) ** 2
                    / torch.linalg.matrix_norm(val_target).clamp_min(1.0e-8) ** 2
                ).cpu()
            )
        if val_loss < best:
            best = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % 25 == 0 or epoch + 1 == args.epochs:
            row = {"epoch": epoch, "validation_loss": val_loss}
            history.append(row)
            print(json.dumps(row))
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    metrics = {}
    with torch.no_grad():
        for name, indices_np in (("validation", validation), ("test", test)):
            indices = torch.as_tensor(indices_np, device=device)
            prediction = model(
                tensors["features"][indices],
                tensors["affine"][indices],
                tensors["projectors"][indices],
            )
            metrics[name] = relative_metrics(prediction, tensors["targets"][indices])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "quad_heat_affine.pt"
    serializable_config = {
        key: value for key, value in config.items() if key not in {"feature_mean", "feature_scale", "base_factor"}
    }
    serializable_config.update(
        {
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "base_factor": torch.as_tensor(base_factor),
        }
    )
    torch.save(
        {
            "model_kind": "HeatAffineComplementCholeskyNet",
            "model_config": serializable_config,
            "model_state_dict": best_state,
            "metrics": metrics,
            "training": vars(args),
        },
        checkpoint,
    )
    report = {"checkpoint": str(checkpoint), "best_validation_loss": best, "metrics": metrics, "history": history}
    (args.out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

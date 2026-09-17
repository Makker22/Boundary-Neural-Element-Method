

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bnem.compact_vertex_rank2 import (
    CompactVertexRank2Net,
    VERTEX_INDICES_16,
    direct_vertex_affine_operator,
    parameters_from_coefficients,
    scalar_affine_null,
)
from src.bnem.quad_affine_operator import _batch_features
from src.bnem.quad_canonical import canonicalize_quad_batch_arrays


DEFAULT_BANK = ROOT / "Data/training/elasticity_compact_labels.npz"
DEFAULT_OUT = ROOT / "runs/elasticity_retrain"


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def prepare_examples(
    vertices: np.ndarray,
    poisson: np.ndarray,
    matrices: np.ndarray,
) -> dict[str, np.ndarray | float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    poisson = np.asarray(poisson, dtype=np.float64)
    matrices = np.asarray(matrices, dtype=np.float64)
    if vertices.shape != (len(poisson), 4, 2):
        raise ValueError("vertices and poisson shapes do not match")
    if matrices.shape != (len(poisson), 16, 16):
        raise ValueError("matrices must have shape (sample_count,16,16)")

    canonical = canonicalize_quad_batch_arrays(
        vertices, bubbles_per_edge=1, build_port_transforms=True
    )
    if canonical.port_transforms is None:
        raise RuntimeError("canonical transforms were not built")
    transform = canonical.port_transforms
    target_full = transform @ matrices @ np.swapaxes(transform, 1, 2)
    target_full = 0.5 * (target_full + np.swapaxes(target_full, 1, 2))
    target_vertex = target_full[:, VERTEX_INDICES_16][:, :, VERTEX_INDICES_16]
    affine = direct_vertex_affine_operator(canonical.vertices, poisson)
    residual = 0.5 * (
        target_vertex - affine + np.swapaxes(target_vertex - affine, 1, 2)
    )
    null = scalar_affine_null(canonical.vertices)
    coefficient = np.empty((len(vertices), 2, 2), dtype=np.float64)
    coefficient[:, 0, 0] = np.einsum(
        "ni,nij,nj->n", null, residual[:, :4, :4], null
    )
    coefficient[:, 0, 1] = np.einsum(
        "ni,nij,nj->n", null, residual[:, :4, 4:], null
    )
    coefficient[:, 1, 0] = np.einsum(
        "ni,nij,nj->n", null, residual[:, 4:, :4], null
    )
    coefficient[:, 1, 1] = np.einsum(
        "ni,nij,nj->n", null, residual[:, 4:, 4:], null
    )
    coefficient = 0.5 * (coefficient + np.swapaxes(coefficient, 1, 2))

    outer = null[:, :, None] * null[:, None, :]
    reconstructed = affine.copy()
    reconstructed[:, :4, :4] += coefficient[:, 0, 0, None, None] * outer
    reconstructed[:, :4, 4:] += coefficient[:, 0, 1, None, None] * outer
    reconstructed[:, 4:, :4] += coefficient[:, 1, 0, None, None] * outer
    reconstructed[:, 4:, 4:] += coefficient[:, 1, 1, None, None] * outer
    reconstruction_relative = np.linalg.norm(
        reconstructed - target_vertex, axis=(1, 2)
    ) / np.maximum(np.linalg.norm(target_vertex, axis=(1, 2)), 1.0e-14)
    minimum_eigenvalue = np.linalg.eigvalsh(coefficient)[:, 0]
    if float(np.max(reconstruction_relative)) > 1.0e-8:
        raise ValueError("rank-two decomposition did not reconstruct vertex labels")
    if float(np.min(minimum_eigenvalue)) <= 0.0:
        raise ValueError("vertex complement labels are not positive definite")
    return {
        "features": _batch_features(canonical.vertices, poisson).astype(np.float32),
        "parameters": parameters_from_coefficients(coefficient).astype(np.float32),
        "coefficients": coefficient.astype(np.float32),
        "target_vertex": target_vertex.astype(np.float32),
        "reconstruction_relative_max": float(np.max(reconstruction_relative)),
        "coefficient_minimum_eigenvalue": float(np.min(minimum_eigenvalue)),
    }


def load_synthetic(path: Path) -> tuple[dict[str, np.ndarray | float], np.ndarray]:
    with np.load(path) as data:
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        poisson = np.asarray(data["poisson"], dtype=np.float64)
        matrices = np.asarray(data["stiffness_unit_E"], dtype=np.float64)
        roles = np.asarray(data["roles"]).astype(str)
        convergence = np.asarray(data["relative_n12_n16"], dtype=np.float64)
    if np.any(convergence > 0.002):
        raise ValueError("training labels do not pass the convergence gate")
    material_count = len(poisson)
    examples = prepare_examples(
        np.repeat(vertices, material_count, axis=0),
        np.tile(poisson, len(vertices)),
        matrices.reshape(-1, 16, 16),
    )
    return examples, np.repeat(roles, material_count)


def load_external(root: Path) -> dict[str, np.ndarray | float]:
    vertices: list[np.ndarray] = []
    poisson: list[float] = []
    matrices: list[np.ndarray] = []
    for path in sorted(root.glob("*.npz")):
        with np.load(path) as data:
            vertices.append(np.asarray(data["eta_vertices"], dtype=np.float64))
            poisson.append(float(data["eta_nu"]))
            scale = float(data["eta_E"]) * float(data["eta_thickness"])
            matrices.append(np.asarray(data["S"], dtype=np.float64) / scale)
    if not vertices:
        raise ValueError(f"no external samples found in {root}")
    return prepare_examples(
        np.stack(vertices), np.asarray(poisson), np.stack(matrices)
    )


def torch_coefficients(parameters: torch.Tensor) -> torch.Tensor:
    l00 = torch.exp(torch.clamp(parameters[:, 0], -30.0, 30.0))
    l10 = parameters[:, 1]
    l11 = torch.exp(torch.clamp(parameters[:, 2], -30.0, 30.0))
    result = torch.empty(
        (len(parameters), 2, 2), dtype=parameters.dtype, device=parameters.device
    )
    result[:, 0, 0] = l00 * l00
    result[:, 0, 1] = l00 * l10
    result[:, 1, 0] = result[:, 0, 1]
    result[:, 1, 1] = l10 * l10 + l11 * l11
    return result


def metrics(
    model: CompactVertexRank2Net,
    examples: dict[str, np.ndarray | float],
    indices: np.ndarray,
    *,
    device: str,
) -> dict[str, float]:
    features = torch.as_tensor(
        np.asarray(examples["features"])[indices], dtype=torch.float32, device=device
    )
    target_parameter = torch.as_tensor(
        np.asarray(examples["parameters"])[indices], dtype=torch.float32, device=device
    )
    target_coefficient = torch.as_tensor(
        np.asarray(examples["coefficients"])[indices], dtype=torch.float32, device=device
    )
    target_vertex = torch.as_tensor(
        np.asarray(examples["target_vertex"])[indices], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        prediction = model(features)
        coefficient = torch_coefficients(prediction)
        coefficient_error = torch.linalg.matrix_norm(
            coefficient - target_coefficient
        ) / torch.clamp(torch.linalg.matrix_norm(target_coefficient), min=1.0e-12)


        vertex_matrix_error = torch.linalg.matrix_norm(
            coefficient - target_coefficient
        ) / torch.clamp(torch.linalg.matrix_norm(target_vertex), min=1.0e-12)
        standardized_error = torch.linalg.vector_norm(
            (prediction - target_parameter) / model.parameter_scale, dim=1
        )
    def summary(values: torch.Tensor, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_median": float(torch.quantile(values, 0.5).cpu()),
            f"{prefix}_p95": float(torch.quantile(values, 0.95).cpu()),
            f"{prefix}_maximum": float(torch.max(values).cpu()),
        }
    return {
        "count": int(len(indices)),
        **summary(coefficient_error, "coefficient_relative"),
        **summary(vertex_matrix_error, "vertex_matrix_relative"),
        **summary(standardized_error, "standardized_parameter_error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument(
        "--external",
        type=Path,
        help="Optional directory containing the independent 512-sample mesh hold-out.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    examples, roles = load_synthetic(args.bank.resolve())
    external = load_external(args.external.resolve()) if args.external else None
    indices = {
        role: np.flatnonzero(roles == role)
        for role in ("train", "validation", "test")
    }
    features = np.asarray(examples["features"], dtype=np.float32)
    parameters = np.asarray(examples["parameters"], dtype=np.float32)
    coefficients = np.asarray(examples["coefficients"], dtype=np.float32)
    train = indices["train"]
    feature_mean = features[train].mean(axis=0)
    feature_scale = features[train].std(axis=0)
    feature_scale[feature_scale < 1.0e-6] = 1.0
    parameter_mean = parameters[train].mean(axis=0)
    parameter_scale = parameters[train].std(axis=0)
    parameter_scale[parameter_scale < 1.0e-6] = 1.0
    model = CompactVertexRank2Net(
        feature_mean=torch.as_tensor(feature_mean),
        feature_scale=torch.as_tensor(feature_scale),
        parameter_mean=torch.as_tensor(parameter_mean),
        parameter_scale=torch.as_tensor(parameter_scale),
        hidden_dim=int(args.hidden_dim),
        hidden_layers=int(args.hidden_layers),
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(args.epochs), 1)
    )
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    parameter_tensor = torch.as_tensor(parameters, dtype=torch.float32)
    coefficient_tensor = torch.as_tensor(coefficients, dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(args.seed))
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = float("inf")
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        permutation = train[torch.randperm(len(train), generator=generator).numpy()]
        losses: list[float] = []
        for first in range(0, len(permutation), int(args.batch_size)):
            batch = permutation[first : first + int(args.batch_size)]
            batch_features = feature_tensor[batch].to(args.device)
            batch_parameters = parameter_tensor[batch].to(args.device)
            batch_coefficients = coefficient_tensor[batch].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_features)
            standardized = (prediction - batch_parameters) / model.parameter_scale
            parameter_loss = torch.mean(standardized * standardized)
            predicted_coefficients = torch_coefficients(prediction)
            coefficient_loss = torch.mean(
                torch.sum((predicted_coefficients - batch_coefficients) ** 2, dim=(1, 2))
                / torch.clamp(
                    torch.sum(batch_coefficients * batch_coefficients, dim=(1, 2)),
                    min=1.0e-12,
                )
            )
            loss = parameter_loss + 0.5 * coefficient_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.epochs):
            model.eval()
            validation = metrics(
                model, examples, indices["validation"], device=args.device
            )
            score = validation["vertex_matrix_relative_p95"]
            row = {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)),
                "validation_vertex_matrix_relative_p95": float(score),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            if score < best_score:
                best_score = float(score)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
            print(json.dumps(row), flush=True)

    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    evaluation = {
        role: metrics(model, examples, role_indices, device=args.device)
        for role, role_indices in indices.items()
    }
    external_indices = np.empty(0, dtype=np.int64)
    if external is not None:
        external_indices = np.arange(len(np.asarray(external["features"])))
        evaluation["external_ogrid_holdout"] = metrics(
            model, external, external_indices, device=args.device
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "compact_vertex_rank2.pt"
    torch.save(
        {
            "schema": "bnem.compact_vertex_rank2.v1",
            "model_kind": "CompactVertexRank2Net",
            "model_config": {
                "feature_dim": 22,
                "output_parameters": 3,
                "hidden_dim": int(args.hidden_dim),
                "hidden_layers": int(args.hidden_layers),
            },
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "parameter_mean": parameter_mean,
            "parameter_scale": parameter_scale,
            "model_state_dict": model.state_dict(),
            "training_bank": portable_path(args.bank),
            "external_used_for_selection": False,
            "best_epoch": best_epoch,
        },
        checkpoint,
    )
    report = {
        "schema": "bnem.compact_vertex_rank2.training.v1",
        "checkpoint": portable_path(checkpoint),
        "training_bank": portable_path(args.bank),
        "external_holdout": portable_path(args.external) if args.external else None,
        "external_used_for_selection": False,
        "split_counts": {key: int(len(value)) for key, value in indices.items()},
        "external_count": int(len(external_indices)),
        "model_config": {
            "feature_dim": 22,
            "output_parameters": 3,
            "hidden_dim": int(args.hidden_dim),
            "hidden_layers": int(args.hidden_layers),
            "trainable_parameters": int(
                sum(value.numel() for value in model.parameters() if value.requires_grad)
            ),
        },
        "label_checks": {
            "rank_two_reconstruction_relative_max": examples[
                "reconstruction_relative_max"
            ],
            "coefficient_minimum_eigenvalue": examples[
                "coefficient_minimum_eigenvalue"
            ],
        },
        "device": str(args.device),
        "training_seconds": float(training_seconds),
        "best_epoch": int(best_epoch),
        "best_validation_vertex_matrix_relative_p95": float(best_score),
        "evaluation": evaluation,
        "history": history,
    }
    output = args.out_dir / "training_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event": "finished", "output": str(output), **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

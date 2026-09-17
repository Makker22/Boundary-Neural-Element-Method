






from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

try:
    from numba import njit
except ImportError:
    njit = None

from .quad_affine_operator import _batch_features, _broadcast
from .quad_canonical import canonicalize_quad_batch_arrays


VERTEX_INDICES_16 = np.asarray((0, 1, 2, 3, 8, 9, 10, 11), dtype=np.int64)


if njit is not None:

    @njit(cache=True)
    def _prepare_compact_cells_numba(
        cells: np.ndarray, poisson: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


        count = len(cells)
        features = np.empty((count, 22), dtype=np.float32)
        affine = np.empty((count, 8, 8), dtype=np.float64)
        null = np.empty((count, 4), dtype=np.float64)
        transforms = np.zeros((count, 8, 8), dtype=np.float64)
        starts = (0, 1, 2, 3)
        steps = (1, -1)
        natural_xi = (-1.0, 1.0, 1.0, -1.0)
        natural_eta = (-1.0, -1.0, 1.0, 1.0)
        for cell in range(count):
            centre_x = 0.0
            centre_y = 0.0
            for vertex in range(4):
                centre_x += cells[cell, vertex, 0]
                centre_y += cells[cell, vertex, 1]
            centre_x *= 0.25
            centre_y *= 0.25
            scale_squared = 0.0
            for vertex in range(4):
                dx = cells[cell, vertex, 0] - centre_x
                dy = cells[cell, vertex, 1] - centre_y
                scale_squared += dx * dx + dy * dy
            scale = np.sqrt(0.25 * scale_squared)

            best_vertices = np.empty((4, 2), dtype=np.float64)
            best_rotation = np.empty((2, 2), dtype=np.float64)
            best_indices = np.empty(4, dtype=np.int64)
            best_step = 2
            found = False
            for step_slot in range(2):
                step = steps[step_slot]
                for start_slot in range(4):
                    start = starts[start_slot]
                    indices = np.empty(4, dtype=np.int64)
                    for offset in range(4):
                        indices[offset] = (start + step * offset) % 4
                    first = indices[0]
                    second_index = indices[1]
                    direction_x = cells[cell, second_index, 0] - cells[cell, first, 0]
                    direction_y = cells[cell, second_index, 1] - cells[cell, first, 1]
                    length = np.sqrt(direction_x * direction_x + direction_y * direction_y)
                    if length <= 1.0e-14:
                        continue
                    tangent_x = direction_x / length
                    tangent_y = direction_y / length
                    signed_area = 0.0
                    for vertex in range(4):
                        following = (vertex + 1) % 4
                        i = indices[vertex]
                        j = indices[following]
                        signed_area += (
                            cells[cell, i, 0] * cells[cell, j, 1]
                            - cells[cell, j, 0] * cells[cell, i, 1]
                        )
                    if signed_area > 0.0:
                        second_x = -tangent_y
                        second_y = tangent_x
                    else:
                        second_x = tangent_y
                        second_y = -tangent_x
                    candidate = np.empty((4, 2), dtype=np.float64)
                    for vertex in range(4):
                        original = indices[vertex]
                        dx = cells[cell, original, 0] - centre_x
                        dy = cells[cell, original, 1] - centre_y
                        candidate[vertex, 0] = (
                            dx * tangent_x + dy * tangent_y
                        ) / scale
                        candidate[vertex, 1] = (
                            dx * second_x + dy * second_y
                        ) / scale

                    better = not found
                    equal = found
                    if found:
                        for vertex in range(4):
                            for component in range(2):
                                candidate_key = np.round(
                                    candidate[vertex, component] * 1.0e12
                                ) / 1.0e12
                                best_key = np.round(
                                    best_vertices[vertex, component] * 1.0e12
                                ) / 1.0e12
                                if candidate_key < best_key:
                                    better = True
                                    equal = False
                                    break
                                if candidate_key > best_key:
                                    better = False
                                    equal = False
                                    break
                            if not equal:
                                break
                        if equal:
                            if step < best_step:
                                better = True
                                equal = False
                            elif step > best_step:
                                better = False
                                equal = False
                        if equal:
                            for vertex in range(4):
                                if indices[vertex] < best_indices[vertex]:
                                    better = True
                                    break
                                if indices[vertex] > best_indices[vertex]:
                                    better = False
                                    break
                    if better:
                        found = True
                        best_step = step
                        for vertex in range(4):
                            best_indices[vertex] = indices[vertex]
                            best_vertices[vertex, 0] = candidate[vertex, 0]
                            best_vertices[vertex, 1] = candidate[vertex, 1]
                        best_rotation[0, 0] = tangent_x
                        best_rotation[0, 1] = tangent_y
                        best_rotation[1, 0] = second_x
                        best_rotation[1, 1] = second_y

            for canonical_vertex in range(4):
                original = best_indices[canonical_vertex]
                transforms[cell, canonical_vertex, original] = best_rotation[0, 0]
                transforms[cell, canonical_vertex, 4 + original] = best_rotation[0, 1]
                transforms[cell, 4 + canonical_vertex, original] = best_rotation[1, 0]
                transforms[cell, 4 + canonical_vertex, 4 + original] = best_rotation[1, 1]
                features[cell, 2 * canonical_vertex] = best_vertices[canonical_vertex, 0]
                features[cell, 2 * canonical_vertex + 1] = best_vertices[canonical_vertex, 1]

            edge_lengths = np.empty(4, dtype=np.float64)
            mean_length = 0.0
            for edge in range(4):
                following = (edge + 1) % 4
                dx = best_vertices[following, 0] - best_vertices[edge, 0]
                dy = best_vertices[following, 1] - best_vertices[edge, 1]
                edge_lengths[edge] = np.sqrt(dx * dx + dy * dy)
                mean_length += edge_lengths[edge]
            mean_length *= 0.25
            for edge in range(4):
                features[cell, 8 + edge] = edge_lengths[edge] / mean_length

            determinants = np.empty(4, dtype=np.float64)
            scaled_determinants = np.empty(4, dtype=np.float64)
            mean_determinant = 0.0
            for corner in range(4):
                xi = natural_xi[corner]
                eta = natural_eta[corner]
                dx_dxi = 0.0
                dy_dxi = 0.0
                dx_deta = 0.0
                dy_deta = 0.0
                for vertex in range(4):
                    if vertex == 0:
                        dxi = -0.25 * (1.0 - eta)
                        deta = -0.25 * (1.0 - xi)
                    elif vertex == 1:
                        dxi = 0.25 * (1.0 - eta)
                        deta = -0.25 * (1.0 + xi)
                    elif vertex == 2:
                        dxi = 0.25 * (1.0 + eta)
                        deta = 0.25 * (1.0 + xi)
                    else:
                        dxi = -0.25 * (1.0 + eta)
                        deta = 0.25 * (1.0 - xi)
                    dx_dxi += dxi * best_vertices[vertex, 0]
                    dy_dxi += dxi * best_vertices[vertex, 1]
                    dx_deta += deta * best_vertices[vertex, 0]
                    dy_deta += deta * best_vertices[vertex, 1]
                determinant = dx_dxi * dy_deta - dy_dxi * dx_deta
                determinants[corner] = determinant
                scaled_determinants[corner] = determinant / np.sqrt(
                    (dx_dxi * dx_dxi + dy_dxi * dy_dxi)
                    * (dx_deta * dx_deta + dy_deta * dy_deta)
                )
                mean_determinant += determinant
            mean_determinant *= 0.25
            for corner in range(4):
                features[cell, 12 + corner] = determinants[corner] / mean_determinant
                features[cell, 16 + corner] = scaled_determinants[corner]
            nu = poisson[cell]
            features[cell, 20] = nu
            features[cell, 21] = nu / (1.0 - 2.0 * nu)

            constitutive = np.zeros((3, 3), dtype=np.float64)
            denominator = (1.0 + nu) * (1.0 - 2.0 * nu)
            constitutive[0, 0] = (1.0 - nu) / denominator
            constitutive[0, 1] = nu / denominator
            constitutive[1, 0] = nu / denominator
            constitutive[1, 1] = (1.0 - nu) / denominator
            constitutive[2, 2] = 0.5 * (1.0 - 2.0 * nu) / denominator
            forces = np.zeros((8, 3), dtype=np.float64)
            for edge in range(4):
                following = (edge + 1) % 4
                direction_x = best_vertices[following, 0] - best_vertices[edge, 0]
                direction_y = best_vertices[following, 1] - best_vertices[edge, 1]
                length = np.sqrt(direction_x * direction_x + direction_y * direction_y)
                normal_x = direction_y / length
                normal_y = -direction_x / length
                for component in range(3):
                    traction_x = (
                        constitutive[0, component] * normal_x
                        + constitutive[2, component] * normal_y
                    )
                    traction_y = (
                        constitutive[2, component] * normal_x
                        + constitutive[1, component] * normal_y
                    )
                    contribution_x = 0.5 * length * traction_x
                    contribution_y = 0.5 * length * traction_y
                    forces[edge, component] += contribution_x
                    forces[following, component] += contribution_x
                    forces[4 + edge, component] += contribution_y
                    forces[4 + following, component] += contribution_y
            area = 0.0
            for vertex in range(4):
                following = (vertex + 1) % 4
                area += (
                    best_vertices[vertex, 0] * best_vertices[following, 1]
                    - best_vertices[following, 0] * best_vertices[vertex, 1]
                )
            area *= 0.5
            inverse_energy = np.linalg.inv(area * constitutive)
            for row in range(8):
                for column in range(8):
                    value = 0.0
                    for first_mode in range(3):
                        for second_mode in range(3):
                            value += (
                                forces[row, first_mode]
                                * inverse_energy[first_mode, second_mode]
                                * forces[column, second_mode]
                            )
                    affine[cell, row, column] = value

            for removed in range(4):
                kept = np.empty(3, dtype=np.int64)
                slot = 0
                for vertex in range(4):
                    if vertex != removed:
                        kept[slot] = vertex
                        slot += 1
                ax = best_vertices[kept[0], 0]
                ay = best_vertices[kept[0], 1]
                bx = best_vertices[kept[1], 0]
                by = best_vertices[kept[1], 1]
                cx = best_vertices[kept[2], 0]
                cy = best_vertices[kept[2], 1]
                determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                null[cell, removed] = determinant if removed % 2 == 0 else -determinant
            null_norm = 0.0
            alternating_dot = 0.0
            for vertex in range(4):
                null_norm += null[cell, vertex] * null[cell, vertex]
                alternating_dot += null[cell, vertex] * (1.0 if vertex % 2 == 0 else -1.0)
            null_norm = np.sqrt(null_norm)
            null_sign = -1.0 if alternating_dot < 0.0 else 1.0
            for vertex in range(4):
                null[cell, vertex] *= null_sign / null_norm
        return features, affine, null, transforms

else:
    _prepare_compact_cells_numba = None


def scalar_affine_null(canonical_vertices: np.ndarray) -> np.ndarray:


    vertices = np.asarray(canonical_vertices, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 2):
        raise ValueError("canonical_vertices must have shape (cell_count,4,2)")
    trace = np.concatenate(
        (np.ones((len(vertices), 4, 1), dtype=np.float64), vertices), axis=2
    )
    null = np.empty((len(vertices), 4), dtype=np.float64)
    for removed in range(4):
        rows = np.asarray([index for index in range(4) if index != removed])
        tri = trace[:, rows]
        determinant = (
            tri[:, 0, 0]
            * (tri[:, 1, 1] * tri[:, 2, 2] - tri[:, 1, 2] * tri[:, 2, 1])
            - tri[:, 0, 1]
            * (tri[:, 1, 0] * tri[:, 2, 2] - tri[:, 1, 2] * tri[:, 2, 0])
            + tri[:, 0, 2]
            * (tri[:, 1, 0] * tri[:, 2, 1] - tri[:, 1, 1] * tri[:, 2, 0])
        )
        null[:, removed] = (-1.0) ** removed * determinant
    norm = np.linalg.norm(null, axis=1)
    if np.any(norm <= 1.0e-14):
        raise ValueError("quadrilateral affine trace is rank deficient")
    null /= norm[:, None]
    alternating = np.asarray((1.0, -1.0, 1.0, -1.0), dtype=np.float64)
    sign = np.where(null @ alternating < 0.0, -1.0, 1.0)
    return null * sign[:, None]


def direct_vertex_affine_operator(
    canonical_vertices: np.ndarray,
    poisson: np.ndarray,
) -> np.ndarray:


    vertices = np.asarray(canonical_vertices, dtype=np.float64)
    nu = np.asarray(poisson, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 2):
        raise ValueError("canonical_vertices must have shape (cell_count,4,2)")
    if nu.shape != (len(vertices),):
        raise ValueError("poisson must have one value per cell")

    constitutive = np.zeros((len(vertices), 3, 3), dtype=np.float64)
    denominator = (1.0 + nu) * (1.0 - 2.0 * nu)
    constitutive[:, 0, 0] = (1.0 - nu) / denominator
    constitutive[:, 0, 1] = nu / denominator
    constitutive[:, 1, 0] = nu / denominator
    constitutive[:, 1, 1] = (1.0 - nu) / denominator
    constitutive[:, 2, 2] = 0.5 * (1.0 - 2.0 * nu) / denominator

    forces = np.zeros((len(vertices), 8, 3), dtype=np.float64)
    for edge in range(4):
        following = (edge + 1) % 4
        direction = vertices[:, following] - vertices[:, edge]
        length = np.linalg.norm(direction, axis=1)
        normal = np.column_stack((direction[:, 1], -direction[:, 0])) / length[:, None]
        traction_x = (
            constitutive[:, 0] * normal[:, 0, None]
            + constitutive[:, 2] * normal[:, 1, None]
        )
        traction_y = (
            constitutive[:, 2] * normal[:, 0, None]
            + constitutive[:, 1] * normal[:, 1, None]
        )
        endpoint_integral = 0.5 * length[:, None]
        forces[:, edge] += endpoint_integral * traction_x
        forces[:, following] += endpoint_integral * traction_x
        forces[:, 4 + edge] += endpoint_integral * traction_y
        forces[:, 4 + following] += endpoint_integral * traction_y

    shifted = np.roll(vertices, -1, axis=1)
    area = 0.5 * np.sum(
        vertices[:, :, 0] * shifted[:, :, 1]
        - shifted[:, :, 0] * vertices[:, :, 1],
        axis=1,
    )
    if np.any(area <= 1.0e-14):
        raise ValueError("canonical quadrilateral must have positive area")
    inverse_energy = np.linalg.inv(area[:, None, None] * constitutive)
    affine = forces @ inverse_energy @ np.swapaxes(forces, 1, 2)
    return 0.5 * (affine + np.swapaxes(affine, 1, 2))


def coefficients_from_parameters(parameters: np.ndarray) -> np.ndarray:


    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("parameters must have shape (cell_count,3)")
    l00 = np.exp(np.clip(values[:, 0], -30.0, 30.0))
    l10 = values[:, 1]
    l11 = np.exp(np.clip(values[:, 2], -30.0, 30.0))
    coefficients = np.empty((len(values), 2, 2), dtype=np.float64)
    coefficients[:, 0, 0] = l00 * l00
    coefficients[:, 0, 1] = l00 * l10
    coefficients[:, 1, 0] = coefficients[:, 0, 1]
    coefficients[:, 1, 1] = l10 * l10 + l11 * l11
    return coefficients


def parameters_from_coefficients(coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("coefficients must have shape (cell_count,2,2)")
    cholesky = np.linalg.cholesky(
        0.5 * (values + np.swapaxes(values, 1, 2))
        + 1.0e-12 * np.eye(2)[None]
    )
    return np.column_stack(
        (np.log(cholesky[:, 0, 0]), cholesky[:, 1, 0], np.log(cholesky[:, 1, 1]))
    )


class CompactVertexRank2Net(nn.Module):
    def __init__(
        self,
        *,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        parameter_mean: torch.Tensor,
        parameter_scale: torch.Tensor,
        hidden_dim: int = 48,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.register_buffer("feature_mean", feature_mean.detach().float().clone())
        self.register_buffer("feature_scale", feature_scale.detach().float().clone())
        self.register_buffer("parameter_mean", parameter_mean.detach().float().clone())
        self.register_buffer("parameter_scale", parameter_scale.detach().float().clone())
        layers: list[nn.Module] = []
        width = int(self.feature_mean.numel())
        for _ in range(self.hidden_layers):
            layers.extend((nn.Linear(width, self.hidden_dim), nn.SiLU()))
            width = self.hidden_dim
        layers.append(nn.Linear(width, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        standardized = self.network(normalized)
        return self.parameter_mean + self.parameter_scale * standardized


@dataclass(frozen=True)
class NumpyCompactVertexRank2Model:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    parameter_mean: np.ndarray
    parameter_scale: np.ndarray
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]

    def __call__(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features, dtype=np.float32) - self.feature_mean) / self.feature_scale
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            values = values @ weight.T + bias
            if index + 1 < len(self.weights):
                values = values / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
        return self.parameter_mean + self.parameter_scale * values


@dataclass(frozen=True)
class CompactVertexRank2Operator:
    model: CompactVertexRank2Net | NumpyCompactVertexRank2Model
    device: str
    payload: dict[str, Any]


def load_compact_vertex_rank2_operator(
    checkpoint: str | Path,
    *,
    device: str = "numpy",
) -> CompactVertexRank2Operator:
    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload["model_config"])
    model = CompactVertexRank2Net(
        feature_mean=torch.as_tensor(payload["feature_mean"]),
        feature_scale=torch.as_tensor(payload["feature_scale"]),
        parameter_mean=torch.as_tensor(payload["parameter_mean"]),
        parameter_scale=torch.as_tensor(payload["parameter_scale"]),
        hidden_dim=int(config["hidden_dim"]),
        hidden_layers=int(config["hidden_layers"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    if device != "numpy":
        model = model.to(device)
        return CompactVertexRank2Operator(model=model, device=device, payload=payload)
    linear = [module for module in model.network if isinstance(module, nn.Linear)]
    numpy_model = NumpyCompactVertexRank2Model(
        feature_mean=model.feature_mean.detach().cpu().numpy(),
        feature_scale=model.feature_scale.detach().cpu().numpy(),
        parameter_mean=model.parameter_mean.detach().cpu().numpy(),
        parameter_scale=model.parameter_scale.detach().cpu().numpy(),
        weights=tuple(module.weight.detach().cpu().numpy() for module in linear),
        biases=tuple(module.bias.detach().cpu().numpy() for module in linear),
    )
    return CompactVertexRank2Operator(model=numpy_model, device="numpy", payload=payload)


def _infer_parameters(
    operator: CompactVertexRank2Operator,
    features: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    if operator.device == "numpy":
        assert isinstance(operator.model, NumpyCompactVertexRank2Model)
        for first in range(0, len(features), batch_size):
            chunks.append(operator.model(features[first : first + batch_size]))
        return np.concatenate(chunks).astype(np.float64)
    torch_device = torch.device(operator.device)
    assert isinstance(operator.model, CompactVertexRank2Net)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    with torch.no_grad():
        for first in range(0, len(features), batch_size):
            result = operator.model(
                torch.as_tensor(
                    features[first : first + batch_size],
                    dtype=torch.float32,
                    device=torch_device,
                )
            )
            chunks.append(result.detach().cpu().numpy())
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    return np.concatenate(chunks).astype(np.float64)


def predict_compact_vertex_stiffnesses(
    operator: CompactVertexRank2Operator,
    vertices: np.ndarray,
    *,
    poisson: float | Sequence[float] | np.ndarray,
    young: float | Sequence[float] | np.ndarray = 1.0,
    thickness: float | Sequence[float] | np.ndarray = 1.0,
    batch_size: int = 65_536,
) -> tuple[np.ndarray, dict[str, float]]:
    total_started = time.perf_counter()
    cells = np.asarray(vertices, dtype=np.float64)
    count = len(cells)
    nu = _broadcast(poisson, count, "poisson")
    young_values = _broadcast(young, count, "young")
    thickness_values = _broadcast(thickness, count, "thickness")

    stage = time.perf_counter()
    if _prepare_compact_cells_numba is not None:
        features, affine, null, transform = _prepare_compact_cells_numba(cells, nu)
        analytic_seconds = time.perf_counter() - stage
        canonical_seconds = analytic_seconds
        feature_seconds = 0.0
        affine_seconds = 0.0
        null_seconds = 0.0
        fused_analytic = True
    else:
        canonical = canonicalize_quad_batch_arrays(
            cells, bubbles_per_edge=0, build_port_transforms=True
        )
        if canonical.port_transforms is None:
            raise RuntimeError("vertex transform was not constructed")
        canonical_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        features = _batch_features(canonical.vertices, nu).astype(np.float32)
        feature_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        affine = direct_vertex_affine_operator(canonical.vertices, nu)
        affine_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        null = scalar_affine_null(canonical.vertices)
        null_seconds = time.perf_counter() - stage
        analytic_seconds = (
            canonical_seconds + feature_seconds + affine_seconds + null_seconds
        )
        transform = canonical.port_transforms
        fused_analytic = False

    stage = time.perf_counter()
    parameters = _infer_parameters(
        operator, features, batch_size=max(int(batch_size), 1)
    )
    mlp_seconds = time.perf_counter() - stage

    stage = time.perf_counter()
    coefficient = coefficients_from_parameters(parameters)
    outer = null[:, :, None] * null[:, None, :]
    canonical_matrix = affine.copy()
    canonical_matrix[:, :4, :4] += coefficient[:, 0, 0, None, None] * outer
    canonical_matrix[:, :4, 4:] += coefficient[:, 0, 1, None, None] * outer
    canonical_matrix[:, 4:, :4] += coefficient[:, 1, 0, None, None] * outer
    canonical_matrix[:, 4:, 4:] += coefficient[:, 1, 1, None, None] * outer
    matrices = np.swapaxes(transform, 1, 2) @ canonical_matrix @ transform
    matrices = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
    matrices *= (young_values * thickness_values)[:, None, None]
    restore_seconds = time.perf_counter() - stage
    return matrices, {
        "analytic_seconds": float(analytic_seconds),
        "canonical_seconds": float(canonical_seconds),
        "feature_seconds": float(feature_seconds),
        "affine_seconds": float(affine_seconds),
        "null_seconds": float(null_seconds),
        "fused_analytic": bool(fused_analytic),
        "mlp_seconds": float(mlp_seconds),
        "restore_seconds": float(restore_seconds),
        "total_seconds": float(time.perf_counter() - total_started),
    }


__all__ = [
    "CompactVertexRank2Net",
    "CompactVertexRank2Operator",
    "VERTEX_INDICES_16",
    "coefficients_from_parameters",
    "direct_vertex_affine_operator",
    "load_compact_vertex_rank2_operator",
    "parameters_from_coefficients",
    "predict_compact_vertex_stiffnesses",
    "scalar_affine_null",
]

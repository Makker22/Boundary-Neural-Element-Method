

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

from src.bnem.assembly import ConformingPolygonMeshAssembly
from src.bnem.quad_affine_model import AffineComplementCholeskyNet
from src.bnem.quad_canonical import (
    CanonicalQuad,
    canonicalize_quad_batch_arrays,
)

def _batch_features(canonical: np.ndarray, poisson: np.ndarray) -> np.ndarray:
    shifted = np.roll(canonical, -1, axis=1)
    edges = shifted - canonical
    lengths = np.linalg.norm(edges, axis=2)
    length_ratios = lengths / np.mean(lengths, axis=1, keepdims=True)
    natural = np.asarray(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    dxi = np.asarray(
        [0.25 * np.asarray([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)]) for xi, eta in natural]
    )
    deta = np.asarray(
        [0.25 * np.asarray([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi]) for xi, eta in natural]
    )
    first = np.einsum("ki,nij->nkj", dxi, canonical)
    second = np.einsum("ki,nij->nkj", deta, canonical)
    determinant = first[:, :, 0] * second[:, :, 1] - first[:, :, 1] * second[:, :, 0]
    determinant_ratio = determinant / np.mean(determinant, axis=1, keepdims=True)
    scaled = determinant / np.maximum(
        np.linalg.norm(first, axis=2) * np.linalg.norm(second, axis=2), 1.0e-14
    )
    lame_ratio = poisson / (1.0 - 2.0 * poisson)
    return np.concatenate(
        [
            canonical.reshape(len(canonical), -1),
            length_ratios,
            determinant_ratio,
            scaled,
            poisson[:, None],
            lame_ratio[:, None],
        ],
        axis=1,
    )

def _batch_affine_and_projection_data(
    canonical: np.ndarray, poisson: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(canonical)
    scalar_dofs = 8
    scalar_trace = np.zeros((count, scalar_dofs, 3), dtype=np.float64)
    scalar_trace[:, :4, 0] = 1.0
    scalar_trace[:, :4, 1] = canonical[:, :, 0]
    scalar_trace[:, :4, 2] = canonical[:, :, 1]

    constitutive = np.zeros((count, 3, 3), dtype=np.float64)
    denominator = (1.0 + poisson) * (1.0 - 2.0 * poisson)
    constitutive[:, 0, 0] = (1.0 - poisson) / denominator
    constitutive[:, 0, 1] = poisson / denominator
    constitutive[:, 1, 0] = poisson / denominator
    constitutive[:, 1, 1] = (1.0 - poisson) / denominator
    constitutive[:, 2, 2] = 0.5 * (1.0 - 2.0 * poisson) / denominator

    stress = constitutive
    forces = np.zeros((count, 16, 3), dtype=np.float64)
    edge_lengths = np.empty((count, 4), dtype=np.float64)
    for edge in range(4):
        direction = canonical[:, (edge + 1) % 4] - canonical[:, edge]
        length = np.linalg.norm(direction, axis=1)
        edge_lengths[:, edge] = length
        normal = np.column_stack([direction[:, 1], -direction[:, 0]]) / length[:, None]
        traction_x = stress[:, 0] * normal[:, 0, None] + stress[:, 2] * normal[:, 1, None]
        traction_y = stress[:, 2] * normal[:, 0, None] + stress[:, 1] * normal[:, 1, None]
        indices = np.asarray([edge, (edge + 1) % 4, 4 + edge])
        integrals = length[:, None] * np.asarray([0.5, 0.5, 2.0 / 3.0])[None, :]
        forces[:, indices] += integrals[:, :, None] * traction_x[:, None, :]
        forces[:, scalar_dofs + indices] += integrals[:, :, None] * traction_y[:, None, :]
    shifted = np.roll(canonical, -1, axis=1)
    area = 0.5 * np.sum(
        canonical[:, :, 0] * shifted[:, :, 1]
        - shifted[:, :, 0] * canonical[:, :, 1],
        axis=1,
    )
    if np.any(area <= 1.0e-14):
        raise ValueError("canonical quadrilateral must have positive area")
    strain_energy_inverse = np.linalg.inv(
        area[:, None, None] * constitutive
    )
    affine = np.matmul(
        np.matmul(forces, strain_energy_inverse),
        np.transpose(forces, (0, 2, 1)),
    )
    affine = 0.5 * (affine + np.transpose(affine, (0, 2, 1)))

    reference = np.asarray(
        [[1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0],
         [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0],
         [1.0 / 3.0, 1.0 / 3.0, 8.0 / 15.0]]
    )
    scalar_mass = np.zeros((count, 8, 8), dtype=np.float64)
    for edge in range(4):
        indices = np.asarray([edge, (edge + 1) % 4, 4 + edge])
        scalar_mass[:, indices[:, None], indices[None, :]] += (
            edge_lengths[:, edge, None, None] * reference[None, :, :]
        )
    scalar_mass_trace = np.matmul(scalar_mass, scalar_trace)
    scalar_gram = np.matmul(
        np.swapaxes(scalar_trace, 1, 2), scalar_mass_trace
    )
    return affine, scalar_trace, scalar_mass_trace, scalar_gram

def _batch_affine_and_projector(
    canonical: np.ndarray, poisson: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:


    affine, scalar_trace, scalar_mass_trace, scalar_gram = (
        _batch_affine_and_projection_data(
            canonical, poisson
        )
    )
    scalar_solution = np.linalg.solve(
        scalar_gram, np.swapaxes(scalar_mass_trace, 1, 2)
    )
    scalar_projector = np.eye(8)[None, :, :] - np.matmul(
        scalar_trace, scalar_solution
    )
    projector = np.zeros((len(canonical), 16, 16), dtype=np.float64)
    projector[:, :8, :8] = scalar_projector
    projector[:, 8:, 8:] = scalar_projector
    return affine, projector

def _batch_project_complement_factors(
    factors: np.ndarray,
    scalar_trace: np.ndarray,
    scalar_mass_trace: np.ndarray,
    scalar_gram: np.ndarray,
) -> np.ndarray:


    rank = factors.shape[2]
    trace_t_factor = np.concatenate(
        (
            np.matmul(np.swapaxes(scalar_trace, 1, 2), factors[:, :8]),
            np.matmul(np.swapaxes(scalar_trace, 1, 2), factors[:, 8:]),
        ),
        axis=2,
    )
    coefficients = np.linalg.solve(scalar_gram, trace_t_factor)
    projected = factors.copy()
    projected[:, :8] -= np.matmul(
        scalar_mass_trace, coefficients[:, :, :rank]
    )
    projected[:, 8:] -= np.matmul(
        scalar_mass_trace, coefficients[:, :, rank:]
    )
    return projected

def _broadcast(value: float | Sequence[float], count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        result = np.full(count, float(array), dtype=np.float64)
    elif array.shape == (count,):
        result = array.copy()
    else:
        raise ValueError(f"{name} must be scalar or shape ({count},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result

def _batch_restore_affine_operators(
    affine: np.ndarray,
    projectors: np.ndarray,
    factors: np.ndarray,
    transforms: np.ndarray,
    material_scale: np.ndarray,
) -> np.ndarray:


    projected = np.matmul(np.swapaxes(projectors, 1, 2), factors)
    canonical = affine + np.matmul(projected, np.swapaxes(projected, 1, 2))
    canonical = 0.5 * (canonical + np.swapaxes(canonical, 1, 2))
    native = np.matmul(
        np.matmul(np.swapaxes(transforms, 1, 2), canonical), transforms
    )
    native = 0.5 * (native + np.swapaxes(native, 1, 2))
    return native * material_scale[:, None, None]

def _batch_canonical_directional_projection(
    canonical: object,
    native_projection: np.ndarray,
) -> np.ndarray:


    projection = np.asarray(native_projection, dtype=np.float64)
    count = len(projection)
    if projection.shape != (count, 16, 12):
        raise ValueError(
            "structured directional projection must have shape (cell_count,16,12)"
        )
    edge = np.arange(4, dtype=np.int64)
    directions = np.stack(
        (
            projection[:, 4 + edge, 8 + edge],
            projection[:, 12 + edge, 8 + edge],
        ),
        axis=2,
    )
    return _batch_canonical_directional_projection_from_directions(
        canonical, directions
    )

def _batch_canonical_directional_projection_from_directions(
    canonical: object,
    native_edge_directions: np.ndarray,
) -> np.ndarray:


    directions = np.asarray(native_edge_directions, dtype=np.float64)
    count = len(directions)
    if directions.shape != (count, 4, 2):
        raise ValueError("native_edge_directions must have shape (cell_count,4,2)")
    rotations = np.asarray(canonical.coordinate_transforms, dtype=np.float64)
    indices = np.asarray(
        canonical.input_indices_for_canonical, dtype=np.int64
    )
    steps = np.asarray(canonical.traversal_steps, dtype=np.int64)
    rows = np.arange(count, dtype=np.int64)
    canonical_projection = np.zeros((count, 16, 12), dtype=np.float64)

    for canonical_vertex in range(4):
        original_vertex = indices[:, canonical_vertex]
        canonical_projection[rows, canonical_vertex, original_vertex] = rotations[
            :, 0, 0
        ]
        canonical_projection[
            rows, canonical_vertex, 4 + original_vertex
        ] = rotations[:, 0, 1]
        canonical_projection[
            rows, 8 + canonical_vertex, original_vertex
        ] = rotations[:, 1, 0]
        canonical_projection[
            rows, 8 + canonical_vertex, 4 + original_vertex
        ] = rotations[:, 1, 1]

    direction_x = directions[:, :, 0]
    direction_y = directions[:, :, 1]
    for canonical_edge in range(4):
        original_edge = np.where(
            steps == 1,
            indices[:, canonical_edge],
            indices[:, (canonical_edge + 1) % 4],
        )
        dx = direction_x[rows, original_edge]
        dy = direction_y[rows, original_edge]
        canonical_projection[
            rows, 4 + canonical_edge, 8 + original_edge
        ] = rotations[:, 0, 0] * dx + rotations[:, 0, 1] * dy
        canonical_projection[
            rows, 12 + canonical_edge, 8 + original_edge
        ] = rotations[:, 1, 0] * dx + rotations[:, 1, 1] * dy
    return canonical_projection

@dataclass(frozen=True)
class NumpyAffineComplementModel:


    feature_mean: np.ndarray
    feature_scale: np.ndarray
    base_factor: np.ndarray
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    correction_scale: float

    def factor(self, features: np.ndarray) -> np.ndarray:
        values = (
            np.asarray(features, dtype=np.float32) - self.feature_mean
        ) / self.feature_scale
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            values = values @ weight.T + bias
            if index + 1 < len(self.weights):
                values = values / (1.0 + np.exp(-values))
        count = len(values)
        return self.base_factor[None, :, :] + float(self.correction_scale) * (
            values.reshape(count, *self.base_factor.shape)
        )

def _infer_factors(
    model: AffineComplementCholeskyNet | NumpyAffineComplementModel,
    feature_array: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    if device == "numpy":
        for start in range(0, len(feature_array), int(batch_size)):
            stop = min(len(feature_array), start + int(batch_size))
            chunks.append(
                np.asarray(model.factor(feature_array[start:stop]), dtype=np.float64)
            )
        return np.concatenate(chunks, axis=0)

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    with torch.no_grad():
        for start in range(0, len(feature_array), int(batch_size)):
            stop = min(len(feature_array), start + int(batch_size))
            factor = model.factor(
                torch.as_tensor(feature_array[start:stop], device=device)
            )
            chunks.append(np.asarray(factor.detach().cpu(), dtype=np.float64))
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    return np.concatenate(chunks, axis=0)

@dataclass(frozen=True)
class AffineOperatorTiming:
    cell_count: int
    canonical_and_analytic_seconds: float
    inference_seconds: float
    stabilization_and_restore_seconds: float
    total_seconds: float

    @property
    def feature_seconds(self) -> float:
        return self.canonical_and_analytic_seconds

    @property
    def frame_transform_seconds(self) -> float:
        return self.stabilization_and_restore_seconds

@dataclass(frozen=True)
class AffineOperatorBatch:
    matrices: np.ndarray
    timing: AffineOperatorTiming
    canonical: tuple[CanonicalQuad, ...]
    canonical_features: np.ndarray

@dataclass(frozen=True)
class ReducedAffineOperatorBatch:


    matrices: np.ndarray
    timing: AffineOperatorTiming
    canonical_vertices: np.ndarray
    canonical_features: np.ndarray

@dataclass(frozen=True)
class FixedQuadAffineOperator:
    model: AffineComplementCholeskyNet | NumpyAffineComplementModel
    payload: dict
    device: str
    checkpoint_load_seconds: float

    @property
    def bubbles_per_edge(self) -> int:
        return 1

    def predict_local_stiffnesses(
        self,
        vertices: Sequence[np.ndarray] | np.ndarray,
        *,
        poisson: float | Sequence[float],
        young: float | Sequence[float] = 1.0,
        thickness: float | Sequence[float] = 1.0,
        batch_size: int = 16384,
    ) -> AffineOperatorBatch:
        total_start = time.perf_counter()
        array = np.asarray(vertices, dtype=np.float64)
        if array.shape == (4, 2):
            batch = array[None, :, :]
        elif array.ndim == 3 and array.shape[1:] == (4, 2):
            batch = array
        else:
            raise ValueError(
                f"vertices must have shape (4,2) or (cell_count,4,2), got {array.shape}"
            )
        if len(batch) == 0:
            raise ValueError("vertices must contain at least one quadrilateral")
        count = len(batch)
        nu = _broadcast(poisson, count, "poisson")
        young_values = _broadcast(young, count, "young")
        thickness_values = _broadcast(thickness, count, "thickness")
        if np.any(young_values <= 0.0) or np.any(thickness_values <= 0.0):
            raise ValueError("young and thickness must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        analytic_start = time.perf_counter()
        canonical_arrays = canonicalize_quad_batch_arrays(
            batch, bubbles_per_edge=1
        )
        canonical_array = canonical_arrays.vertices
        transforms = canonical_arrays.port_transforms
        if transforms is None:
            raise RuntimeError("full prediction requires canonical port transforms")
        feature_array = _batch_features(canonical_array, nu).astype(np.float32)
        affine, trace, mass_trace, gram = _batch_affine_and_projection_data(
            canonical_array, nu
        )
        analytic_seconds = time.perf_counter() - analytic_start

        inference_start = time.perf_counter()
        factors = _infer_factors(
            self.model,
            feature_array,
            device=self.device,
            batch_size=int(batch_size),
        )
        inference_seconds = time.perf_counter() - inference_start

        restore_start = time.perf_counter()
        projected_factor = _batch_project_complement_factors(
            factors, trace, mass_trace, gram
        )
        canonical_stiffness = affine + np.matmul(
            projected_factor, np.swapaxes(projected_factor, 1, 2)
        )
        canonical_stiffness = 0.5 * (
            canonical_stiffness + np.swapaxes(canonical_stiffness, 1, 2)
        )
        matrices = np.matmul(
            np.matmul(np.swapaxes(transforms, 1, 2), canonical_stiffness),
            transforms,
        )
        matrices = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
        matrices *= (young_values * thickness_values)[:, None, None]
        canonical = tuple(
            CanonicalQuad(
                vertices=canonical_array[index],
                port_transform=transforms[index],
                coordinate_transform=canonical_arrays.coordinate_transforms[index],
                input_indices_for_canonical=tuple(
                    int(value)
                    for value in canonical_arrays.input_indices_for_canonical[index]
                ),
                traversal_step=int(canonical_arrays.traversal_steps[index]),
                scale=float(canonical_arrays.scales[index]),
            )
            for index in range(count)
        )
        restore_seconds = time.perf_counter() - restore_start
        return AffineOperatorBatch(
            matrices=matrices,
            timing=AffineOperatorTiming(
                cell_count=count,
                canonical_and_analytic_seconds=float(analytic_seconds),
                inference_seconds=float(inference_seconds),
                stabilization_and_restore_seconds=float(restore_seconds),
                total_seconds=float(time.perf_counter() - total_start),
            ),
            canonical=canonical,
            canonical_features=np.asarray(feature_array, dtype=np.float64),
        )

    def predict_reduced_local_stiffnesses(
        self,
        vertices: Sequence[np.ndarray] | np.ndarray,
        native_projection: np.ndarray,
        *,
        poisson: float | Sequence[float],
        young: float | Sequence[float] = 1.0,
        thickness: float | Sequence[float] = 1.0,
        batch_size: int = 16384,
        structured_directional: bool = False,
    ) -> ReducedAffineOperatorBatch:









        total_start = time.perf_counter()
        array = np.asarray(vertices, dtype=np.float64)
        if array.shape == (4, 2):
            batch = array[None, :, :]
        elif array.ndim == 3 and array.shape[1:] == (4, 2):
            batch = array
        else:
            raise ValueError(
                f"vertices must have shape (4,2) or (cell_count,4,2), got {array.shape}"
            )
        if len(batch) == 0:
            raise ValueError("vertices must contain at least one quadrilateral")
        count = len(batch)
        projection = np.asarray(native_projection, dtype=np.float64)
        compact_directions = bool(
            structured_directional and projection.shape == (count, 4, 2)
        )
        if not compact_directions and (
            projection.ndim != 3 or projection.shape[:2] != (count, 16)
        ):
            raise ValueError(
                "native_projection must have shape (cell_count,16,reduced_dofs), "
                "or (cell_count,4,2) for a compact structured directional map"
            )
        nu = _broadcast(poisson, count, "poisson")
        young_values = _broadcast(young, count, "young")
        thickness_values = _broadcast(thickness, count, "thickness")
        if np.any(young_values <= 0.0) or np.any(thickness_values <= 0.0):
            raise ValueError("young and thickness must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        analytic_start = time.perf_counter()
        canonical = canonicalize_quad_batch_arrays(
            batch,
            bubbles_per_edge=1,
            build_port_transforms=not structured_directional,
        )
        feature_array = _batch_features(canonical.vertices, nu).astype(np.float32)
        affine, trace, mass_trace, gram = _batch_affine_and_projection_data(
            canonical.vertices, nu
        )
        analytic_seconds = time.perf_counter() - analytic_start

        inference_start = time.perf_counter()
        factors = _infer_factors(
            self.model,
            feature_array,
            device=self.device,
            batch_size=int(batch_size),
        )
        inference_seconds = time.perf_counter() - inference_start

        restore_start = time.perf_counter()
        projected_factor = _batch_project_complement_factors(
            factors, trace, mass_trace, gram
        )
        if structured_directional:
            if compact_directions:
                canonical_projection = (
                    _batch_canonical_directional_projection_from_directions(
                        canonical, projection
                    )
                )
            else:
                canonical_projection = _batch_canonical_directional_projection(
                    canonical, projection
                )
        else:
            if canonical.port_transforms is None:
                raise RuntimeError("generic projection requires port transforms")
            canonical_projection = np.matmul(
                canonical.port_transforms, projection
            )
        affine_reduced = np.matmul(
            np.swapaxes(canonical_projection, 1, 2),
            np.matmul(affine, canonical_projection),
        )
        reduced_factor = np.matmul(
            np.swapaxes(canonical_projection, 1, 2), projected_factor
        )
        matrices = affine_reduced + np.matmul(
            reduced_factor, np.swapaxes(reduced_factor, 1, 2)
        )
        matrices = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
        matrices *= (young_values * thickness_values)[:, None, None]
        restore_seconds = time.perf_counter() - restore_start
        return ReducedAffineOperatorBatch(
            matrices=matrices,
            timing=AffineOperatorTiming(
                cell_count=count,
                canonical_and_analytic_seconds=float(analytic_seconds),
                inference_seconds=float(inference_seconds),
                stabilization_and_restore_seconds=float(restore_seconds),
                total_seconds=float(time.perf_counter() - total_start),
            ),
            canonical_vertices=np.asarray(canonical.vertices, dtype=np.float64),
            canonical_features=np.asarray(feature_array, dtype=np.float64),
        )

    def predict_and_assemble(
        self,
        mesh: ConformingPolygonMeshAssembly,
        *,
        poisson: float | Sequence[float],
        young: float | Sequence[float] = 1.0,
        thickness: float | Sequence[float] = 1.0,
        batch_size: int = 16384,
    ):


        from src.bnem.quad_elastic_operator import (
            QuadElasticAssemblyResult,
            assemble_sparse_quad_stiffness,
        )

        if mesh.bubbles_per_edge != 1 or any(len(cell) != 4 for cell in mesh.cells):
            raise ValueError("compact affine operator requires U1 quadrilateral cells")
        started = time.perf_counter()
        cells = [mesh.vertices[np.asarray(cell, dtype=np.int64)] for cell in mesh.cells]
        local = self.predict_local_stiffnesses(
            cells,
            poisson=poisson,
            young=young,
            thickness=thickness,
            batch_size=batch_size,
        )
        assembled = assemble_sparse_quad_stiffness(mesh, local.matrices)
        return QuadElasticAssemblyResult(
            matrix=assembled.matrix,
            local=local,
            assembly_seconds=assembled.assembly_seconds,
            total_seconds=float(time.perf_counter() - started),
        )

def load_fixed_quad_affine_operator(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> FixedQuadAffineOperator:
    started = time.perf_counter()
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if payload.get("model_kind") != "AffineComplementCholeskyNet":
        raise ValueError("checkpoint is not an AffineComplementCholeskyNet")
    config = dict(payload["model_config"])
    model = AffineComplementCholeskyNet(**config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return FixedQuadAffineOperator(
        model=model,
        payload=payload,
        device=str(device),
        checkpoint_load_seconds=float(time.perf_counter() - started),
    )

def export_fixed_quad_affine_numpy_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
) -> Path:


    operator = load_fixed_quad_affine_operator(checkpoint, device="cpu")
    if not isinstance(operator.model, AffineComplementCholeskyNet):
        raise TypeError("expected a PyTorch affine complement model")
    linear = [
        module
        for module in operator.model.network
        if isinstance(module, torch.nn.Linear)
    ]
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray([1], dtype=np.int64),
        "layer_count": np.asarray([len(linear)], dtype=np.int64),
        "feature_mean": operator.model.feature_mean.detach().cpu().numpy(),
        "feature_scale": operator.model.feature_scale.detach().cpu().numpy(),
        "base_factor": operator.model.base_factor.detach().cpu().numpy(),
        "correction_scale": np.asarray(
            [operator.model.correction_scale], dtype=np.float64
        ),
    }
    for index, layer in enumerate(linear):
        arrays[f"weight_{index}"] = layer.weight.detach().cpu().numpy()
        arrays[f"bias_{index}"] = layer.bias.detach().cpu().numpy()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        header = [
            260717.0,
            float(len(linear)),
            float(arrays["feature_mean"].size),
            float(arrays["base_factor"].shape[0]),
            float(arrays["base_factor"].shape[1]),
            float(arrays["correction_scale"][0]),
        ]
        for layer in linear:
            header.extend(
                (float(layer.weight.shape[0]), float(layer.weight.shape[1]))
            )
        packed = np.concatenate(
            [
                np.asarray(header, dtype=np.float32),
                arrays["feature_mean"].astype(np.float32).reshape(-1),
                arrays["feature_scale"].astype(np.float32).reshape(-1),
                arrays["base_factor"].astype(np.float32).reshape(-1),
                *[
                    value.astype(np.float32).reshape(-1)
                    for index in range(len(linear))
                    for value in (
                        arrays[f"weight_{index}"],
                        arrays[f"bias_{index}"],
                    )
                ],
            ]
        )
        np.save(path, packed, allow_pickle=False)
    else:
        np.savez(path, **arrays)
    return path

def load_fixed_quad_affine_numpy_operator(
    checkpoint: str | Path,
) -> FixedQuadAffineOperator:


    started = time.perf_counter()
    path = Path(checkpoint)
    if path.suffix.lower() == ".npy":
        packed = np.load(path, allow_pickle=False)
        if packed.ndim != 1 or int(packed[0]) != 260717:
            raise ValueError("invalid packed NumPy affine checkpoint")
        layer_count = int(packed[1])
        feature_dim = int(packed[2])
        port_dofs = int(packed[3])
        complement_rank = int(packed[4])
        correction_scale = float(packed[5])
        layer_shapes = [
            (int(packed[6 + 2 * index]), int(packed[7 + 2 * index]))
            for index in range(layer_count)
        ]
        offset = 6 + 2 * layer_count

        def take(count: int) -> np.ndarray:
            nonlocal offset
            result = np.asarray(packed[offset : offset + count], dtype=np.float32)
            offset += count
            return result

        feature_mean = take(feature_dim)
        feature_scale = take(feature_dim)
        base_factor = take(port_dofs * complement_rank).reshape(
            port_dofs, complement_rank
        )
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for output_dim, input_dim in layer_shapes:
            weights.append(take(output_dim * input_dim).reshape(output_dim, input_dim))
            biases.append(take(output_dim))
        if offset != len(packed):
            raise ValueError("packed NumPy affine checkpoint has trailing data")
        model = NumpyAffineComplementModel(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            base_factor=base_factor,
            weights=tuple(weights),
            biases=tuple(biases),
            correction_scale=correction_scale,
        )
    else:
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["format_version"][0]) != 1:
                raise ValueError("unsupported NumPy affine checkpoint version")
            layer_count = int(archive["layer_count"][0])
            model = NumpyAffineComplementModel(
                feature_mean=np.asarray(archive["feature_mean"], dtype=np.float32),
                feature_scale=np.asarray(archive["feature_scale"], dtype=np.float32),
                base_factor=np.asarray(archive["base_factor"], dtype=np.float32),
                weights=tuple(
                    np.asarray(archive[f"weight_{index}"], dtype=np.float32)
                    for index in range(layer_count)
                ),
                biases=tuple(
                    np.asarray(archive[f"bias_{index}"], dtype=np.float32)
                    for index in range(layer_count)
                ),
                correction_scale=float(archive["correction_scale"][0]),
            )
    return FixedQuadAffineOperator(
        model=model,
        payload={
            "model_kind": "NumpyAffineComplementCholeskyNet",
            "format_version": 1,
        },
        device="numpy",
        checkpoint_load_seconds=float(time.perf_counter() - started),
    )

__all__ = [
    "AffineOperatorBatch",
    "AffineOperatorTiming",
    "FixedQuadAffineOperator",
    "NumpyAffineComplementModel",
    "ReducedAffineOperatorBatch",
    "export_fixed_quad_affine_numpy_checkpoint",
    "load_fixed_quad_affine_operator",
    "load_fixed_quad_affine_numpy_operator",
]

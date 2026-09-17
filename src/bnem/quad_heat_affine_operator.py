

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

from src.bnem.quad_affine_model import AffineComplementCholeskyNet
from src.bnem.quad_canonical import (
    CanonicalQuad,
    canonicalize_quad_batch_arrays,
    scalar_quad_port_transforms_from_arrays,
)
from src.bnem.quad_affine_operator import (
    NumpyAffineComplementModel,
    _batch_features,
    _infer_factors,
    load_fixed_quad_affine_numpy_operator,
)
from src.bnem.quad_affine_schur import _edge_reference_integrals

def _batch_heat_affine_and_projector(
    canonical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(canonical)
    trace = np.zeros((count, 12, 3), dtype=np.float64)
    trace[:, :4, 0] = 1.0
    trace[:, :4, 1:] = canonical
    forces = np.zeros((count, 12, 3), dtype=np.float64)
    edge_lengths = np.empty((count, 4), dtype=np.float64)
    reference_mass, reference_load = _edge_reference_integrals(2)
    for edge in range(4):
        direction = canonical[:, (edge + 1) % 4] - canonical[:, edge]
        length = np.linalg.norm(direction, axis=1)
        edge_lengths[:, edge] = length
        normal = np.column_stack([direction[:, 1], -direction[:, 0]]) / length[:, None]
        indices = np.asarray([edge, (edge + 1) % 4, 4 + 2 * edge, 5 + 2 * edge])
        integrals = length[:, None] * reference_load[None, :]
        forces[:, indices, 1] += integrals * normal[:, 0, None]
        forces[:, indices, 2] += integrals * normal[:, 1, None]
    affine_energy = 0.5 * (
        np.matmul(np.transpose(trace, (0, 2, 1)), forces)
        + np.matmul(np.transpose(forces, (0, 2, 1)), trace)
    )
    values, vectors = np.linalg.eigh(affine_energy)
    scale = np.maximum(np.max(np.abs(values), axis=1), 1.0)
    inverse = np.zeros_like(values)
    np.divide(1.0, values, out=inverse, where=values > 1.0e-12 * scale[:, None])
    pseudoinverse = np.matmul(
        vectors * inverse[:, None, :], np.transpose(vectors, (0, 2, 1))
    )
    affine = np.matmul(np.matmul(forces, pseudoinverse), np.transpose(forces, (0, 2, 1)))
    affine = 0.5 * (affine + np.transpose(affine, (0, 2, 1)))
    mass = np.zeros((count, 12, 12), dtype=np.float64)
    for edge in range(4):
        indices = np.asarray([edge, (edge + 1) % 4, 4 + 2 * edge, 5 + 2 * edge])
        mass[:, indices[:, None], indices[None, :]] += (
            edge_lengths[:, edge, None, None] * reference_mass[None, :, :]
        )
    trace_t_mass = np.matmul(np.transpose(trace, (0, 2, 1)), mass)
    gram = np.matmul(trace_t_mass, trace)
    projector = np.eye(12)[None, :, :] - np.matmul(
        trace, np.linalg.solve(gram, trace_t_mass)
    )
    return affine, projector

def _batch_heat_affine_and_projection_data(
    canonical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


    count = len(canonical)
    trace = np.zeros((count, 12, 3), dtype=np.float64)
    trace[:, :4, 0] = 1.0
    trace[:, :4, 1:] = canonical
    forces = np.zeros((count, 12, 3), dtype=np.float64)
    edge_lengths = np.empty((count, 4), dtype=np.float64)
    reference_mass, reference_load = _edge_reference_integrals(2)
    for edge in range(4):
        direction = canonical[:, (edge + 1) % 4] - canonical[:, edge]
        length = np.linalg.norm(direction, axis=1)
        edge_lengths[:, edge] = length
        normal = np.column_stack([direction[:, 1], -direction[:, 0]]) / length[:, None]
        indices = np.asarray([edge, (edge + 1) % 4, 4 + 2 * edge, 5 + 2 * edge])
        integrals = length[:, None] * reference_load[None, :]
        forces[:, indices, 1] += integrals * normal[:, 0, None]
        forces[:, indices, 2] += integrals * normal[:, 1, None]
    shifted = np.roll(canonical, -1, axis=1)
    area = 0.5 * np.sum(
        canonical[:, :, 0] * shifted[:, :, 1]
        - shifted[:, :, 0] * canonical[:, :, 1],
        axis=1,
    )
    if np.any(area <= 1.0e-14):
        raise ValueError("canonical quadrilateral must have positive area")

    gradient_forces = forces[:, :, 1:]
    affine = np.matmul(
        gradient_forces, np.transpose(gradient_forces, (0, 2, 1))
    ) / area[:, None, None]
    affine = 0.5 * (affine + np.transpose(affine, (0, 2, 1)))
    mass_trace = np.zeros((count, 12, 3), dtype=np.float64)
    for edge in range(4):
        indices = np.asarray([edge, (edge + 1) % 4, 4 + 2 * edge, 5 + 2 * edge])
        mass_trace[:, indices, :] += (
            edge_lengths[:, edge, None, None]
            * np.matmul(reference_mass[None, :, :], trace[:, indices, :])
        )
    gram = np.matmul(np.transpose(trace, (0, 2, 1)), mass_trace)
    return affine, trace, mass_trace, gram

def _batch_project_heat_complement_factors(
    factors: np.ndarray,
    trace: np.ndarray,
    mass_trace: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:


    coefficients = np.linalg.solve(
        gram,
        np.matmul(np.transpose(trace, (0, 2, 1)), factors),
    )
    return factors - np.matmul(mass_trace, coefficients)

def _batch_restore_projected_heat_operators(
    affine: np.ndarray,
    projected_factors: np.ndarray,
    transforms: np.ndarray,
    conductivity: np.ndarray,
) -> np.ndarray:


    matrices = affine + np.matmul(
        projected_factors, np.swapaxes(projected_factors, 1, 2)
    )
    matrices = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
    native = np.matmul(
        np.matmul(np.swapaxes(transforms, 1, 2), matrices), transforms
    )
    native = 0.5 * (native + np.swapaxes(native, 1, 2))
    return native * conductivity[:, None, None]

def _batch_restore_heat_operators(
    affine: np.ndarray,
    projectors: np.ndarray,
    factors: np.ndarray,
    transforms: np.ndarray,
    conductivity: np.ndarray,
) -> np.ndarray:


    projected = np.matmul(np.swapaxes(projectors, 1, 2), factors)
    matrices = affine + np.matmul(projected, np.swapaxes(projected, 1, 2))
    matrices = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
    native = np.matmul(
        np.matmul(np.swapaxes(transforms, 1, 2), matrices), transforms
    )
    native = 0.5 * (native + np.swapaxes(native, 1, 2))
    return native * conductivity[:, None, None]

@dataclass(frozen=True)
class HeatAffineTiming:
    cell_count: int
    analytic_seconds: float
    inference_seconds: float
    restore_seconds: float
    total_seconds: float

@dataclass(frozen=True)
class HeatAffineBatch:
    matrices: np.ndarray
    canonical: tuple[CanonicalQuad, ...]
    canonical_features: np.ndarray
    timing: HeatAffineTiming

@dataclass(frozen=True)
class TorchNumpyAffineComplementModel:


    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    base_factor: torch.Tensor
    weights: tuple[torch.Tensor, ...]
    biases: tuple[torch.Tensor, ...]
    correction_scale: float

    @classmethod
    def from_numpy(
        cls,
        model: NumpyAffineComplementModel,
        *,
        device: str,
    ) -> "TorchNumpyAffineComplementModel":
        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        return cls(
            feature_mean=tensor(model.feature_mean),
            feature_scale=tensor(model.feature_scale),
            base_factor=tensor(model.base_factor),
            weights=tuple(tensor(value) for value in model.weights),
            biases=tuple(tensor(value) for value in model.biases),
            correction_scale=float(model.correction_scale),
        )

    def factor(self, features: torch.Tensor) -> torch.Tensor:
        values = (features - self.feature_mean) / self.feature_scale
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            values = values @ weight.T + bias
            if index + 1 < len(self.weights):
                values = values / (1.0 + torch.exp(-values))
        return self.base_factor[None, :, :] + self.correction_scale * values.reshape(
            len(values), *self.base_factor.shape
        )

@dataclass(frozen=True)
class FixedQuadHeatAffineOperator:
    model: (
        AffineComplementCholeskyNet
        | NumpyAffineComplementModel
        | TorchNumpyAffineComplementModel
    )
    payload: dict
    device: str
    checkpoint_load_seconds: float

    @property
    def bubbles_per_edge(self) -> int:
        return 2

    def predict_local_stiffnesses(
        self,
        vertices: Sequence[np.ndarray] | np.ndarray,
        *,
        conductivity: float | Sequence[float] = 1.0,
        batch_size: int = 4096,
        return_metadata: bool = True,
    ) -> HeatAffineBatch:


        started = time.perf_counter()
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
        conductivity_values = np.asarray(conductivity, dtype=np.float64)
        if conductivity_values.ndim == 0:
            conductivity_values = np.full(count, float(conductivity_values))
        if conductivity_values.shape != (count,) or np.any(conductivity_values <= 0.0):
            raise ValueError("conductivity must be positive scalar or one value per cell")

        analytic_started = time.perf_counter()
        canonical_arrays = canonicalize_quad_batch_arrays(
            batch,
            bubbles_per_edge=2,
            build_port_transforms=bool(return_metadata),
        )
        canonical_array = canonical_arrays.vertices
        transforms = scalar_quad_port_transforms_from_arrays(
            canonical_arrays, bubbles_per_edge=2
        )
        feature_array = _batch_features(
            canonical_array, np.zeros(count, dtype=np.float64)
        ).astype(np.float32)
        if return_metadata:
            affine, projectors = _batch_heat_affine_and_projector(canonical_array)
            projection_data = None
        else:
            affine, trace, mass_trace, gram = _batch_heat_affine_and_projection_data(
                canonical_array
            )
            projectors = None
            projection_data = (trace, mass_trace, gram)
        analytic_seconds = time.perf_counter() - analytic_started

        inference_started = time.perf_counter()
        factor_array = _infer_factors(
            self.model,
            feature_array,
            device=self.device,
            batch_size=batch_size,
        )
        inference_seconds = time.perf_counter() - inference_started

        restore_started = time.perf_counter()
        if return_metadata:
            if projectors is None:
                raise RuntimeError("metadata path unexpectedly omitted projectors")
            matrices = _batch_restore_heat_operators(
                affine,
                projectors,
                factor_array,
                transforms,
                conductivity_values,
            )
        else:
            if projection_data is None:
                raise RuntimeError("production path unexpectedly omitted projection data")
            trace, mass_trace, gram = projection_data
            projected_factors = _batch_project_heat_complement_factors(
                factor_array, trace, mass_trace, gram
            )
            matrices = _batch_restore_projected_heat_operators(
                affine,
                projected_factors,
                transforms,
                conductivity_values,
            )
        if return_metadata:
            if canonical_arrays.port_transforms is None:
                raise RuntimeError("metadata path unexpectedly omitted port transforms")
            canonical = tuple(
                CanonicalQuad(
                    vertices=canonical_array[index],
                    port_transform=canonical_arrays.port_transforms[index],
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
            reported_features = np.asarray(feature_array, dtype=np.float64)
        else:
            canonical = ()
            reported_features = np.empty((0, feature_array.shape[1]), dtype=np.float64)
        restore_seconds = time.perf_counter() - restore_started
        return HeatAffineBatch(
            matrices=matrices,
            canonical=canonical,
            canonical_features=reported_features,
            timing=HeatAffineTiming(
                cell_count=count,
                analytic_seconds=float(analytic_seconds),
                inference_seconds=float(inference_seconds),
                restore_seconds=float(restore_seconds),
                total_seconds=float(time.perf_counter() - started),
            ),
        )

def load_fixed_quad_heat_affine_operator(
    checkpoint: str | Path, *, device: str = "cpu"
) -> FixedQuadHeatAffineOperator:
    started = time.perf_counter()
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if payload.get("model_kind") != "HeatAffineComplementCholeskyNet":
        raise ValueError("checkpoint is not a compact affine heat operator")
    model = AffineComplementCholeskyNet(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return FixedQuadHeatAffineOperator(
        model=model,
        payload=payload,
        device=str(device),
        checkpoint_load_seconds=float(time.perf_counter() - started),
    )

def export_fixed_quad_heat_affine_numpy_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
) -> Path:


    operator = load_fixed_quad_heat_affine_operator(checkpoint, device="cpu")
    if not isinstance(operator.model, AffineComplementCholeskyNet):
        raise TypeError("expected a PyTorch affine complement model")
    linear = [
        module
        for module in operator.model.network
        if isinstance(module, torch.nn.Linear)
    ]
    header = [
        260717.0,
        float(len(linear)),
        float(operator.model.feature_mean.numel()),
        float(operator.model.base_factor.shape[0]),
        float(operator.model.base_factor.shape[1]),
        float(operator.model.correction_scale),
    ]
    for layer in linear:
        header.extend((float(layer.weight.shape[0]), float(layer.weight.shape[1])))
    packed = np.concatenate(
        [
            np.asarray(header, dtype=np.float32),
            operator.model.feature_mean.detach().cpu().numpy().astype(np.float32).reshape(-1),
            operator.model.feature_scale.detach().cpu().numpy().astype(np.float32).reshape(-1),
            operator.model.base_factor.detach().cpu().numpy().astype(np.float32).reshape(-1),
            *[
                value.detach().cpu().numpy().astype(np.float32).reshape(-1)
                for layer in linear
                for value in (layer.weight, layer.bias)
            ],
        ]
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, packed, allow_pickle=False)
    return path

def load_fixed_quad_heat_affine_numpy_operator(
    checkpoint: str | Path,
) -> FixedQuadHeatAffineOperator:


    started = time.perf_counter()
    generic = load_fixed_quad_affine_numpy_operator(checkpoint)
    return FixedQuadHeatAffineOperator(
        model=generic.model,
        payload={
            "model_kind": "NumpyHeatAffineComplementCholeskyNet",
            "format_version": 1,
        },
        device="numpy",
        checkpoint_load_seconds=float(time.perf_counter() - started),
    )

def load_fixed_quad_heat_affine_numpy_torch_operator(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> FixedQuadHeatAffineOperator:


    started = time.perf_counter()
    generic = load_fixed_quad_affine_numpy_operator(checkpoint)
    model = TorchNumpyAffineComplementModel.from_numpy(
        generic.model,
        device=str(device),
    )
    return FixedQuadHeatAffineOperator(
        model=model,
        payload={
            "model_kind": "TorchNumpyHeatAffineComplementCholeskyNet",
            "format_version": 1,
        },
        device=str(device),
        checkpoint_load_seconds=float(time.perf_counter() - started),
    )

__all__ = [
    "FixedQuadHeatAffineOperator",
    "HeatAffineBatch",
    "HeatAffineTiming",
    "export_fixed_quad_heat_affine_numpy_checkpoint",
    "load_fixed_quad_heat_affine_operator",
    "load_fixed_quad_heat_affine_numpy_operator",
    "load_fixed_quad_heat_affine_numpy_torch_operator",
]

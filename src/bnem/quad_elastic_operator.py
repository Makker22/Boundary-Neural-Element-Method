from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

from src.bnem.assembly import ConformingPolygonMeshAssembly
from src.bnem.elastic_dataset import polygon_operator_features
from src.bnem.polygon_geometry import LEGACY_VERTEX_RMS_SCHEMA
from src.bnem.polygon_ports import PolygonVertexBubblePort
from src.bnem.structured_delta_schur import FixedPortStructuredEnergyKernelElasticNet

_DIRECT_MODEL_ARGUMENTS = {
    "port_dofs",
    "node_feature_dim",
    "global_feature_dim",
    "hidden_dim",
    "message_layers",
    "stencil_radius",
    "diagonal_block",
    "symbol_modes",
    "corner_rank",
    "vector_symbol_coupling_modes",
    "global_modal_rank",
    "global_modal_basis_mode",
    "pairwise_experts",
    "global_context_slots",
    "pair_geometry_mode",
    "geometry_feature_mode",
    "modal_basis_mode",
    "far_rank",
}

def local_tn_to_global_transform(node_features: np.ndarray) -> np.ndarray:









    features = np.asarray(node_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] % 2 != 0 or features.shape[1] < 6:
        raise ValueError(f"unexpected node feature shape {features.shape}")
    scalar_count = int(features.shape[0] // 2)
    tangent = np.asarray(features[:scalar_count, 2:4], dtype=np.float64).copy()
    normal = np.asarray(features[:scalar_count, 4:6], dtype=np.float64).copy()
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
    if np.any(tangent_norm <= 1.0e-12) or np.any(normal_norm <= 1.0e-12):
        raise ValueError("node features contain a degenerate tangent/normal frame")
    tangent /= tangent_norm
    normal /= normal_norm
    orthogonality = np.sum(tangent * normal, axis=1)
    if np.max(np.abs(orthogonality)) > 1.0e-7:
        raise ValueError("node tangent/normal frames are not orthogonal")

    transform = np.zeros((2 * scalar_count, 2 * scalar_count), dtype=np.float64)
    scalar = np.arange(scalar_count)
    rows_x = scalar
    rows_y = scalar_count + scalar
    cols_t = scalar
    cols_n = scalar_count + scalar
    transform[rows_x, cols_t] = tangent[:, 0]
    transform[rows_x, cols_n] = normal[:, 0]
    transform[rows_y, cols_t] = tangent[:, 1]
    transform[rows_y, cols_n] = normal[:, 1]
    return transform

@dataclass(frozen=True)
class LocalPredictionTiming:
    cell_count: int
    feature_seconds: float
    inference_seconds: float
    frame_transform_seconds: float
    total_seconds: float

@dataclass(frozen=True)
class LocalStiffnessBatch:


    matrices: np.ndarray
    timing: LocalPredictionTiming

@dataclass(frozen=True)
class SparseAssemblyResult:
    matrix: Any
    assembly_seconds: float
    cell_count: int

@dataclass(frozen=True)
class QuadElasticAssemblyResult:
    matrix: Any
    local: LocalStiffnessBatch
    assembly_seconds: float
    total_seconds: float

def _broadcast_cell_values(
    values: float | Sequence[float] | np.ndarray,
    cell_count: int,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        result = np.full(cell_count, float(array), dtype=np.float64)
    elif array.shape == (cell_count,):
        result = array.copy()
    else:
        raise ValueError(f"{name} must be scalar or shape ({cell_count},), got {array.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result

def _validate_quad(vertices: np.ndarray, index: int) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (4, 2):
        raise ValueError(f"cell {index} vertices must have shape (4,2), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"cell {index} vertices contain non-finite values")
    shifted = np.roll(points, -1, axis=0)
    signed_area = 0.5 * float(
        np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1])
    )
    if signed_area <= 1.0e-14:
        raise ValueError(f"cell {index} must be nondegenerate and counter-clockwise")
    return points

def _synchronize(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)

@dataclass(frozen=True)
class FixedQuadElasticOperator:








    model: FixedPortStructuredEnergyKernelElasticNet
    checkpoint_payload: dict[str, Any]
    device: str = "cpu"
    checkpoint_load_seconds: float = 0.0

    @property
    def component_frame(self) -> str:
        return str(self.checkpoint_payload.get("component_frame", "global"))

    @property
    def port_dofs(self) -> int:
        return int(self.checkpoint_payload.get("model_config", {}).get("port_dofs", self.model.port_dofs))

    @property
    def bubbles_per_edge(self) -> int:
        scalar_dofs = self.port_dofs // 2
        if scalar_dofs % 4 != 0:
            raise ValueError(
                f"quadrilateral scalar port count must be divisible by four, got {scalar_dofs}"
            )
        bubbles = scalar_dofs // 4 - 1
        if bubbles < 0:
            raise ValueError(f"invalid quadrilateral scalar port count {scalar_dofs}")
        return int(bubbles)

    @property
    def material_feature_mode(self) -> str:
        config = dict(self.checkpoint_payload.get("model_config", {}))
        return str(
            config.get(
                "material_feature_mode",
                self.checkpoint_payload.get("material_feature_mode", "raw-nu"),
            )
        )

    @property
    def geometry_normalization_schema(self) -> str:
        config = dict(self.checkpoint_payload.get("model_config", {}))
        return str(
            config.get(
                "geometry_normalization_schema",
                self.checkpoint_payload.get(
                    "geometry_normalization_schema", LEGACY_VERTEX_RMS_SCHEMA
                ),
            )
        )

    def predict_local_stiffnesses(
        self,
        vertices: Sequence[np.ndarray] | np.ndarray,
        *,
        poisson: float | Sequence[float] | np.ndarray,
        young: float | Sequence[float] | np.ndarray = 1.0,
        thickness: float | Sequence[float] | np.ndarray = 1.0,
        batch_size: int = 256,
    ) -> LocalStiffnessBatch:







        total_start = time.perf_counter()
        vertices_array = np.asarray(vertices, dtype=np.float64)
        if vertices_array.shape == (4, 2):
            raw_cells = [vertices_array]
        else:
            raw_cells = list(vertices)
        if not raw_cells:
            raise ValueError("vertices must contain at least one quadrilateral")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cells = [_validate_quad(cell, index) for index, cell in enumerate(raw_cells)]
        cell_count = len(cells)
        poisson_values = _broadcast_cell_values(poisson, cell_count, name="poisson")
        young_values = _broadcast_cell_values(young, cell_count, name="young")
        thickness_values = _broadcast_cell_values(thickness, cell_count, name="thickness")
        if np.any(young_values <= 0.0):
            raise ValueError("young must be positive")
        if np.any(thickness_values <= 0.0):
            raise ValueError("thickness must be positive")

        feature_start = time.perf_counter()
        nodes: list[np.ndarray] = []
        global_rows: list[np.ndarray] = []
        model_rigid: list[np.ndarray] = []
        transforms: list[np.ndarray] = []
        for index, (cell, nu) in enumerate(zip(cells, poisson_values)):
            node, global_features = polygon_operator_features(
                cell,
                self.bubbles_per_edge,
                float(nu),
                material_feature_mode=self.material_feature_mode,
                geometry_normalization_schema=self.geometry_normalization_schema,
            )
            if node.shape != (self.port_dofs, self.model.node_feature_dim):
                raise ValueError(
                    f"cell {index} features have shape {node.shape}; checkpoint expects "
                    f"({self.port_dofs},{self.model.node_feature_dim})"
                )
            rigid_global = PolygonVertexBubblePort(
                cell, bubbles_per_edge=self.bubbles_per_edge
            ).rigid_body_modes()
            if self.component_frame == "local-tn":
                transform = local_tn_to_global_transform(node)
                rigid = transform.T @ rigid_global
            elif self.component_frame == "global":
                transform = np.eye(self.port_dofs, dtype=np.float64)
                rigid = rigid_global
            else:
                raise ValueError(f"unsupported checkpoint component_frame={self.component_frame!r}")
            nodes.append(node)
            global_rows.append(global_features)
            model_rigid.append(rigid)
            transforms.append(transform)
        node_array = np.stack(nodes)
        global_array = np.stack(global_rows)
        rigid_array = np.stack(model_rigid)
        feature_seconds = time.perf_counter() - feature_start

        inference_start = time.perf_counter()
        predicted_chunks: list[np.ndarray] = []
        _synchronize(self.device)
        with torch.no_grad():
            for start in range(0, cell_count, int(batch_size)):
                stop = min(cell_count, start + int(batch_size))
                prediction = self.model(
                    torch.as_tensor(node_array[start:stop], dtype=torch.float32, device=self.device),
                    torch.as_tensor(global_array[start:stop], dtype=torch.float32, device=self.device),
                    torch.as_tensor(rigid_array[start:stop], dtype=torch.float32, device=self.device),
                )
                predicted_chunks.append(
                    np.asarray(prediction.detach().cpu(), dtype=np.float64)
                )
        _synchronize(self.device)
        inference_seconds = time.perf_counter() - inference_start
        predicted = np.concatenate(predicted_chunks, axis=0)

        frame_start = time.perf_counter()
        matrices = np.empty_like(predicted, dtype=np.float64)
        for index, (matrix, transform) in enumerate(zip(predicted, transforms)):
            global_matrix = transform @ matrix @ transform.T
            global_matrix *= float(young_values[index] * thickness_values[index])
            matrices[index] = 0.5 * (global_matrix + global_matrix.T)
        frame_seconds = time.perf_counter() - frame_start
        total_seconds = time.perf_counter() - total_start
        return LocalStiffnessBatch(
            matrices=matrices,
            timing=LocalPredictionTiming(
                cell_count=cell_count,
                feature_seconds=float(feature_seconds),
                inference_seconds=float(inference_seconds),
                frame_transform_seconds=float(frame_seconds),
                total_seconds=float(total_seconds),
            ),
        )

    def predict_and_assemble(
        self,
        mesh: ConformingPolygonMeshAssembly,
        *,
        poisson: float | Sequence[float] | np.ndarray,
        young: float | Sequence[float] | np.ndarray = 1.0,
        thickness: float | Sequence[float] | np.ndarray = 1.0,
        batch_size: int = 256,
    ) -> QuadElasticAssemblyResult:


        if mesh.bubbles_per_edge != self.bubbles_per_edge:
            raise ValueError(
                f"mesh has {mesh.bubbles_per_edge} bubbles per edge; checkpoint expects "
                f"{self.bubbles_per_edge}"
            )
        non_quads = [index for index, cell in enumerate(mesh.cells) if len(cell) != 4]
        if non_quads:
            raise ValueError(f"fixed quadrilateral checkpoint received non-quad cells {non_quads[:8]}")
        total_start = time.perf_counter()
        cell_vertices = [
            mesh.vertices[np.asarray(cell, dtype=np.int64)] for cell in mesh.cells
        ]
        local = self.predict_local_stiffnesses(
            cell_vertices,
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
            total_seconds=float(time.perf_counter() - total_start),
        )

def load_fixed_quad_elastic_operator(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> FixedQuadElasticOperator:


    start = time.perf_counter()
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("elastic checkpoint must contain a dictionary payload")
    model_kind = str(payload.get("model_kind", ""))
    if model_kind != "FixedPortStructuredEnergyKernelElasticNet":
        raise ValueError(
            "fixed quadrilateral bridge currently requires a direct "
            f"FixedPortStructuredEnergyKernelElasticNet checkpoint, got {model_kind!r}"
        )
    if int(payload.get("edge_count", 4)) != 4:
        raise ValueError(f"checkpoint edge_count must be four, got {payload.get('edge_count')!r}")
    raw_config = dict(payload.get("model_config", {}))
    config = {key: value for key, value in raw_config.items() if key in _DIRECT_MODEL_ARGUMENTS}
    config.setdefault("global_modal_basis_mode", "legacy-eigh-complement")
    config.setdefault("pairwise_experts", 1)
    config.setdefault("global_context_slots", 1)
    config.setdefault("pair_geometry_mode", "basic")
    config.setdefault("diagonal_block", True)
    config.setdefault("geometry_feature_mode", "raw-global")
    config.setdefault("modal_basis_mode", "legacy-indexed")
    model = FixedPortStructuredEnergyKernelElasticNet(**config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    operator = FixedQuadElasticOperator(
        model=model,
        checkpoint_payload=payload,
        device=str(device),
        checkpoint_load_seconds=float(time.perf_counter() - start),
    )

    _ = operator.bubbles_per_edge
    if operator.component_frame not in {"global", "local-tn"}:
        raise ValueError(f"unsupported checkpoint component_frame={operator.component_frame!r}")
    return operator

def assemble_sparse_quad_stiffness(
    mesh: ConformingPolygonMeshAssembly,
    local_stiffnesses: Sequence[np.ndarray] | np.ndarray,
) -> SparseAssemblyResult:


    non_quads = [index for index, cell in enumerate(mesh.cells) if len(cell) != 4]
    if non_quads:
        raise ValueError(f"sparse quadrilateral assembly received non-quad cells {non_quads[:8]}")
    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in local_stiffnesses]
    start = time.perf_counter()
    matrix = mesh.assemble_vector(matrices, sparse=True)
    elapsed = time.perf_counter() - start
    return SparseAssemblyResult(
        matrix=matrix,
        assembly_seconds=float(elapsed),
        cell_count=len(matrices),
    )

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.bnem.polygon_ports import PolygonVertexBubblePort

def symmetrize(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (mat + mat.T)

def torch_symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))

def canonicalize_modal_basis_np(basis: np.ndarray) -> np.ndarray:


    arr = np.asarray(basis, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return arr.copy()
    out = arr.copy()
    pivots = np.argmax(np.abs(out), axis=0)
    signs = np.sign(out[pivots, np.arange(out.shape[1])])
    signs[signs == 0.0] = 1.0
    return out * signs[None, :]

def torch_canonicalize_modal_basis(basis: torch.Tensor) -> torch.Tensor:


    if basis.ndim != 2 or int(basis.shape[1]) == 0:
        return basis
    pivots = torch.argmax(torch.abs(basis), dim=0)
    cols = torch.arange(basis.shape[1], dtype=torch.long, device=basis.device)
    signs = torch.sign(basis[pivots, cols])
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    return basis * signs.unsqueeze(0)

def rbm_projector(rigid_body_modes: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    modes = np.asarray(rigid_body_modes, dtype=np.float64)
    if modes.ndim != 2:
        raise ValueError(f"rigid_body_modes must be 2-D, got {modes.shape}")
    if modes.shape[0] < modes.shape[1]:

        modes = modes.T
    gram = modes.T @ modes
    if float(np.linalg.norm(gram)) <= eps:
        return np.eye(modes.shape[0], dtype=np.float64)
    q, _ = np.linalg.qr(modes)
    return np.eye(modes.shape[0], dtype=np.float64) - q @ q.T

def polygon_shape_diagnostics(vertices: np.ndarray) -> dict[str, float]:


    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError(f"vertices must have shape (n>=3,2), got {points.shape}")
    shifted = np.roll(points, -1, axis=0)
    edge_vectors = shifted - points
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    perimeter = max(float(np.sum(edge_lengths)), 1.0e-12)
    signed_area = 0.5 * float(np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1]))
    orientation = 1.0 if signed_area >= 0.0 else -1.0
    compactness = float(4.0 * np.pi * abs(signed_area) / max(perimeter * perimeter, 1.0e-12))
    angles: list[float] = []
    for index in range(int(points.shape[0])):
        incoming = points[index] - points[(index - 1) % points.shape[0]]
        outgoing = points[(index + 1) % points.shape[0]] - points[index]
        incoming /= max(float(np.linalg.norm(incoming)), 1.0e-12)
        outgoing /= max(float(np.linalg.norm(outgoing)), 1.0e-12)
        cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        dot = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
        turn = np.arctan2(orientation * cross, dot)
        internal = np.pi + abs(turn) if turn < 0.0 else np.pi - turn
        angles.append(float(np.degrees(internal)))
    edge_aspect = float(np.max(edge_lengths) / max(float(np.min(edge_lengths)), 1.0e-12))
    return {
        "compactness": compactness,
        "min_angle": float(np.min(angles)),
        "max_angle": float(np.max(angles)),
        "edge_aspect": edge_aspect,
    }

def torch_rbm_projector(rigid_body_modes: torch.Tensor) -> torch.Tensor:
    modes = rigid_body_modes
    if modes.ndim != 2:
        raise ValueError(f"rigid_body_modes must be 2-D, got {tuple(modes.shape)}")
    if modes.shape[0] < modes.shape[1]:
        modes = modes.transpose(0, 1)
    q, _ = torch.linalg.qr(modes, mode="reduced")
    eye = torch.eye(modes.shape[0], dtype=modes.dtype, device=modes.device)
    return eye - q @ q.transpose(0, 1)

def direct_geometry_s0(
    port_points: np.ndarray,
    rigid_body_modes: np.ndarray,
    *,
    edge_weight: float = 1.0,
    steklov_weight: float = 0.15,
    long_range_weight: float = 0.03,
    eps: float = 1.0e-10,
) -> np.ndarray:








    points = np.asarray(port_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"port_points must have shape (n,2), got {points.shape}")
    n = int(points.shape[0])
    dofs = 2 * n
    if dofs != int(np.asarray(rigid_body_modes).shape[-1]) and dofs != int(np.asarray(rigid_body_modes).shape[0]):
        raise ValueError("port point count and rigid-body mode size do not match")

    shifted = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(shifted - points, axis=1)
    mean_length = max(float(np.mean(lengths)), eps)
    weights = 1.0 / np.maximum(lengths / mean_length, eps)
    scalar = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        w = float(edge_weight) * float(weights[i])
        scalar[i, i] += w
        scalar[j, j] += w
        scalar[i, j] -= w
        scalar[j, i] -= w

    if long_range_weight > 0.0:
        span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), eps)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                distance = float(np.linalg.norm(points[i] - points[j]))
                w = float(long_range_weight) / max(distance / span, eps)
                scalar[i, i] += w
                scalar[j, j] += w
                scalar[i, j] -= w
                scalar[j, i] -= w

    if steklov_weight > 0.0:
        tributary = 0.5 * (lengths + np.roll(lengths, 1))
        mass = np.diag(tributary / max(float(np.mean(tributary)), eps))
        scalar += float(steklov_weight) * mass

    raw = np.zeros((dofs, dofs), dtype=np.float64)

    raw[:n, :n] = scalar
    raw[n:, n:] = scalar
    p = rbm_projector(rigid_body_modes)
    projected = symmetrize(p @ raw @ p)
    scale = max(float(np.linalg.norm(projected, ord="fro")), eps)
    return projected / scale

def plane_strain_matrix_unit(poisson: float) -> np.ndarray:


    nu = float(poisson)
    return np.asarray(
        [
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)],
        ],
        dtype=np.float64,
    ) / max((1.0 + nu) * (1.0 - 2.0 * nu), 1.0e-12)

def polygon_area(vertices: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=np.float64)
    shifted = np.roll(points, -1, axis=0)
    return 0.5 * abs(float(np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1])))

def boundary_sample_points(
    vertices: np.ndarray,
    *,
    segments_per_edge: int,
) -> tuple[np.ndarray, list[tuple[int, float]]]:
    points_in = np.asarray(vertices, dtype=np.float64)
    edge_count = int(points_in.shape[0])
    segments = max(int(segments_per_edge), 1)
    points: list[np.ndarray] = []
    params: list[tuple[int, float]] = []
    for edge in range(edge_count):
        start = points_in[edge]
        end = points_in[(edge + 1) % edge_count]
        for local in range(segments):
            t = float(local) / float(segments)
            points.append((1.0 - t) * start + t * end)
            params.append((edge, t))
    return np.asarray(points, dtype=np.float64), params

def port_lift_matrix(
    vertices: np.ndarray,
    bubbles_per_edge: int,
    params: list[tuple[int, float]],
) -> np.ndarray:
    port = PolygonVertexBubblePort(vertices, bubbles_per_edge=int(bubbles_per_edge))
    scalar_count = int(port.scalar_dofs)
    boundary_count = len(params)
    lift = np.zeros((2 * boundary_count, 2 * scalar_count), dtype=np.float64)
    for node_index, (edge, t) in enumerate(params):
        shape = port.shape_on_edge(edge, t)
        lift[2 * node_index, :scalar_count] = shape
        lift[2 * node_index + 1, scalar_count:] = shape
    return lift

def boundary_sample_weights(
    vertices: np.ndarray,
    params: list[tuple[int, float]],
    *,
    segments_per_edge: int,
) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return np.asarray(
        [lengths[int(edge)] / float(max(int(segments_per_edge), 1)) for edge, _ in params],
        dtype=np.float64,
    )

def affine_design(points: np.ndarray) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float64)
    rows = np.zeros((2 * xy.shape[0], 6), dtype=np.float64)
    for index, (x, y) in enumerate(xy):
        rows[2 * index, 0] = 1.0
        rows[2 * index, 2] = float(x)
        rows[2 * index, 3] = float(y)
        rows[2 * index + 1, 1] = 1.0
        rows[2 * index + 1, 4] = float(x)
        rows[2 * index + 1, 5] = float(y)
    return rows

def affine_vem_consistency_stabilization(
    vertices: np.ndarray,
    bubbles_per_edge: int,
    poisson: float,
    rigid_body_modes: np.ndarray,
    *,
    segments_per_edge: int = 8,
) -> tuple[np.ndarray, np.ndarray]:








    boundary_xy, params = boundary_sample_points(vertices, segments_per_edge=segments_per_edge)
    lift = port_lift_matrix(vertices, bubbles_per_edge, params)
    weights = np.repeat(
        boundary_sample_weights(vertices, params, segments_per_edge=segments_per_edge),
        2,
    )
    design = affine_design(boundary_xy)
    weighted_design = design * weights[:, None]
    gram = design.T @ weighted_design
    try:
        fit = np.linalg.solve(gram, design.T * weights[None, :])
    except np.linalg.LinAlgError:
        fit = np.linalg.pinv(gram) @ (design.T * weights[None, :])

    strain_from_affine = np.zeros((3, 6), dtype=np.float64)
    strain_from_affine[0, 2] = 1.0
    strain_from_affine[1, 5] = 1.0
    strain_from_affine[2, 3] = 1.0
    strain_from_affine[2, 4] = 1.0
    strain_map = strain_from_affine @ fit @ lift
    consistency = polygon_area(vertices) * (
        strain_map.T @ plane_strain_matrix_unit(float(poisson)) @ strain_map
    )

    residual_lift = (np.eye(design.shape[0], dtype=np.float64) - design @ fit) @ lift
    stabilization = residual_lift.T @ (weights[:, None] * residual_lift)
    p = rbm_projector(rigid_body_modes)
    consistency = symmetrize(p @ consistency @ p)
    stabilization = symmetrize(p @ stabilization @ p)
    return consistency, stabilization

def direct_affine_vem_s0(
    vertices: np.ndarray,
    bubbles_per_edge: int,
    poisson: float,
    rigid_body_modes: np.ndarray,
    *,
    consistency_coefficient: float = 1.0,
    stabilization_coefficient: float = 1.0,
    segments_per_edge: int = 8,
) -> np.ndarray:
    consistency, stabilization = affine_vem_consistency_stabilization(
        vertices,
        bubbles_per_edge,
        poisson,
        rigid_body_modes,
        segments_per_edge=segments_per_edge,
    )
    return symmetrize(
        float(consistency_coefficient) * consistency
        + float(stabilization_coefficient) * stabilization
    )

def port_points_from_node_features(node_features: np.ndarray) -> np.ndarray:
    features = np.asarray(node_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] < 2 or features.shape[0] % 2 != 0:
        raise ValueError(f"unexpected node feature shape {features.shape}")
    scalar_count = features.shape[0] // 2
    return np.asarray(features[:scalar_count, 0:2], dtype=np.float64)

def whitened_multiplicative_operator(
    s0: np.ndarray,
    c_full: np.ndarray,
    projector: np.ndarray,
    *,
    eig_floor_relative: float = 1.0e-8,
    eig_floor_absolute: float = 1.0e-10,
) -> np.ndarray:
    s0p = symmetrize(projector @ np.asarray(s0, dtype=np.float64) @ projector)
    eigvals, eigvecs = np.linalg.eigh(s0p)
    scale = max(float(np.max(np.abs(eigvals))), eig_floor_absolute)
    floor = max(float(eig_floor_absolute), float(eig_floor_relative) * scale)
    keep = eigvals > floor
    if not np.any(keep):
        return np.zeros_like(s0p)
    v = eigvecs[:, keep]
    sqrt_lam = np.sqrt(np.maximum(eigvals[keep], floor))
    l0 = v * sqrt_lam[None, :]
    c = symmetrize(v.T @ projector @ np.asarray(c_full, dtype=np.float64) @ projector @ v)
    return symmetrize(projector @ l0 @ c @ l0.T @ projector)

class FixedPortStructuredEnergyKernelElasticNet(nn.Module):

































    def __init__(
        self,
        *,
        port_dofs: int = 16,
        node_feature_dim: int = 19,
        global_feature_dim: int = 8,
        hidden_dim: int = 192,
        message_layers: int = 2,
        stencil_radius: int = 2,
        diagonal_block: bool = True,
        symbol_modes: int | None = None,
        corner_rank: int = 0,
        vector_symbol_coupling_modes: int = 0,
        global_modal_rank: int = 8,
        global_modal_basis_mode: str = "physical-project",
        pairwise_experts: int = 1,
        global_context_slots: int = 1,
        pair_geometry_mode: str = "basic",
        geometry_feature_mode: str = "intrinsic-local",
        modal_basis_mode: str = "complete-cyclic",
        far_rank: int = 8,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if port_dofs < 8 or port_dofs % 2 != 0:
            raise ValueError("port_dofs must be an even vector-port size")
        if message_layers < 0:
            raise ValueError("message_layers must be nonnegative")
        if stencil_radius < 0:
            raise ValueError("stencil_radius must be nonnegative")
        if corner_rank < 0:
            raise ValueError("corner_rank must be nonnegative")
        if vector_symbol_coupling_modes < 0:
            raise ValueError("vector_symbol_coupling_modes must be nonnegative")
        if global_modal_rank < 0:
            raise ValueError("global_modal_rank must be nonnegative")
        if pairwise_experts < 1:
            raise ValueError("pairwise_experts must be >= 1")
        if global_context_slots < 1:
            raise ValueError("global_context_slots must be >= 1")
        if pair_geometry_mode not in {"basic", "relative-frame"}:
            raise ValueError("pair_geometry_mode must be 'basic' or 'relative-frame'")
        if global_modal_basis_mode not in {
            "physical-project",
            "physical-pairwise",
            "physical-pairwise-symmetric",
            "legacy-eigh-complement",
        }:
            raise ValueError(
                "global_modal_basis_mode must be 'physical-project', 'physical-pairwise', "
                "'physical-pairwise-symmetric', "
                "or 'legacy-eigh-complement'"
            )
        if geometry_feature_mode not in {"intrinsic-local", "raw-global"}:
            raise ValueError("geometry_feature_mode must be 'intrinsic-local' or 'raw-global'")
        if modal_basis_mode not in {"complete-cyclic", "legacy-indexed"}:
            raise ValueError("modal_basis_mode must be 'complete-cyclic' or 'legacy-indexed'")
        if far_rank < 0:
            raise ValueError("far_rank must be nonnegative")
        self.port_dofs = int(port_dofs)
        self.scalar_dofs = self.port_dofs // 2
        self.node_feature_dim = int(node_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.message_layers = int(message_layers)
        self.stencil_radius = int(stencil_radius)
        self.diagonal_block = bool(diagonal_block)
        self.elastic_modal_dim = self.port_dofs - 3
        self.symbol_modes = int(symbol_modes if symbol_modes is not None else self.port_dofs)
        self.corner_rank = int(corner_rank)
        self.vector_symbol_coupling_modes = int(vector_symbol_coupling_modes)
        self.global_modal_rank = int(global_modal_rank)
        self.global_modal_basis_mode = str(global_modal_basis_mode)
        self.pairwise_experts = int(pairwise_experts)
        self.global_context_slots = int(global_context_slots)
        self.pair_geometry_mode = str(pair_geometry_mode)
        self.pair_relative_dim = 4 if self.pair_geometry_mode == "basic" else 9
        self.geometry_feature_mode = str(geometry_feature_mode)
        self.modal_basis_mode = str(modal_basis_mode)
        self.far_rank = int(far_rank)
        self.eps = float(eps)

        self.token_embed = nn.Sequential(
            nn.Linear(self.node_feature_dim + self.global_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_linears = nn.ModuleList(
            [nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(self.message_layers)]
        )
        self.message_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.message_layers)])
        self.local_block_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.diagonal_block_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        if self.symbol_modes > 0:
            self.symbol_scale_head: nn.Module | None = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.symbol_modes),
            )
        else:
            self.symbol_scale_head = None
        if self.corner_rank > 0:
            self.corner_block_head: nn.Module | None = nn.Sequential(
                nn.Linear(2 * hidden_dim + 6, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.corner_rank * 6),
            )
        else:
            self.corner_block_head = None
        if self.vector_symbol_coupling_modes > 0:
            self.vector_symbol_coupling_head: nn.Module | None = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.vector_symbol_coupling_modes * 3),
            )
            self.vector_symbol_coupling_log_scale = nn.Parameter(torch.tensor(-2.0))
        else:
            self.vector_symbol_coupling_head = None
        self.pairwise_gate: nn.Module | None = None
        self.global_context_score_head: nn.Module | None = None
        if self.global_modal_rank > 0:
            if self.global_modal_basis_mode == "physical-project":

                self.global_modal_head: nn.Module | None = nn.Sequential(
                    nn.Linear(2 * hidden_dim + self.global_feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, self.global_modal_rank),
                )
            elif self.global_modal_basis_mode in {"physical-pairwise", "physical-pairwise-symmetric"}:

                self.global_modal_head = nn.Sequential(
                    nn.Linear(
                        (2 + self.global_context_slots) * hidden_dim
                        + self.global_feature_dim
                        + self.pair_relative_dim,
                        hidden_dim,
                    ),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, self.pairwise_experts),
                )
                if self.global_context_slots > 1:
                    self.global_context_score_head = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, self.global_context_slots - 1),
                    )
                    score_final = self.global_context_score_head[-1]
                    if isinstance(score_final, nn.Linear):
                        nn.init.zeros_(score_final.weight)
                        nn.init.zeros_(score_final.bias)
                if self.pairwise_experts > 1:
                    self.pairwise_gate = nn.Sequential(
                        nn.Linear(self.global_feature_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, self.pairwise_experts),
                    )
                    gate_final = self.pairwise_gate[-1]
                    if isinstance(gate_final, nn.Linear):
                        nn.init.zeros_(gate_final.weight)
                        nn.init.zeros_(gate_final.bias)
                else:
                    self.pairwise_gate = None
            else:
                self.global_modal_head = nn.Sequential(
                    nn.Linear(self.port_dofs * hidden_dim + self.global_feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, self.global_modal_rank * self.elastic_modal_dim),
                )
        else:
            self.global_modal_head = None
            self.pairwise_gate = None
            self.global_context_score_head = None
        if self.far_rank > 0:
            self.far_weight_head: nn.Module | None = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.far_rank),
            )
            self.far_scale_head: nn.Module | None = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.far_rank),
            )
        else:
            self.far_weight_head = None
            self.far_scale_head = None
        self.consistency_log_scale = nn.Parameter(torch.tensor(-2.0))
        self.global_modal_log_scale = nn.Parameter(
            torch.tensor(
                0.0
                if self.global_modal_basis_mode in {"physical-pairwise", "physical-pairwise-symmetric"}
                else -2.0
            )
        )
        self.log_factor_scale = nn.Parameter(torch.tensor(0.0))
        self._reset_structured_heads()

    def _reset_structured_heads(self) -> None:
        for head in (
            self.local_block_head,
            self.diagonal_block_head,
            self.symbol_scale_head,
            self.corner_block_head,
            self.vector_symbol_coupling_head,
            self.far_scale_head,
        ):
            if head is None:
                continue
            last = head[-1]
            if isinstance(last, nn.Linear):
                nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)
                if head is self.corner_block_head:
                    nn.init.zeros_(last.bias)
                elif head is self.vector_symbol_coupling_head:
                    with torch.no_grad():
                        last.bias.zero_()
                        for mode in range(self.vector_symbol_coupling_modes):
                            last.bias[3 * mode] = -3.0
                            last.bias[3 * mode + 2] = -3.0
                else:
                    nn.init.constant_(last.bias, -3.0)
        for head in (self.global_modal_head, self.far_weight_head):
            if head is None:
                continue
            last = head[-1]
            if isinstance(last, nn.Linear):
                nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)
                nn.init.zeros_(last.bias)

    def _scalar_order(self, node_features: torch.Tensor) -> torch.Tensor:
        return torch.argsort(torch.remainder(node_features[:, : self.scalar_dofs, 16], 1.0), dim=1)

    def _intrinsic_node_features(self, node_features: torch.Tensor) -> torch.Tensor:









        if self.geometry_feature_mode == "raw-global":
            return node_features
        intrinsic = node_features.clone()
        position = node_features[..., 0:2]
        tangent = node_features[..., 2:4]
        normal = node_features[..., 4:6]
        intrinsic[..., 0] = torch.sum(position * tangent, dim=-1)
        intrinsic[..., 1] = torch.sum(position * normal, dim=-1)
        intrinsic[..., 2] = 1.0
        intrinsic[..., 3] = 0.0
        intrinsic[..., 4] = 0.0
        intrinsic[..., 5] = 1.0
        intrinsic[..., 16] = 0.0
        return intrinsic

    def _gather_scalar_ordered(self, values: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        return torch.gather(values, 1, order[:, :, None].expand(-1, -1, values.shape[-1]))

    def _scatter_scalar_ordered(self, ordered: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        inverse = torch.empty_like(order)
        base = torch.arange(order.shape[1], dtype=order.dtype, device=order.device)
        inverse.scatter_(1, order, base[None, :].expand_as(order))
        return torch.gather(ordered, 1, inverse[:, :, None].expand(-1, -1, ordered.shape[-1]))

    def encoded_tokens(self, node_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        if node_features.ndim != 3 or node_features.shape[1:] != (self.port_dofs, self.node_feature_dim):
            raise ValueError(
                "node_features must have shape "
                f"[B,{self.port_dofs},{self.node_feature_dim}], got {tuple(node_features.shape)}"
            )
        if global_features.ndim != 2 or global_features.shape != (node_features.shape[0], self.global_feature_dim):
            raise ValueError(
                "global_features must have shape "
                f"[B,{self.global_feature_dim}], got {tuple(global_features.shape)}"
            )
        encoder_features = self._intrinsic_node_features(node_features)
        combined = torch.cat(
            [
                encoder_features,
                global_features[:, None, :].expand(-1, self.port_dofs, -1),
            ],
            dim=-1,
        )
        encoded = self.token_embed(combined)
        if self.message_layers == 0:
            return encoded
        scalar_order = self._scalar_order(node_features)
        for linear, norm in zip(self.message_linears, self.message_norms):
            next_encoded = encoded.clone()
            for component_offset in (0, self.scalar_dofs):
                component_order = scalar_order + component_offset
                ordered = torch.gather(
                    encoded,
                    1,
                    component_order[:, :, None].expand(-1, -1, encoded.shape[-1]),
                )
                previous = torch.roll(ordered, shifts=1, dims=1)
                following = torch.roll(ordered, shifts=-1, dims=1)
                update = torch.nn.functional.silu(linear(torch.cat([previous, ordered, following], dim=-1)))
                ordered = norm(ordered + update)
                next_encoded.scatter_(
                    1,
                    component_order[:, :, None].expand(-1, -1, encoded.shape[-1]),
                    ordered,
                )
            encoded = next_encoded
        return encoded

    def _projector_and_complement(
        self,
        rigid_body_modes: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rigid = rigid_body_modes.to(dtype=dtype, device=device)
        if rigid.ndim == 1:
            rigid = rigid[:, None]
        if rigid.ndim != 2:
            raise ValueError(f"rigid_body_modes must be 2-D for one sample, got {tuple(rigid.shape)}")
        if rigid.shape[0] < rigid.shape[1]:
            rigid = rigid.transpose(0, 1)
        if rigid.shape[0] != self.port_dofs:
            raise ValueError(f"rigid_body_modes has incompatible shape {tuple(rigid.shape)}")

        with torch.no_grad():
            rigid_static = rigid.detach()
            q, _ = torch.linalg.qr(rigid_static, mode="reduced")
            eye = torch.eye(self.port_dofs, dtype=dtype, device=device)
            projector = eye - q @ q.T
            if self.global_modal_basis_mode != "legacy-eigh-complement":
                return projector, torch.zeros(
                    (0, self.port_dofs), dtype=dtype, device=device
                )
            eigvals, eigvecs = torch.linalg.eigh(projector)
            complement_rows = eigvecs[:, eigvals > 0.5].T
        if complement_rows.shape != (self.elastic_modal_dim, self.port_dofs):
            raise RuntimeError(
                f"Expected RBM-complement shape {(self.elastic_modal_dim, self.port_dofs)}, "
                f"got {tuple(complement_rows.shape)}"
            )
        return projector, complement_rows

    def _infer_edge_count(self, global_features: torch.Tensor) -> int:

        edge_count = int(round(float(global_features[1].detach().cpu()) * 16.0))
        edge_count = max(3, min(edge_count, self.scalar_dofs))
        if self.scalar_dofs % edge_count != 0:

            edge_count = 4 if self.scalar_dofs % 4 == 0 else edge_count
        if self.scalar_dofs % edge_count != 0:
            raise ValueError(f"Cannot infer edge_count from scalar_dofs={self.scalar_dofs}")
        return edge_count

    def _append_normalized_row(self, rows: list[torch.Tensor], row: torch.Tensor) -> None:
        norm = torch.linalg.norm(row)
        rows.append(row / torch.clamp(norm, min=1.0e-7))

    def _scalar_modal_basis_with_groups(
        self,
        global_features: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_count = self._infer_edge_count(global_features)
        bubbles_per_edge = self.scalar_dofs // edge_count - 1
        positions = torch.arange(edge_count, dtype=dtype, device=device) / float(edge_count)
        rows: list[torch.Tensor] = []
        groups: list[int] = []
        next_group = 0

        def append(row: torch.Tensor, group: int | None = None) -> None:
            nonlocal next_group
            norm = torch.linalg.norm(row)
            rows.append(row / torch.clamp(norm, min=1.0e-7))
            if group is None:
                group = next_group
                next_group += 1
            groups.append(int(group))

        if self.modal_basis_mode == "complete-cyclic":
            families = [torch.arange(edge_count, dtype=torch.long, device=device)]
            families.extend(
                torch.tensor(
                    [edge_count + edge * bubbles_per_edge + bubble_order for edge in range(edge_count)],
                    dtype=torch.long,
                    device=device,
                )
                for bubble_order in range(max(0, bubbles_per_edge))
            )
            for indices in families:
                row = torch.zeros(self.scalar_dofs, dtype=dtype, device=device)
                row[indices] = 1.0
                append(row)
                for mode in range(1, (edge_count - 1) // 2 + 1):
                    phase = 2.0 * math.pi * float(mode) * positions
                    pair_group = next_group
                    next_group += 1
                    row = torch.zeros(self.scalar_dofs, dtype=dtype, device=device)
                    row[indices] = torch.cos(phase)
                    append(row, pair_group)
                    row = torch.zeros(self.scalar_dofs, dtype=dtype, device=device)
                    row[indices] = torch.sin(phase)
                    append(row, pair_group)
                if edge_count % 2 == 0:
                    phase = math.pi * torch.arange(edge_count, dtype=dtype, device=device)
                    row = torch.zeros(self.scalar_dofs, dtype=dtype, device=device)
                    row[indices] = torch.cos(phase)
                    append(row)
            if not rows:
                return (
                    torch.zeros((0, self.scalar_dofs), dtype=dtype, device=device),
                    torch.zeros((0,), dtype=torch.long, device=device),
                )
            return torch.stack(rows, dim=0), torch.tensor(groups, dtype=torch.long, device=device)

        vertex_indices = torch.arange(edge_count, dtype=torch.long, device=device)
        max_mode = edge_count // 2
        for mode in range(1, max_mode + 1):
            phase = 2.0 * math.pi * float(mode) * positions
            row = torch.zeros(self.scalar_dofs, dtype=dtype, device=device)
            row[vertex_indices] = torch.cos(phase)
            append(row)
            row = torch.zeros(self.scalar_dofs, dtype=dtype, device=device)
            row[vertex_indices] = torch.sin(phase)
            append(row)
        for bubble_order in range(max(0, bubbles_per_edge)):
            indices = torch.tensor(
                [edge_count + edge * bubbles_per_edge + bubble_order for edge in range(edge_count)],
                dtype=torch.long,
                device=device,
            )
            row = torch.zeros(self.scalar_dofs, dtype=dtype, device=device)
            row[indices] = 1.0
            append(row)
            for mode in range(1, max_mode + 1):
                phase = 2.0 * math.pi * float(mode) * positions
                row = torch.zeros(self.scalar_dofs, dtype=dtype, device=device)
                row[indices] = torch.cos(phase)
                append(row)
                row = torch.zeros(self.scalar_dofs, dtype=dtype, device=device)
                row[indices] = torch.sin(phase)
                append(row)
        if not rows:
            return (
                torch.zeros((0, self.scalar_dofs), dtype=dtype, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )
        return torch.stack(rows, dim=0), torch.tensor(groups, dtype=torch.long, device=device)

    def _scalar_modal_basis(self, global_features: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        basis, _ = self._scalar_modal_basis_with_groups(global_features, dtype=dtype, device=device)
        return basis

    @staticmethod
    def _complete_group_indices(groups: torch.Tensor, usable: int) -> torch.Tensor:


        if usable <= 0 or groups.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=groups.device)
        prefix = groups[:usable]
        selected: list[torch.Tensor] = []
        for group in torch.unique(prefix):
            if torch.count_nonzero(prefix == group) == torch.count_nonzero(groups == group):
                selected.append(torch.nonzero(prefix == group, as_tuple=False).flatten())
        if not selected:
            return torch.zeros((0,), dtype=torch.long, device=groups.device)
        return torch.cat(selected, dim=0).sort().values

    def _vector_modal_basis(
        self,
        global_features: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        basis, _ = self._vector_modal_basis_with_groups(global_features, dtype=dtype, device=device)
        return basis

    def _vector_modal_basis_with_groups(
        self,
        global_features: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scalar_basis, scalar_groups = self._scalar_modal_basis_with_groups(
            global_features, dtype=dtype, device=device
        )
        rows: list[torch.Tensor] = []
        groups: list[int] = []
        for row_index, scalar_row in enumerate(scalar_basis):
            row_x = torch.zeros(self.port_dofs, dtype=dtype, device=device)
            row_y = torch.zeros(self.port_dofs, dtype=dtype, device=device)
            row_x[: self.scalar_dofs] = scalar_row
            row_y[self.scalar_dofs :] = scalar_row
            rows.extend([row_x, row_y])
            groups.extend([2 * int(scalar_groups[row_index]) + 0, 2 * int(scalar_groups[row_index]) + 1])
        if not rows:
            return (
                torch.zeros((0, self.port_dofs), dtype=dtype, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )
        return torch.stack(rows, dim=0), torch.tensor(groups, dtype=torch.long, device=device)

    def _energy_factors_single(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        encoded: torch.Tensor,
        rigid_body_modes: torch.Tensor,
    ) -> torch.Tensor:
        dtype = node_features.dtype
        device = node_features.device
        projector, complement_rows = self._projector_and_complement(
            rigid_body_modes,
            dtype=dtype,
            device=device,
        )

        scalar_features = node_features[: self.scalar_dofs]
        scalar_order = torch.argsort(torch.remainder(scalar_features[:, 16], 1.0))
        scalar_weights = torch.clamp(scalar_features[:, 10], min=1.0e-4)
        scalar_weights = scalar_weights / torch.clamp(torch.mean(scalar_weights), min=1.0e-8)
        sqrt_scalar_weights = torch.sqrt(scalar_weights)
        normalized_dof_weights = torch.cat([scalar_weights, scalar_weights], dim=0)
        normalized_dof_weights = normalized_dof_weights / torch.clamp(normalized_dof_weights.sum(), min=1.0e-8)
        context = (encoded * normalized_dof_weights[:, None]).sum(dim=0)
        context_bank = context[None, :]
        if self.global_context_score_head is not None:
            score = self.global_context_score_head(encoded)
            score = score + torch.log(torch.clamp(normalized_dof_weights, min=1.0e-12))[:, None]
            attention = torch.softmax(score, dim=0)
            attended = attention.transpose(0, 1) @ encoded
            context_bank = torch.cat([context_bank, attended], dim=0)

        rows: list[torch.Tensor] = []

        for radius in range(1, self.stencil_radius + 1):
            radius_scale = torch.exp(self.consistency_log_scale) / math.sqrt(float(radius))
            for pos in range(self.scalar_dofs):
                left = scalar_order[pos]
                right = scalar_order[(pos + radius) % self.scalar_dofs]
                left_x = left
                right_x = right
                left_y = left + self.scalar_dofs
                right_y = right + self.scalar_dofs
                pair_embedding = torch.cat(
                    [
                        encoded[left_x],
                        encoded[right_x],
                        encoded[left_y],
                        encoded[right_y],
                    ],
                    dim=0,
                )
                block = self.local_block_head(pair_embedding).reshape(2, 2)
                if radius == 1:
                    block = block + torch.eye(2, dtype=dtype, device=device)
                learned_scale = torch.sqrt(0.5 * (sqrt_scalar_weights[left] + sqrt_scalar_weights[right]))
                for row_component in range(2):
                    row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                    row[left_x] = radius_scale * learned_scale * block[row_component, 0]
                    row[right_x] = -radius_scale * learned_scale * block[row_component, 0]
                    row[left_y] = radius_scale * learned_scale * block[row_component, 1]
                    row[right_y] = -radius_scale * learned_scale * block[row_component, 1]
                    rows.append(row)

        if self.diagonal_block:
            for index in range(self.scalar_dofs):
                x_index = index
                y_index = index + self.scalar_dofs
                diag_input = torch.cat([encoded[x_index], encoded[y_index], context], dim=0)
                raw = self.diagonal_block_head(diag_input)
                l00 = torch.sqrt(torch.nn.functional.softplus(raw[0]) + self.eps)
                l10 = 0.25 * raw[1]
                l11 = torch.sqrt(torch.nn.functional.softplus(raw[2]) + self.eps)
                weight = sqrt_scalar_weights[index]
                row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                row[x_index] = weight * l00
                rows.append(row)
                row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                row[x_index] = weight * l10
                row[y_index] = weight * l11
                rows.append(row)

        modal_basis, modal_groups = self._vector_modal_basis_with_groups(
            global_features, dtype=dtype, device=device
        )
        if self.symbol_scale_head is not None and modal_basis.numel() and self.symbol_modes > 0:
            usable = min(int(modal_basis.shape[0]), self.symbol_modes)
            kept = self._complete_group_indices(modal_groups, usable)
            raw_scales = self.symbol_scale_head(context)[:usable]
            for group in torch.unique(modal_groups[kept]):
                indices = kept[modal_groups[kept] == group]
                shared_scale = torch.sqrt(
                    torch.nn.functional.softplus(torch.mean(raw_scales[indices])) + self.eps
                )
                for row_index in indices:
                    rows.append(shared_scale * modal_basis[row_index])

        if self.vector_symbol_coupling_head is not None and self.vector_symbol_coupling_modes > 0:
            scalar_modal_basis, scalar_modal_groups = self._scalar_modal_basis_with_groups(
                global_features, dtype=dtype, device=device
            )
            usable = min(int(scalar_modal_basis.shape[0]), self.vector_symbol_coupling_modes)
            if usable > 0:
                raw_coupling = self.vector_symbol_coupling_head(context).reshape(
                    self.vector_symbol_coupling_modes,
                    3,
                )
                scale = torch.exp(self.vector_symbol_coupling_log_scale)
                kept = self._complete_group_indices(scalar_modal_groups, usable)
                for mode_index in kept:
                    scalar_row = scalar_modal_basis[mode_index]
                    group = scalar_modal_groups[mode_index]
                    group_indices = kept[scalar_modal_groups[kept] == group]
                    raw = torch.mean(raw_coupling[group_indices], dim=0)
                    l00 = torch.sqrt(torch.nn.functional.softplus(raw[0]) + self.eps)
                    l10 = 0.25 * raw[1]
                    l11 = torch.sqrt(torch.nn.functional.softplus(raw[2]) + self.eps)
                    row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                    row[: self.scalar_dofs] = scale * l00 * scalar_row
                    rows.append(row)
                    row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                    row[: self.scalar_dofs] = scale * l10 * scalar_row
                    row[self.scalar_dofs :] = scale * l11 * scalar_row
                    rows.append(row)

        if self.corner_block_head is not None and self.corner_rank > 0:
            vertex_indices = torch.nonzero(scalar_features[:, 6] > 0.5, as_tuple=False).flatten()
            position_of = torch.empty_like(scalar_order)
            position_of[scalar_order] = torch.arange(
                self.scalar_dofs, dtype=scalar_order.dtype, device=device
            )
            for vertex in vertex_indices:
                pos = position_of[vertex]
                previous = scalar_order[(pos - 1) % self.scalar_dofs]
                following = scalar_order[(pos + 1) % self.scalar_dofs]
                corner_features = torch.stack(
                    [
                        scalar_features[vertex, 10],
                        scalar_features[vertex, 11],
                        scalar_features[vertex, 12],
                        scalar_features[vertex, 13],
                        scalar_features[vertex, 14],
                        scalar_features[vertex, 15],
                    ]
                )
                raw = self.corner_block_head(
                    torch.cat(
                        [
                            encoded[vertex],
                            encoded[vertex + self.scalar_dofs],
                            corner_features,
                        ],
                        dim=0,
                    )
                ).reshape(self.corner_rank, 6)
                indices = [
                    previous,
                    vertex,
                    following,
                    previous + self.scalar_dofs,
                    vertex + self.scalar_dofs,
                    following + self.scalar_dofs,
                ]
                weights = torch.stack(
                    [
                        sqrt_scalar_weights[previous],
                        sqrt_scalar_weights[vertex],
                        sqrt_scalar_weights[following],
                        sqrt_scalar_weights[previous],
                        sqrt_scalar_weights[vertex],
                        sqrt_scalar_weights[following],
                    ]
                )
                for row_index in range(self.corner_rank):
                    row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                    for local_index, dof_index in enumerate(indices):
                        row[dof_index] = raw[row_index, local_index] * weights[local_index]
                    rows.append(row)

        if self.global_modal_head is not None and self.global_modal_rank > 0:
            if self.global_modal_basis_mode == "physical-project":
                point_context = torch.cat(
                    [
                        encoded,
                        context[None, :].expand(self.port_dofs, -1),
                        global_features[None, :].expand(self.port_dofs, -1),
                    ],
                    dim=1,
                )
                raw = self.global_modal_head(point_context).transpose(0, 1)
                coeff = raw / math.sqrt(float(max(self.port_dofs, 1)))

                global_rows = torch.exp(self.global_modal_log_scale) * (coeff @ projector)
            elif self.global_modal_basis_mode in {"physical-pairwise", "physical-pairwise-symmetric"}:
                row_encoded = encoded[:, None, :].expand(-1, self.port_dofs, -1)
                column_encoded = encoded[None, :, :].expand(self.port_dofs, -1, -1)
                pair_context = context_bank.reshape(-1)[None, None, :].expand(
                    self.port_dofs, self.port_dofs, -1
                )
                pair_global = global_features[None, None, :].expand(
                    self.port_dofs, self.port_dofs, -1
                )
                arc = torch.remainder(node_features[:, 16], 1.0)
                delta = 2.0 * math.pi * (arc[None, :] - arc[:, None])
                component = node_features[:, 17:19]
                same_component = component @ component.transpose(0, 1)
                scalar_index = torch.arange(self.port_dofs, device=device) % self.scalar_dofs
                same_port = (scalar_index[:, None] == scalar_index[None, :]).to(dtype=dtype)
                relative = torch.stack(
                    [torch.sin(delta), torch.cos(delta), same_component, same_port], dim=-1
                )
                if self.pair_geometry_mode == "relative-frame":
                    position = node_features[:, 0:2]
                    tangent = node_features[:, 2:4]
                    normal = node_features[:, 4:6]
                    relative_position = position[None, :, :] - position[:, None, :]
                    tangent_dot = tangent @ tangent.transpose(0, 1)
                    tangent_cross = (
                        tangent[:, None, 0] * tangent[None, :, 1]
                        - tangent[:, None, 1] * tangent[None, :, 0]
                    )
                    relative_tangent = torch.sum(relative_position * tangent[:, None, :], dim=-1)
                    relative_normal = torch.sum(relative_position * normal[:, None, :], dim=-1)
                    relative_distance = torch.linalg.norm(relative_position, dim=-1)
                    relative = torch.cat(
                        [
                            relative,
                            tangent_dot[..., None],
                            tangent_cross[..., None],
                            relative_tangent[..., None],
                            relative_normal[..., None],
                            relative_distance[..., None],
                        ],
                        dim=-1,
                    )
                pair_input = torch.cat(
                    [row_encoded, column_encoded, pair_context, pair_global, relative], dim=-1
                )
                raw_all = self.global_modal_head(pair_input)
                if self.pairwise_gate is not None:
                    expert_weights = torch.softmax(self.pairwise_gate(global_features), dim=-1)
                    raw = torch.sum(raw_all * expert_weights[None, None, :], dim=-1)
                else:
                    raw = raw_all[..., 0]
                if self.global_modal_basis_mode == "physical-pairwise-symmetric":
                    raw = 0.5 * (raw + raw.transpose(0, 1))
                raw = raw + torch.eye(self.port_dofs, dtype=dtype, device=device)
                global_rows = torch.exp(self.global_modal_log_scale) * (raw @ projector)
            else:
                raw = self.global_modal_head(
                    torch.cat([encoded.reshape(-1), global_features], dim=0)
                ).reshape(self.global_modal_rank, self.elastic_modal_dim)
                coeff = raw / math.sqrt(float(max(self.elastic_modal_dim, 1)))
                global_rows = torch.exp(self.global_modal_log_scale) * (coeff @ complement_rows)
            rows.extend(global_rows[row] for row in range(int(global_rows.shape[0])))

        if self.far_weight_head is not None and self.far_scale_head is not None and self.far_rank > 0:
            far_weights = self.far_weight_head(encoded)
            far_weights = far_weights - (normalized_dof_weights[:, None] * far_weights).sum(dim=0, keepdim=True)
            far_scales = torch.sqrt(torch.nn.functional.softplus(self.far_scale_head(context)) + self.eps)
            weighted = far_weights * torch.sqrt(torch.clamp(normalized_dof_weights, min=1.0e-8))[:, None]
            for rank in range(self.far_rank):
                rows.append(far_scales[rank] * weighted[:, rank])

        factors = torch.stack(rows, dim=0)
        return torch.exp(self.log_factor_scale) * factors

    def energy_factors_batch(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
    ) -> list[torch.Tensor]:
        encoded = self.encoded_tokens(node_features, global_features)
        rigid = rigid_body_modes
        if rigid.ndim == 2:
            rigid = rigid.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        return [
            self._energy_factors_single(node_features[i], global_features[i], encoded[i], rigid[i])
            for i in range(int(node_features.shape[0]))
        ]

    def stiffness_batch(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.ndim != 3:
            raise ValueError("node_features must be batched")
        if global_features.ndim != 2:
            raise ValueError("global_features must be batched")
        rigid = rigid_body_modes
        if rigid.ndim == 2:
            rigid = rigid.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        if rigid.ndim != 3 or rigid.shape[0] != node_features.shape[0]:
            raise ValueError(
                "rigid_body_modes must have shape "
                f"[B,{self.port_dofs},m] or [{self.port_dofs},m], got {tuple(rigid_body_modes.shape)}"
            )
        encoded = self.encoded_tokens(node_features, global_features)
        matrices: list[torch.Tensor] = []
        for i in range(int(node_features.shape[0])):
            projector, _ = self._projector_and_complement(
                rigid[i],
                dtype=node_features.dtype,
                device=node_features.device,
            )
            factors = self._energy_factors_single(node_features[i], global_features[i], encoded[i], rigid[i])
            projected = factors @ projector
            matrices.append(projected.transpose(0, 1) @ projected + self.eps * projector)
        return torch.stack(matrices, dim=0)

    def stiffness(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.ndim == 2:
            return self.stiffness_batch(
                node_features.unsqueeze(0),
                global_features.unsqueeze(0) if global_features.ndim == 1 else global_features,
                rigid_body_modes.unsqueeze(0) if rigid_body_modes.ndim == 2 else rigid_body_modes,
            )[0]
        return self.stiffness_batch(node_features, global_features, rigid_body_modes)

    def compile_operator(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:


        if node_features.ndim != 2 or global_features.ndim != 1 or rigid_body_modes.ndim != 2:
            raise ValueError("compile_operator expects one unbatched geometry")
        encoded = self.encoded_tokens(node_features.unsqueeze(0), global_features.unsqueeze(0))[0]
        projector, _ = self._projector_and_complement(
            rigid_body_modes, dtype=node_features.dtype, device=node_features.device
        )
        factors = self._energy_factors_single(
            node_features, global_features, encoded, rigid_body_modes
        )
        return factors @ projector, projector

    def compiled_response(
        self,
        compiled_factor: torch.Tensor,
        projector: torch.Tensor,
        displacement: torch.Tensor,
    ) -> torch.Tensor:


        return (
            compiled_factor.transpose(-1, -2) @ (compiled_factor @ displacement)
            + self.eps * (projector @ displacement)
        )

    def response(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        displacement: torch.Tensor,
        rigid_body_modes: torch.Tensor,
    ) -> torch.Tensor:
        single = node_features.ndim == 2
        if single and displacement.ndim == 1:
            compiled_factor, projector = self.compile_operator(
                node_features, global_features, rigid_body_modes
            )
            return self.compiled_response(compiled_factor, projector, displacement)
        node = node_features.unsqueeze(0) if single else node_features
        global_batch = global_features.unsqueeze(0) if global_features.ndim == 1 else global_features
        rigid = rigid_body_modes.unsqueeze(0) if rigid_body_modes.ndim == 2 else rigid_body_modes
        vector = displacement.unsqueeze(0) if displacement.ndim == 1 else displacement
        encoded = self.encoded_tokens(node, global_batch)
        outputs: list[torch.Tensor] = []
        for index in range(int(node.shape[0])):
            projector, _ = self._projector_and_complement(
                rigid[index], dtype=node.dtype, device=node.device
            )
            factors = self._energy_factors_single(
                node[index], global_batch[index], encoded[index], rigid[index]
            )
            projected = factors @ projector
            outputs.append(
                projected.transpose(0, 1) @ (projected @ vector[index])
                + self.eps * (projector @ vector[index])
            )
        result = torch.stack(outputs, dim=0)
        return result[0] if single else result

    def energy(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        displacement: torch.Tensor,
        rigid_body_modes: torch.Tensor,
    ) -> torch.Tensor:
        response = self.response(node_features, global_features, displacement, rigid_body_modes)
        if displacement.ndim == 1:
            return 0.5 * torch.dot(displacement, response)
        return 0.5 * torch.sum(displacement * response, dim=-1)

    def diagnostics(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
    ) -> dict[str, Any]:
        with torch.no_grad():
            matrix = self.stiffness(node_features, global_features, rigid_body_modes)
            if matrix.ndim == 3:
                matrix = matrix[0]
            rigid = rigid_body_modes
            if rigid.ndim == 3:
                rigid = rigid[0]
            if rigid.shape[0] < rigid.shape[1]:
                rigid = rigid.transpose(0, 1)
            denom = torch.linalg.norm(matrix) + 1.0e-12
            eigvals = torch.linalg.eigvalsh(0.5 * (matrix + matrix.transpose(0, 1)))
            factor_rows = int(
                self.energy_factors_batch(
                    node_features.unsqueeze(0) if node_features.ndim == 2 else node_features,
                    global_features.unsqueeze(0) if global_features.ndim == 1 else global_features,
                    rigid_body_modes.unsqueeze(0) if rigid_body_modes.ndim == 2 else rigid_body_modes,
                )[0].shape[0]
            )
            return {
                "symmetry_residual": float((torch.linalg.norm(matrix - matrix.transpose(0, 1)) / denom).cpu()),
                "min_eigenvalue": float(eigvals.min().cpu()),
                "max_eigenvalue": float(eigvals.max().cpu()),
                "rbm_residual_relative": float((torch.linalg.norm(matrix @ rigid.to(matrix)) / denom).cpu()),
                "factor_rows": factor_rows,
            }

    def forward(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
    ) -> torch.Tensor:
        return self.stiffness(node_features, global_features, rigid_body_modes)

class FixedPortAffineS0MultiplicativeElasticNet(FixedPortStructuredEnergyKernelElasticNet):


    def __init__(
        self,
        *,
        port_dofs: int = 16,
        node_feature_dim: int = 19,
        global_feature_dim: int = 8,
        hidden_dim: int = 192,
        message_layers: int = 2,
        correction_scale: float = 0.05,
        correction_experts: int = 1,
        structured_residual_scale: float = 0.0,
        structured_residual_mode: str = "multiplicative",
        structured_residual_radius: int = 1,
        normal_trace_residual_scale: float = 0.0,
        normal_trace_pair_scale: float | None = None,
        normal_trace_point_scale: float | None = None,
        s0_factor_basis_mode: str = "physical-cholesky",
        geometry_feature_mode: str = "intrinsic-local",
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__(
            port_dofs=port_dofs,
            node_feature_dim=node_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            message_layers=message_layers,
            stencil_radius=1,
            symbol_modes=0,
            global_modal_rank=0,
            geometry_feature_mode=geometry_feature_mode,
            far_rank=0,
            eps=eps,
        )
        if correction_scale < 0.0:
            raise ValueError("correction_scale must be nonnegative")
        if correction_experts < 1:
            raise ValueError("correction_experts must be >= 1")
        if structured_residual_scale < 0.0:
            raise ValueError("structured_residual_scale must be nonnegative")
        if structured_residual_mode not in {"multiplicative", "additive"}:
            raise ValueError("structured_residual_mode must be 'multiplicative' or 'additive'")
        if structured_residual_radius < 1:
            raise ValueError("structured_residual_radius must be >= 1")
        if normal_trace_residual_scale < 0.0:
            raise ValueError("normal_trace_residual_scale must be nonnegative")
        if s0_factor_basis_mode not in {"physical-cholesky", "legacy-eigh-rows"}:
            raise ValueError(
                "s0_factor_basis_mode must be 'physical-cholesky' or 'legacy-eigh-rows'"
            )
        resolved_normal_trace_pair_scale = (
            float(normal_trace_residual_scale)
            if normal_trace_pair_scale is None
            else float(normal_trace_pair_scale)
        )
        resolved_normal_trace_point_scale = (
            float(normal_trace_residual_scale)
            if normal_trace_point_scale is None
            else float(normal_trace_point_scale)
        )
        if resolved_normal_trace_pair_scale < 0.0:
            raise ValueError("normal_trace_pair_scale must be nonnegative")
        if resolved_normal_trace_point_scale < 0.0:
            raise ValueError("normal_trace_point_scale must be nonnegative")
        self.correction_scale = float(correction_scale)
        self.correction_experts = int(correction_experts)
        self.structured_residual_scale = float(structured_residual_scale)
        self.structured_residual_mode = str(structured_residual_mode)
        self.structured_residual_radius = int(structured_residual_radius)
        self.normal_trace_residual_scale = float(normal_trace_residual_scale)
        self.normal_trace_pair_scale = float(resolved_normal_trace_pair_scale)
        self.normal_trace_point_scale = float(resolved_normal_trace_point_scale)
        self.s0_factor_basis_mode = str(s0_factor_basis_mode)
        self.correction_dim = (
            self.port_dofs
            if self.s0_factor_basis_mode == "physical-cholesky"
            else self.elastic_modal_dim
        )
        self.correction_head = nn.Sequential(
            nn.Linear(self.port_dofs * hidden_dim + self.global_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.correction_experts * self.correction_dim * self.correction_dim),
        )
        if self.correction_experts > 1:
            self.correction_gate: nn.Module | None = nn.Sequential(
                nn.Linear(self.global_feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.correction_experts),
            )
        else:
            self.correction_gate = None
        if self.structured_residual_scale > 0.0:
            self.structured_residual_pair_head: nn.Module | None = nn.Sequential(
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 4),
            )
        else:
            self.structured_residual_pair_head = None
        if (
            self.structured_residual_scale > 0.0
            and (self.normal_trace_pair_scale > 0.0 or self.normal_trace_point_scale > 0.0)
        ):
            self.structured_normal_trace_head: nn.Module | None = nn.Sequential(
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.structured_normal_trace_head = None
        final = self.correction_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        if self.structured_residual_pair_head is not None:
            residual_final = self.structured_residual_pair_head[-1]
            if isinstance(residual_final, nn.Linear):
                nn.init.zeros_(residual_final.weight)
                nn.init.zeros_(residual_final.bias)
        if self.structured_normal_trace_head is not None:
            trace_final = self.structured_normal_trace_head[-1]
            if isinstance(trace_final, nn.Linear):
                nn.init.zeros_(trace_final.weight)
                nn.init.zeros_(trace_final.bias)

    def _s0_factor_rows(
        self,
        s0: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projector, _ = self._projector_and_complement(
            rigid_body_modes,
            dtype=dtype,
            device=device,
        )
        baseline = s0.to(dtype=dtype, device=device)
        if baseline.shape != (self.port_dofs, self.port_dofs):
            raise ValueError(f"s0 must have shape {(self.port_dofs, self.port_dofs)}, got {tuple(baseline.shape)}")
        with torch.no_grad():
            projected = projector @ baseline.detach() @ projector
            projected = 0.5 * (projected + projected.transpose(0, 1))
            if self.s0_factor_basis_mode == "physical-cholesky":

                null_projector = torch.eye(
                    self.port_dofs, dtype=dtype, device=device
                ) - projector
                positive_scale = torch.clamp(
                    torch.trace(projected) / float(max(self.elastic_modal_dim, 1)),
                    min=torch.tensor(1.0e-8, dtype=dtype, device=device),
                )
                regularized = projected + positive_scale * null_projector
                regularized = 0.5 * (regularized + regularized.transpose(0, 1))
                cholesky = torch.linalg.cholesky(regularized)
                rows = cholesky.transpose(0, 1) @ projector
                return rows, projector
            eigvals, eigvecs = torch.linalg.eigh(projected)
            scale = torch.clamp(torch.max(torch.abs(eigvals)), min=torch.tensor(1.0e-12, dtype=dtype, device=device))
            keep = eigvals > 1.0e-8 * scale
            if int(torch.count_nonzero(keep).detach().cpu()) == 0:
                rows = torch.zeros((self.elastic_modal_dim, self.port_dofs), dtype=dtype, device=device)
                return rows, projector
            kept_indices = torch.nonzero(keep, as_tuple=False).flatten()

            if int(kept_indices.numel()) > self.elastic_modal_dim:
                kept_indices = kept_indices[-self.elastic_modal_dim :]
            basis = eigvecs[:, kept_indices]
            sqrt_lam = torch.sqrt(torch.clamp(eigvals[kept_indices], min=1.0e-12))
            rows = (basis * sqrt_lam[None, :]).transpose(0, 1)
            if rows.shape[0] < self.elastic_modal_dim:
                pad = torch.zeros(
                    (self.elastic_modal_dim - rows.shape[0], self.port_dofs),
                    dtype=dtype,
                    device=device,
                )
                rows = torch.cat([rows, pad], dim=0)
            elif rows.shape[0] > self.elastic_modal_dim:
                rows = rows[: self.elastic_modal_dim]
        return rows, projector

    def _correction_factor_single(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        encoded: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if s0_factor_rows is None or s0_projector is None:
            l0_rows, projector = self._s0_factor_rows(
                s0,
                rigid_body_modes,
                dtype=node_features.dtype,
                device=node_features.device,
            )
        else:
            l0_rows = s0_factor_rows.to(dtype=node_features.dtype, device=node_features.device)
            projector = s0_projector.to(dtype=node_features.dtype, device=node_features.device)
        raw_all = self.correction_head(torch.cat([encoded.reshape(-1), global_features], dim=0)).reshape(
            self.correction_experts,
            self.correction_dim,
            self.correction_dim,
        )
        if self.correction_gate is not None:
            weights = torch.softmax(self.correction_gate(global_features), dim=-1)
            raw = torch.sum(weights[:, None, None] * raw_all, dim=0)
        else:
            raw = raw_all[0]
        correction = self.correction_scale * raw / math.sqrt(float(max(self.correction_dim, 1)))
        correction = correction + self._structured_modal_update_single(
            node_features,
            encoded,
            l0_rows,
            structured_modal_rows,
            structured_edge_weights,
        )
        identity = torch.eye(self.correction_dim, dtype=node_features.dtype, device=node_features.device)
        factor = (identity + correction) @ l0_rows
        structured_rows = self._structured_residual_rows_single(
            node_features,
            global_features,
            encoded,
        )
        if structured_rows.numel() > 0:
            factor = torch.cat([factor, structured_rows], dim=0)
        return factor, projector

    def _structured_raw_scales_single(
        self,
        node_features: torch.Tensor,
        encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.structured_residual_scale <= 0.0 or self.structured_residual_pair_head is None:
            empty = torch.zeros((0,), dtype=node_features.dtype, device=node_features.device)
            return empty, empty
        scalar_features = node_features[: self.scalar_dofs]
        scalar_order = torch.argsort(torch.remainder(scalar_features[:, 16], 1.0))
        scalar_weights = torch.clamp(scalar_features[:, 10], min=1.0e-4)
        scalar_weights = scalar_weights / torch.clamp(torch.mean(scalar_weights), min=1.0e-8)
        raw_rows: list[torch.Tensor] = []
        edge_rows: list[torch.Tensor] = []
        for radius in range(1, self.structured_residual_radius + 1):
            radius_weight = 1.0 / math.sqrt(float(radius))
            for pos in range(self.scalar_dofs):
                left = scalar_order[pos]
                right = scalar_order[(pos + radius) % self.scalar_dofs]
                left_t = left
                right_t = right
                left_n = left + self.scalar_dofs
                right_n = right + self.scalar_dofs
                pair_embedding = torch.cat(
                    [
                        encoded[left_t],
                        encoded[right_t],
                        encoded[left_n],
                        encoded[right_n],
                    ],
                    dim=0,
                )
                raw_scales = self.structured_residual_pair_head(pair_embedding)
                edge_weight = radius_weight * torch.sqrt(
                    torch.clamp(0.5 * (scalar_weights[left] + scalar_weights[right]), min=1.0e-8)
                )
                for component in range(4):
                    raw_rows.append(raw_scales[component])
                    edge_rows.append(edge_weight)
        if self.structured_normal_trace_head is not None:
            for pos in range(self.scalar_dofs):
                left = scalar_order[pos]
                right = scalar_order[(pos + 1) % self.scalar_dofs]
                left_t = left
                right_t = right
                left_n = left + self.scalar_dofs
                right_n = right + self.scalar_dofs
                pair_embedding = torch.cat(
                    [
                        encoded[left_t],
                        encoded[right_t],
                        encoded[left_n],
                        encoded[right_n],
                    ],
                    dim=0,
                )
                raw_scales = self.structured_normal_trace_head(pair_embedding)
                base_edge_weight = torch.sqrt(
                    torch.clamp(0.5 * (scalar_weights[left] + scalar_weights[right]), min=1.0e-8)
                )
                if self.normal_trace_pair_scale > 0.0:
                    raw_rows.append(raw_scales[0])
                    edge_rows.append(self.normal_trace_pair_scale * base_edge_weight)
                if self.normal_trace_point_scale > 0.0:
                    raw_rows.append(raw_scales[1])
                    edge_rows.append(self.normal_trace_point_scale * base_edge_weight)
        return torch.stack(raw_rows, dim=0), torch.stack(edge_rows, dim=0)

    def _structured_base_rows_and_raw_scales(
        self,
        node_features: torch.Tensor,
        encoded: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.structured_residual_scale <= 0.0 or self.structured_residual_pair_head is None:
            return []
        dtype = node_features.dtype
        device = node_features.device
        scalar_features = node_features[: self.scalar_dofs]
        scalar_order = torch.argsort(torch.remainder(scalar_features[:, 16], 1.0))
        scalar_weights = torch.clamp(scalar_features[:, 10], min=1.0e-4)
        scalar_weights = scalar_weights / torch.clamp(torch.mean(scalar_weights), min=1.0e-8)
        one = torch.ones((), dtype=dtype, device=device)
        out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def normalized(row: torch.Tensor) -> torch.Tensor:
            return row / torch.clamp(torch.linalg.norm(row), min=1.0e-7)

        for radius in range(1, self.structured_residual_radius + 1):
            radius_weight = 1.0 / math.sqrt(float(radius))
            for pos in range(self.scalar_dofs):
                previous = scalar_order[(pos - radius) % self.scalar_dofs]
                left = scalar_order[pos]
                right = scalar_order[(pos + radius) % self.scalar_dofs]
                left_t = left
                right_t = right
                previous_n = previous + self.scalar_dofs
                left_n = left + self.scalar_dofs
                right_n = right + self.scalar_dofs

                pair_embedding = torch.cat(
                    [
                        encoded[left_t],
                        encoded[right_t],
                        encoded[left_n],
                        encoded[right_n],
                    ],
                    dim=0,
                )
                raw_scales = self.structured_residual_pair_head(pair_embedding)
                edge_weight = radius_weight * torch.sqrt(
                    torch.clamp(0.5 * (scalar_weights[left] + scalar_weights[right]), min=1.0e-8)
                )

                row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                row[left_t] = one
                row[right_t] = -one
                out.append((normalized(row), raw_scales[0], edge_weight))

                row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                row[left_n] = one
                row[right_n] = -one
                out.append((normalized(row), raw_scales[1], edge_weight))

                row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                row[previous_n] = one
                row[left_n] = -2.0 * one
                row[right_n] = one
                out.append((normalized(row), raw_scales[2], edge_weight))

                row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                row[left_t] = one
                row[right_t] = -one
                row[left_n] = 0.5 * one
                row[right_n] = -0.5 * one
                out.append((normalized(row), raw_scales[3], edge_weight))

        if self.structured_normal_trace_head is not None:
            for pos in range(self.scalar_dofs):
                left = scalar_order[pos]
                right = scalar_order[(pos + 1) % self.scalar_dofs]
                left_t = left
                right_t = right
                left_n = left + self.scalar_dofs
                right_n = right + self.scalar_dofs
                pair_embedding = torch.cat(
                    [
                        encoded[left_t],
                        encoded[right_t],
                        encoded[left_n],
                        encoded[right_n],
                    ],
                    dim=0,
                )
                raw_scales = self.structured_normal_trace_head(pair_embedding)
                base_edge_weight = torch.sqrt(
                    torch.clamp(0.5 * (scalar_weights[left] + scalar_weights[right]), min=1.0e-8)
                )

                if self.normal_trace_pair_scale > 0.0:
                    row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                    row[left_n] = one
                    row[right_n] = one
                    out.append((normalized(row), raw_scales[0], self.normal_trace_pair_scale * base_edge_weight))

                if self.normal_trace_point_scale > 0.0:
                    row = torch.zeros(self.port_dofs, dtype=dtype, device=device)
                    row[left_n] = one
                    out.append((normalized(row), raw_scales[1], self.normal_trace_point_scale * base_edge_weight))

        return out

    def _structured_modal_update_single(
        self,
        node_features: torch.Tensor,
        encoded: torch.Tensor,
        l0_rows: torch.Tensor,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = node_features.dtype
        device = node_features.device
        if self.structured_residual_mode != "multiplicative":
            return torch.zeros((self.correction_dim, self.correction_dim), dtype=dtype, device=device)
        if structured_modal_rows is not None and structured_edge_weights is not None:
            modal = structured_modal_rows.to(dtype=dtype, device=device)
            edge_weights = structured_edge_weights.to(dtype=dtype, device=device)
            raw_scales, computed_edge_weights = self._structured_raw_scales_single(node_features, encoded)
            del computed_edge_weights
            if raw_scales.numel() == 0:
                return torch.zeros((self.correction_dim, self.correction_dim), dtype=dtype, device=device)
            if modal.shape != (raw_scales.shape[0], self.correction_dim):
                raise ValueError(
                    "structured_modal_rows must have shape "
                    f"[{raw_scales.shape[0]},{self.correction_dim}], got {tuple(modal.shape)}"
                )
            if edge_weights.shape != raw_scales.shape:
                raise ValueError(
                    f"structured_edge_weights must have shape {tuple(raw_scales.shape)}, got {tuple(edge_weights.shape)}"
                )
            denominator = math.sqrt(float(max(int(raw_scales.shape[0]), 1)))
            coefficients = self.structured_residual_scale * edge_weights * torch.tanh(raw_scales) / denominator
            return modal.transpose(0, 1) @ (coefficients[:, None] * modal)
        structured = self._structured_base_rows_and_raw_scales(node_features, encoded)
        if not structured:
            return torch.zeros((self.correction_dim, self.correction_dim), dtype=dtype, device=device)
        with torch.no_grad():
            l0_pinv = torch.linalg.pinv(l0_rows.detach())
            rows = torch.stack([item[0].detach() for item in structured], dim=0)
            modal = rows @ l0_pinv
            modal = modal / torch.clamp(torch.linalg.norm(modal, dim=1, keepdim=True), min=1.0e-7)
        raw_scales = torch.stack([item[1] for item in structured], dim=0)
        edge_weights = torch.stack([item[2] for item in structured], dim=0)
        denominator = math.sqrt(float(max(len(structured), 1)))
        coefficients = self.structured_residual_scale * edge_weights * torch.tanh(raw_scales) / denominator
        return modal.transpose(0, 1) @ (coefficients[:, None] * modal)

    def _structured_residual_rows_single(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        encoded: torch.Tensor,
    ) -> torch.Tensor:


        if (
            self.structured_residual_scale <= 0.0
            or self.structured_residual_pair_head is None
            or self.structured_residual_mode != "additive"
        ):
            return torch.zeros((0, self.port_dofs), dtype=node_features.dtype, device=node_features.device)
        del global_features
        dtype = node_features.dtype
        device = node_features.device
        softplus_one = torch.nn.functional.softplus(torch.zeros((), dtype=dtype, device=device))
        rows: list[torch.Tensor] = []

        def append_row(row: torch.Tensor, scale: torch.Tensor) -> None:
            rows.append(scale * row)

        for row, raw_scale, edge_weight in self._structured_base_rows_and_raw_scales(node_features, encoded):
            scale = (
                self.structured_residual_scale
                * edge_weight
                * torch.nn.functional.softplus(raw_scale)
                / torch.clamp(softplus_one, min=1.0e-8)
            )
            append_row(row, scale)

        if not rows:
            return torch.zeros((0, self.port_dofs), dtype=dtype, device=device)
        return torch.stack(rows, dim=0)

    def stiffness_batch(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_features.ndim != 3:
            raise ValueError("node_features must be batched")
        if global_features.ndim != 2:
            raise ValueError("global_features must be batched")
        rigid = rigid_body_modes
        if rigid.ndim == 2:
            rigid = rigid.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        baseline = s0
        if baseline.ndim == 2:
            baseline = baseline.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        if baseline.ndim != 3 or baseline.shape[0] != node_features.shape[0]:
            raise ValueError(
                "s0 must have shape "
                f"[B,{self.port_dofs},{self.port_dofs}] or [{self.port_dofs},{self.port_dofs}], got {tuple(s0.shape)}"
            )
        factor_cache = s0_factor_rows
        if factor_cache is not None and factor_cache.ndim == 2:
            factor_cache = factor_cache.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        projector_cache = s0_projector
        if projector_cache is not None and projector_cache.ndim == 2:
            projector_cache = projector_cache.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        structured_modal_cache = structured_modal_rows
        if structured_modal_cache is not None and structured_modal_cache.ndim == 2:
            structured_modal_cache = structured_modal_cache.unsqueeze(0).expand(node_features.shape[0], -1, -1)
        structured_edge_cache = structured_edge_weights
        if structured_edge_cache is not None and structured_edge_cache.ndim == 1:
            structured_edge_cache = structured_edge_cache.unsqueeze(0).expand(node_features.shape[0], -1)
        encoded = self.encoded_tokens(node_features, global_features)
        matrices: list[torch.Tensor] = []
        for i in range(int(node_features.shape[0])):
            factors, projector = self._correction_factor_single(
                node_features[i],
                global_features[i],
                encoded[i],
                rigid[i],
                baseline[i],
                None if factor_cache is None else factor_cache[i],
                None if projector_cache is None else projector_cache[i],
                None if structured_modal_cache is None else structured_modal_cache[i],
                None if structured_edge_cache is None else structured_edge_cache[i],
            )
            projected = factors @ projector
            matrices.append(projected.transpose(0, 1) @ projected + self.eps * projector)
        return torch.stack(matrices, dim=0)

    def stiffness(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_features.ndim == 2:
            return self.stiffness_batch(
                node_features.unsqueeze(0),
                global_features.unsqueeze(0) if global_features.ndim == 1 else global_features,
                rigid_body_modes.unsqueeze(0) if rigid_body_modes.ndim == 2 else rigid_body_modes,
                s0.unsqueeze(0) if s0.ndim == 2 else s0,
                None if s0_factor_rows is None else s0_factor_rows.unsqueeze(0) if s0_factor_rows.ndim == 2 else s0_factor_rows,
                None if s0_projector is None else s0_projector.unsqueeze(0) if s0_projector.ndim == 2 else s0_projector,
                (
                    None
                    if structured_modal_rows is None
                    else structured_modal_rows.unsqueeze(0)
                    if structured_modal_rows.ndim == 2
                    else structured_modal_rows
                ),
                (
                    None
                    if structured_edge_weights is None
                    else structured_edge_weights.unsqueeze(0)
                    if structured_edge_weights.ndim == 1
                    else structured_edge_weights
                ),
            )[0]
        return self.stiffness_batch(
            node_features,
            global_features,
            rigid_body_modes,
            s0,
            s0_factor_rows,
            s0_projector,
            structured_modal_rows,
            structured_edge_weights,
        )

    def response(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        displacement: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        matrix = self.stiffness(
            node_features,
            global_features,
            rigid_body_modes,
            s0,
            s0_factor_rows,
            s0_projector,
            structured_modal_rows,
            structured_edge_weights,
        )
        if displacement.ndim == 1:
            return matrix @ displacement
        return torch.einsum("bij,bj->bi", matrix, displacement)

    def energy(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        displacement: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        matrix = self.stiffness(
            node_features,
            global_features,
            rigid_body_modes,
            s0,
            s0_factor_rows,
            s0_projector,
            structured_modal_rows,
            structured_edge_weights,
        )
        if displacement.ndim == 1:
            return 0.5 * displacement @ matrix @ displacement
        return 0.5 * torch.einsum("bi,bij,bj->b", displacement, matrix, displacement)

    def diagnostics(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        with torch.no_grad():
            matrix = self.stiffness(
                node_features,
                global_features,
                rigid_body_modes,
                s0,
                s0_factor_rows,
                s0_projector,
                structured_modal_rows,
                structured_edge_weights,
            )
            if matrix.ndim == 3:
                matrix = matrix[0]
            rigid = rigid_body_modes
            if rigid.ndim == 3:
                rigid = rigid[0]
            if rigid.shape[0] < rigid.shape[1]:
                rigid = rigid.transpose(0, 1)
            denom = torch.linalg.norm(matrix) + 1.0e-12
            eigvals = torch.linalg.eigvalsh(0.5 * (matrix + matrix.transpose(0, 1)))
            encoded = self.encoded_tokens(
                node_features.unsqueeze(0) if node_features.ndim == 2 else node_features,
                global_features.unsqueeze(0) if global_features.ndim == 1 else global_features,
            )
            baseline = s0
            if baseline.ndim == 3:
                baseline = baseline[0]
            factors, _ = self._correction_factor_single(
                node_features if node_features.ndim == 2 else node_features[0],
                global_features if global_features.ndim == 1 else global_features[0],
                encoded[0],
                rigid,
                baseline,
                s0_factor_rows if s0_factor_rows is None or s0_factor_rows.ndim == 2 else s0_factor_rows[0],
                s0_projector if s0_projector is None or s0_projector.ndim == 2 else s0_projector[0],
                (
                    structured_modal_rows
                    if structured_modal_rows is None or structured_modal_rows.ndim == 2
                    else structured_modal_rows[0]
                ),
                (
                    structured_edge_weights
                    if structured_edge_weights is None or structured_edge_weights.ndim == 1
                    else structured_edge_weights[0]
                ),
            )
            return {
                "symmetry_residual": float((torch.linalg.norm(matrix - matrix.transpose(0, 1)) / denom).cpu()),
                "min_eigenvalue": float(eigvals.min().cpu()),
                "max_eigenvalue": float(eigvals.max().cpu()),
                "rbm_residual_relative": float((torch.linalg.norm(matrix @ rigid.to(matrix)) / denom).cpu()),
                "factor_rows": int(factors.shape[0]),
            }

    def forward(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.stiffness(
            node_features,
            global_features,
            rigid_body_modes,
            s0,
            s0_factor_rows,
            s0_projector,
            structured_modal_rows,
            structured_edge_weights,
        )

class FixedPortAffineS0U1BlendElasticNet(nn.Module):















    def __init__(
        self,
        *,
        port_dofs: int = 16,
        node_feature_dim: int = 19,
        global_feature_dim: int = 8,
        hidden_dim: int = 192,
        message_layers: int = 2,
        correction_scale: float = 0.05,
        correction_experts: int = 1,
        structured_residual_scale: float = 0.0,
        structured_residual_mode: str = "multiplicative",
        structured_residual_radius: int = 1,
        normal_trace_residual_scale: float = 0.0,
        normal_trace_pair_scale: float | None = None,
        normal_trace_point_scale: float | None = None,
        s0_factor_basis_mode: str = "physical-cholesky",
        u1_stencil_radius: int = 2,
        u1_symbol_modes: int | None = None,
        u1_corner_rank: int = 2,
        u1_vector_symbol_coupling_modes: int = 0,
        u1_global_modal_rank: int = 8,
        u1_global_modal_basis_mode: str = "physical-project",
        geometry_feature_mode: str = "intrinsic-local",
        u1_modal_basis_mode: str = "complete-cyclic",
        u1_far_rank: int = 8,
        blend_initial_alpha: float = 0.25,
        blend_gate_mode: str = "constant",
        blend_gate_hidden_dim: int = 64,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if blend_gate_mode not in {"constant", "geometry"}:
            raise ValueError("blend_gate_mode must be 'constant' or 'geometry'")
        if not (0.0 < float(blend_initial_alpha) < 1.0):
            raise ValueError("blend_initial_alpha must be in (0,1)")
        self.port_dofs = int(port_dofs)
        self.node_feature_dim = int(node_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.blend_gate_mode = str(blend_gate_mode)
        self.blend_initial_alpha = float(blend_initial_alpha)
        self.eps = float(eps)

        self.affine_model = FixedPortAffineS0MultiplicativeElasticNet(
            port_dofs=port_dofs,
            node_feature_dim=node_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            message_layers=message_layers,
            correction_scale=correction_scale,
            correction_experts=correction_experts,
            structured_residual_scale=structured_residual_scale,
            structured_residual_mode=structured_residual_mode,
            structured_residual_radius=structured_residual_radius,
            normal_trace_residual_scale=normal_trace_residual_scale,
            normal_trace_pair_scale=normal_trace_pair_scale,
            normal_trace_point_scale=normal_trace_point_scale,
            s0_factor_basis_mode=s0_factor_basis_mode,
            geometry_feature_mode=geometry_feature_mode,
            eps=eps,
        )
        self.u1_model = FixedPortStructuredEnergyKernelElasticNet(
            port_dofs=port_dofs,
            node_feature_dim=node_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            message_layers=message_layers,
            stencil_radius=u1_stencil_radius,
            symbol_modes=u1_symbol_modes,
            corner_rank=u1_corner_rank,
            vector_symbol_coupling_modes=u1_vector_symbol_coupling_modes,
            global_modal_rank=u1_global_modal_rank,
            global_modal_basis_mode=u1_global_modal_basis_mode,
            geometry_feature_mode=geometry_feature_mode,
            modal_basis_mode=u1_modal_basis_mode,
            far_rank=u1_far_rank,
            eps=eps,
        )

        initial_logit = math.log(self.blend_initial_alpha / (1.0 - self.blend_initial_alpha))
        if self.blend_gate_mode == "constant":
            self.blend_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
            self.blend_gate = None
        else:
            self.blend_logit = None
            self.blend_gate = nn.Sequential(
                nn.Linear(global_feature_dim, blend_gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(blend_gate_hidden_dim, 1),
            )
            final = self.blend_gate[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.constant_(final.bias, initial_logit)

    def blend_alpha(self, global_features: torch.Tensor) -> torch.Tensor:
        if self.blend_gate_mode == "constant":
            assert self.blend_logit is not None
            return torch.sigmoid(self.blend_logit).to(dtype=global_features.dtype, device=global_features.device)
        assert self.blend_gate is not None
        if global_features.ndim == 1:
            return torch.sigmoid(self.blend_gate(global_features.unsqueeze(0))[0, 0])
        return torch.sigmoid(self.blend_gate(global_features).squeeze(-1))

    def stiffness(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        affine = self.affine_model(
            node_features,
            global_features,
            rigid_body_modes,
            s0,
            s0_factor_rows,
            s0_projector,
            structured_modal_rows,
            structured_edge_weights,
        )
        u1 = self.u1_model(node_features, global_features, rigid_body_modes)
        alpha = self.blend_alpha(global_features).to(dtype=affine.dtype, device=affine.device)
        if affine.ndim == 3:
            if alpha.ndim == 0:
                alpha = alpha.expand(affine.shape[0])
            alpha_view = alpha.reshape(-1, 1, 1)
        else:
            alpha_view = alpha.reshape(()) if alpha.ndim == 0 else alpha.reshape(-1)[0]
        return (1.0 - alpha_view) * affine + alpha_view * u1

    def response(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        displacement: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        matrix = self.stiffness(
            node_features,
            global_features,
            rigid_body_modes,
            s0,
            s0_factor_rows,
            s0_projector,
            structured_modal_rows,
            structured_edge_weights,
        )
        if displacement.ndim == 1:
            return matrix @ displacement
        return torch.einsum("bij,bj->bi", matrix, displacement)

    def energy(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        displacement: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        matrix = self.stiffness(
            node_features,
            global_features,
            rigid_body_modes,
            s0,
            s0_factor_rows,
            s0_projector,
            structured_modal_rows,
            structured_edge_weights,
        )
        if displacement.ndim == 1:
            return 0.5 * displacement @ matrix @ displacement
        return 0.5 * torch.einsum("bi,bij,bj->b", displacement, matrix, displacement)

    def diagnostics(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        with torch.no_grad():
            matrix = self.stiffness(
                node_features,
                global_features,
                rigid_body_modes,
                s0,
                s0_factor_rows,
                s0_projector,
                structured_modal_rows,
                structured_edge_weights,
            )
            if matrix.ndim == 3:
                matrix = matrix[0]
            rigid = rigid_body_modes
            if rigid.ndim == 3:
                rigid = rigid[0]
            if rigid.shape[0] < rigid.shape[1]:
                rigid = rigid.transpose(0, 1)
            denom = torch.linalg.norm(matrix) + 1.0e-12
            eigvals = torch.linalg.eigvalsh(0.5 * (matrix + matrix.transpose(0, 1)))
            alpha = self.blend_alpha(
                global_features if global_features.ndim == 1 else global_features[0]
            )
            return {
                "symmetry_residual": float((torch.linalg.norm(matrix - matrix.transpose(0, 1)) / denom).cpu()),
                "min_eigenvalue": float(eigvals.min().cpu()),
                "max_eigenvalue": float(eigvals.max().cpu()),
                "rbm_residual_relative": float((torch.linalg.norm(matrix @ rigid.to(matrix)) / denom).cpu()),
                "blend_alpha": float(alpha.detach().cpu()),
            }

    def forward(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
        s0_factor_rows: torch.Tensor | None = None,
        s0_projector: torch.Tensor | None = None,
        structured_modal_rows: torch.Tensor | None = None,
        structured_edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.stiffness(
            node_features,
            global_features,
            rigid_body_modes,
            s0,
            s0_factor_rows,
            s0_projector,
            structured_modal_rows,
            structured_edge_weights,
        )

class StructuredDeltaSchurNet(nn.Module):











    def __init__(
        self,
        node_feature_dim: int = 19,
        global_feature_dim: int = 8,
        hidden_dim: int = 160,
        factor_rank: int = 32,
        factor_scale: float = 0.05,
        log_diag_scale: float = 0.25,
        metric_mode: str = "dense-factor",
        dense_factor_scale: float = 2.0,
        modal_kernel_scale: float = 2.0,
        boundary_kernel_scale: float = 1.0,
        hide_absolute_arc: bool = False,
        modal_adapter: bool = False,
        modal_adapter_scale: float = 0.0,
        shape_adapter: bool = False,
        shape_adapter_scale: float = 0.0,
        shape_adapter_experts: int = 4,
        local_pair_adapter: bool = False,
        local_pair_adapter_scale: float = 0.0,
        local_pair_adapter_edge_distance: int = 1,
        rich_pair_features: bool = False,
    ) -> None:
        super().__init__()
        self.node_feature_dim = int(node_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.factor_rank = int(factor_rank)
        self.factor_scale = float(factor_scale)
        self.log_diag_scale = float(log_diag_scale)
        if metric_mode not in {"dense-factor", "low-rank", "modal-kernel", "boundary-kernel"}:
            raise ValueError(f"unknown metric_mode {metric_mode!r}")
        self.metric_mode = str(metric_mode)
        self.dense_factor_scale = float(dense_factor_scale)
        self.modal_kernel_scale = float(modal_kernel_scale)
        self.boundary_kernel_scale = float(boundary_kernel_scale)
        self.hide_absolute_arc = bool(hide_absolute_arc)
        self.modal_adapter = bool(modal_adapter)
        self.modal_adapter_scale = float(modal_adapter_scale)
        self.shape_adapter = bool(shape_adapter)
        self.shape_adapter_scale = float(shape_adapter_scale)
        self.shape_adapter_experts = int(shape_adapter_experts)
        self.local_pair_adapter = bool(local_pair_adapter)
        self.local_pair_adapter_scale = float(local_pair_adapter_scale)
        self.local_pair_adapter_edge_distance = int(local_pair_adapter_edge_distance)
        self.rich_pair_features = bool(rich_pair_features)
        if self.shape_adapter_experts < 1:
            raise ValueError("shape_adapter_experts must be >= 1")
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim + self.global_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.base_pair_feature_dim = 10
        self.rich_pair_extra_dim = 16
        self.pair_feature_dim = self.base_pair_feature_dim + (
            self.rich_pair_extra_dim if self.rich_pair_features else 0
        )
        self.pair_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.message_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_gate = nn.Parameter(torch.tensor(-1.5))
        self.factor_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, factor_rank),
        )
        self.log_diag_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.dense_factor_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.shape_gate = nn.Sequential(
            nn.Linear(self.global_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.shape_adapter_experts),
        )
        self.shape_log_diag_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.shape_adapter_experts),
        )
        self.shape_dense_factor_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.shape_adapter_experts),
        )
        self.local_pair_feature_dim = 4
        self.local_pair_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.pair_feature_dim + self.local_pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.modal_feature_dim = self.node_feature_dim + self.global_feature_dim + 2
        self.modal_pair_feature_dim = 4
        self.modal_encoder = nn.Sequential(
            nn.Linear(self.modal_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.modal_pair_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.modal_pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.modal_kernel_feature_dim = 2 * self.node_feature_dim + self.global_feature_dim + 2
        self.modal_kernel_pair_feature_dim = 6
        self.modal_kernel_encoder = nn.Sequential(
            nn.Linear(self.modal_kernel_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.modal_kernel_log_diag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.modal_kernel_pair_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.modal_kernel_pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.boundary_kernel_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim + self.global_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.boundary_kernel_log_diag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.boundary_kernel_pair_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        final = self.log_diag_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        dense_final = self.dense_factor_head[-1]
        if isinstance(dense_final, nn.Linear):
            nn.init.zeros_(dense_final.weight)
            nn.init.zeros_(dense_final.bias)
        factor_final = self.factor_head[-1]
        if isinstance(factor_final, nn.Linear):

            nn.init.normal_(factor_final.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(factor_final.bias)
        modal_final = self.modal_pair_head[-1]
        if isinstance(modal_final, nn.Linear):

            nn.init.zeros_(modal_final.weight)
            nn.init.zeros_(modal_final.bias)
        modal_kernel_final = self.modal_kernel_pair_head[-1]
        if isinstance(modal_kernel_final, nn.Linear):

            nn.init.zeros_(modal_kernel_final.weight)
            nn.init.zeros_(modal_kernel_final.bias)
        modal_kernel_diag_final = self.modal_kernel_log_diag_head[-1]
        if isinstance(modal_kernel_diag_final, nn.Linear):

            nn.init.zeros_(modal_kernel_diag_final.weight)
            nn.init.zeros_(modal_kernel_diag_final.bias)
        boundary_kernel_pair_final = self.boundary_kernel_pair_head[-1]
        if isinstance(boundary_kernel_pair_final, nn.Linear):

            nn.init.zeros_(boundary_kernel_pair_final.weight)
            nn.init.zeros_(boundary_kernel_pair_final.bias)
        boundary_kernel_diag_final = self.boundary_kernel_log_diag_head[-1]
        if isinstance(boundary_kernel_diag_final, nn.Linear):
            nn.init.zeros_(boundary_kernel_diag_final.weight)
            nn.init.zeros_(boundary_kernel_diag_final.bias)
        shape_log_final = self.shape_log_diag_head[-1]
        if isinstance(shape_log_final, nn.Linear):

            nn.init.zeros_(shape_log_final.weight)
            nn.init.zeros_(shape_log_final.bias)
        shape_dense_final = self.shape_dense_factor_head[-1]
        if isinstance(shape_dense_final, nn.Linear):
            nn.init.zeros_(shape_dense_final.weight)
            nn.init.zeros_(shape_dense_final.bias)
        local_pair_final = self.local_pair_head[-1]
        if isinstance(local_pair_final, nn.Linear):

            nn.init.zeros_(local_pair_final.weight)
            nn.init.zeros_(local_pair_final.bias)

    def shape_adapter_gate(self, global_features: torch.Tensor) -> torch.Tensor:


        logits = self.shape_gate(global_features)
        return torch.softmax(logits, dim=-1)

    def _pair_features(self, node_features: torch.Tensor) -> torch.Tensor:
        count = int(node_features.shape[0])
        xy = node_features[:, 0:2]
        tangent = node_features[:, 2:4]
        normal = node_features[:, 4:6]
        type_flags = node_features[:, 6:8]
        mode = node_features[:, 8:9]
        radial = node_features[:, 9:10]
        edge_ratio = node_features[:, 10:11]
        turn_sin = node_features[:, 14:15]
        turn_cos = node_features[:, 15:16]
        arc = node_features[:, 16:17]
        component = node_features[:, -2:]

        delta = xy.unsqueeze(1) - xy.unsqueeze(0)
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
        inverse_distance = 1.0 / (1.0 + distance)
        tangent_dot = (tangent.unsqueeze(1) * tangent.unsqueeze(0)).sum(dim=-1, keepdim=True)
        normal_dot = (normal.unsqueeze(1) * normal.unsqueeze(0)).sum(dim=-1, keepdim=True)
        component_dot = (component.unsqueeze(1) * component.unsqueeze(0)).sum(dim=-1, keepdim=True)
        type_dot = (type_flags.unsqueeze(1) * type_flags.unsqueeze(0)).sum(dim=-1, keepdim=True)
        radial_delta = torch.abs(radial.unsqueeze(1) - radial.unsqueeze(0))
        arc_delta = torch.abs(arc.unsqueeze(1) - arc.unsqueeze(0))
        arc_delta = torch.minimum(arc_delta, 1.0 - arc_delta)
        features = [
            delta,
            distance,
            inverse_distance,
            tangent_dot,
            normal_dot,
            component_dot,
            type_dot,
            radial_delta,
            arc_delta,
        ]
        if self.rich_pair_features:
            raw_signed_arc = arc.unsqueeze(1) - arc.unsqueeze(0)
            signed_arc = torch.remainder(raw_signed_arc + 0.5, 1.0) - 0.5
            two_pi = float(2.0 * np.pi)
            signed_arc_sin = torch.sin(two_pi * signed_arc)
            signed_arc_cos = torch.cos(two_pi * signed_arc)
            signed_arc2_sin = torch.sin(2.0 * two_pi * signed_arc)
            signed_arc2_cos = torch.cos(2.0 * two_pi * signed_arc)
            tangent_cross = (
                tangent[:, 0:1].unsqueeze(1) * tangent[:, 1:2].unsqueeze(0)
                - tangent[:, 1:2].unsqueeze(1) * tangent[:, 0:1].unsqueeze(0)
            )
            normal_cross = (
                normal[:, 0:1].unsqueeze(1) * normal[:, 1:2].unsqueeze(0)
                - normal[:, 1:2].unsqueeze(1) * normal[:, 0:1].unsqueeze(0)
            )
            tangent_normal = (tangent.unsqueeze(1) * normal.unsqueeze(0)).sum(dim=-1, keepdim=True)
            normal_tangent = (normal.unsqueeze(1) * tangent.unsqueeze(0)).sum(dim=-1, keepdim=True)
            edge_delta = torch.abs(edge_ratio.unsqueeze(1) - edge_ratio.unsqueeze(0))
            edge_mean = 0.5 * (edge_ratio.unsqueeze(1) + edge_ratio.unsqueeze(0))
            turn_dot = turn_sin.unsqueeze(1) * turn_sin.unsqueeze(0) + turn_cos.unsqueeze(1) * turn_cos.unsqueeze(0)
            turn_cross = turn_sin.unsqueeze(1) * turn_cos.unsqueeze(0) - turn_cos.unsqueeze(1) * turn_sin.unsqueeze(0)
            mode_delta = torch.abs(mode.unsqueeze(1) - mode.unsqueeze(0))
            mode_similarity = 1.0 / (1.0 + mode_delta)
            component_cross = (
                component[:, 0:1].unsqueeze(1) * component[:, 1:2].unsqueeze(0)
                - component[:, 1:2].unsqueeze(1) * component[:, 0:1].unsqueeze(0)
            )
            type_cross = (
                type_flags[:, 0:1].unsqueeze(1) * type_flags[:, 1:2].unsqueeze(0)
                + type_flags[:, 1:2].unsqueeze(1) * type_flags[:, 0:1].unsqueeze(0)
            )
            features.extend(
                [
                    signed_arc_sin,
                    signed_arc_cos,
                    signed_arc2_sin,
                    signed_arc2_cos,
                    tangent_cross,
                    normal_cross,
                    tangent_normal,
                    normal_tangent,
                    edge_delta,
                    edge_mean,
                    turn_dot,
                    turn_cross,
                    mode_delta,
                    mode_similarity,
                    component_cross,
                    type_cross,
                ]
            )
        pair = torch.cat(features, dim=-1)
        if pair.shape != (count, count, self.pair_feature_dim):
            raise RuntimeError(f"Internal pair feature shape mismatch: {tuple(pair.shape)}")
        return pair

    def _local_pair_mask_features(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:







        count = int(node_features.shape[0])
        if count % 2 != 0:
            raise ValueError(f"expected component-major vector DOFs, got count={count}")
        scalar_count = count // 2
        edge_count = max(1, int(round(float(global_features[1].detach().cpu()) * 16.0)))
        edge_count = min(edge_count, scalar_count)
        if scalar_count % edge_count != 0:
            raise ValueError(
                f"cannot infer bubbles_per_edge from scalar_count={scalar_count}, edge_count={edge_count}"
            )
        bubbles = max(0, scalar_count // edge_count - 1)
        scalar_index = torch.arange(scalar_count, dtype=torch.long, device=node_features.device)
        if bubbles == 0:
            scalar_edge = scalar_index % edge_count
        else:
            bubble_edge = torch.div(torch.clamp(scalar_index - edge_count, min=0), bubbles, rounding_mode="floor")
            scalar_edge = torch.where(scalar_index < edge_count, scalar_index, bubble_edge)
            scalar_edge = torch.clamp(scalar_edge, min=0, max=edge_count - 1)
        dof_edge = torch.cat([scalar_edge, scalar_edge], dim=0)
        distance = torch.abs(dof_edge.unsqueeze(1) - dof_edge.unsqueeze(0))
        circular = torch.minimum(distance, int(edge_count) - distance)
        if int(self.local_pair_adapter_edge_distance) < 0:
            local_mask = torch.ones_like(circular, dtype=node_features.dtype)
        else:
            local_mask = (circular <= max(0, int(self.local_pair_adapter_edge_distance))).to(node_features.dtype)
        denom = max(float(edge_count), 1.0)
        distance_norm = circular.to(node_features.dtype) / denom
        same_flag = (circular == 0).to(node_features.dtype)
        near_flag = (circular == 1).to(node_features.dtype)
        active_flag = local_mask
        features = torch.stack([distance_norm, same_flag, near_flag, active_flag], dim=-1)
        if features.shape != (count, count, self.local_pair_feature_dim):
            raise RuntimeError(f"Internal local pair feature shape mismatch: {tuple(features.shape)}")
        return local_mask, features

    @staticmethod
    def _component_major_from_blocks(blocks: torch.Tensor) -> torch.Tensor:


        if blocks.ndim != 4 or blocks.shape[-2:] != (2, 2):
            raise ValueError(f"expected blocks [M,M,2,2], got {tuple(blocks.shape)}")
        scalar_count = int(blocks.shape[0])
        if int(blocks.shape[1]) != scalar_count:
            raise ValueError(f"expected square block grid, got {tuple(blocks.shape)}")
        full = torch.zeros(
            (2 * scalar_count, 2 * scalar_count),
            dtype=blocks.dtype,
            device=blocks.device,
        )
        full[:scalar_count, :scalar_count] = blocks[:, :, 0, 0]
        full[:scalar_count, scalar_count:] = blocks[:, :, 0, 1]
        full[scalar_count:, :scalar_count] = blocks[:, :, 1, 0]
        full[scalar_count:, scalar_count:] = blocks[:, :, 1, 1]
        return full

    def boundary_kernel_metric(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:


        count = int(node_features.shape[0])
        if count % 2 != 0:
            raise ValueError(f"expected component-major vector DOFs, got count={count}")
        scalar_count = count // 2
        scalar_features = node_features[:scalar_count]
        scalar_input = scalar_features
        if self.hide_absolute_arc:
            scalar_input = scalar_features.clone()
            scalar_input[:, 16] = 0.0
        global_rows = global_features.unsqueeze(0).expand(scalar_count, -1)
        encoded = self.boundary_kernel_encoder(torch.cat([scalar_input, global_rows], dim=-1))
        pair_features = self._pair_features(scalar_features)
        gate_left = encoded.unsqueeze(1).expand(scalar_count, scalar_count, -1)
        gate_right = encoded.unsqueeze(0).expand(scalar_count, scalar_count, -1)
        attention = torch.softmax(
            self.pair_gate(torch.cat([gate_left, gate_right, pair_features], dim=-1)).squeeze(-1),
            dim=1,
        )
        message = attention @ self.message_value(encoded)
        updated = encoded + torch.sigmoid(self.message_gate) * self.message_update(
            torch.cat([encoded, message], dim=-1)
        )
        left = updated.unsqueeze(1).expand(scalar_count, scalar_count, -1)
        right = updated.unsqueeze(0).expand(scalar_count, scalar_count, -1)
        local_blocks = self.boundary_kernel_pair_head(torch.cat([left, right, pair_features], dim=-1))
        local_blocks = local_blocks.reshape(scalar_count, scalar_count, 2, 2)
        local_blocks = float(self.boundary_kernel_scale) * local_blocks / max(float(scalar_count) ** 0.5, 1.0)
        eye_scalar = torch.eye(scalar_count, dtype=node_features.dtype, device=node_features.device)
        local_blocks = local_blocks * (1.0 - eye_scalar).view(scalar_count, scalar_count, 1, 1)

        log_diag = self.log_diag_scale * torch.tanh(self.boundary_kernel_log_diag_head(updated))
        diag_local = torch.zeros(
            (scalar_count, 2, 2),
            dtype=node_features.dtype,
            device=node_features.device,
        )
        diag_local[:, 0, 0] = torch.exp(0.5 * log_diag[:, 0])
        diag_local[:, 1, 1] = torch.exp(0.5 * log_diag[:, 1])
        local_blocks = local_blocks + eye_scalar.view(scalar_count, scalar_count, 1, 1) * diag_local.unsqueeze(1)

        tangent = scalar_features[:, 2:4]
        normal = scalar_features[:, 4:6]
        left_frame = torch.stack([tangent, normal], dim=-1)
        right_frame_t = left_frame.transpose(1, 2)
        global_blocks = torch.einsum("iac,ijcd,jdb->ijab", left_frame, local_blocks, right_frame_t)
        factor = self._component_major_from_blocks(global_blocks)
        return torch_symmetrize(factor @ factor.transpose(0, 1))

    def correction_metric(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
    ) -> torch.Tensor:
        if node_features.ndim != 2 or node_features.shape[1] != self.node_feature_dim:
            raise ValueError(f"Expected node features [N,{self.node_feature_dim}], got {tuple(node_features.shape)}")
        if global_features.shape != (self.global_feature_dim,):
            raise ValueError(f"Expected global features [{self.global_feature_dim}], got {tuple(global_features.shape)}")
        count = int(node_features.shape[0])
        if self.metric_mode == "boundary-kernel":
            metric = self.boundary_kernel_metric(node_features, global_features)
            p = torch_rbm_projector(rigid_body_modes)
            eye = torch.eye(count, dtype=metric.dtype, device=metric.device)
            return p @ metric @ p + (eye - p) * 0.0
        global_rows = global_features.unsqueeze(0).expand(count, -1)
        node_input = node_features
        if self.hide_absolute_arc:

            node_input = node_features.clone()
            node_input[:, 16] = 0.0
        local_encoded = self.node_encoder(torch.cat([node_input, global_rows], dim=-1))
        pair_features = self._pair_features(node_features)
        left = local_encoded.unsqueeze(1).expand(count, count, -1)
        right = local_encoded.unsqueeze(0).expand(count, count, -1)
        attention = torch.softmax(self.pair_gate(torch.cat([left, right, pair_features], dim=-1)).squeeze(-1), dim=1)
        message = attention @ self.message_value(local_encoded)
        updated = local_encoded + torch.sigmoid(self.message_gate) * self.message_update(
            torch.cat([local_encoded, message], dim=-1)
        )
        context = updated.mean(dim=0, keepdim=True).expand(count, -1)
        combined = torch.cat([local_encoded, updated, context], dim=-1)
        log_diag = self.log_diag_scale * torch.tanh(self.log_diag_head(combined).squeeze(-1))
        shape_gate = None
        if self.shape_adapter and float(self.shape_adapter_scale) != 0.0:
            shape_gate = self.shape_adapter_gate(global_features)
            shape_log = self.shape_log_diag_head(combined)
            shape_log = (shape_log * shape_gate.unsqueeze(0)).sum(dim=-1)
            log_diag = log_diag + float(self.shape_adapter_scale) * torch.tanh(shape_log)
        diagonal = torch.exp(log_diag)
        if self.metric_mode == "modal-kernel":
            raise ValueError("modal-kernel is a reduced C_theta mode; use reduced_correction_metric()")
        if self.metric_mode == "dense-factor":
            updated_left = updated.unsqueeze(1).expand(count, count, -1)
            updated_right = updated.unsqueeze(0).expand(count, count, -1)
            dense_input = torch.cat([updated_left, updated_right, pair_features], dim=-1)
            residual = self.dense_factor_head(dense_input).squeeze(-1)
            residual = self.dense_factor_scale * residual / max(float(count) ** 0.5, 1.0)
            if self.shape_adapter and float(self.shape_adapter_scale) != 0.0:
                if shape_gate is None:
                    shape_gate = self.shape_adapter_gate(global_features)
                shape_residual = self.shape_dense_factor_head(dense_input)
                shape_residual = (shape_residual * shape_gate.view(1, 1, -1)).sum(dim=-1)
                residual = residual + float(self.shape_adapter_scale) * shape_residual / max(
                    float(count) ** 0.5,
                    1.0,
                )
            if self.local_pair_adapter and float(self.local_pair_adapter_scale) != 0.0:
                local_mask, local_pair_features = self._local_pair_mask_features(node_features, global_features)
                local_input = torch.cat([updated_left, updated_right, pair_features, local_pair_features], dim=-1)
                local_residual = self.local_pair_head(local_input).squeeze(-1)
                residual = residual + float(self.local_pair_adapter_scale) * local_mask * local_residual / max(
                    float(count) ** 0.5,
                    1.0,
                )
            factor_matrix = torch.diag(torch.sqrt(diagonal)) + residual
            metric = factor_matrix @ factor_matrix.transpose(0, 1)
        elif self.metric_mode == "low-rank":
            factor = self.factor_scale * self.factor_head(combined) / max(float(self.factor_rank) ** 0.5, 1.0)
            metric = torch.diag(diagonal) + factor @ factor.transpose(0, 1)
        else:
            raise ValueError(f"unknown metric_mode {self.metric_mode!r}")
        p = torch_rbm_projector(rigid_body_modes)
        eye = torch.eye(count, dtype=metric.dtype, device=metric.device)

        return p @ metric @ p + (eye - p) * 0.0

    def _s0_modal_basis(
        self,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


        p = torch_rbm_projector(rigid_body_modes)
        s0p = torch_symmetrize(p @ s0 @ p)
        eigvals, eigvecs = torch.linalg.eigh(s0p)
        scale = torch.clamp(torch.max(torch.abs(eigvals)), min=1.0e-10)
        keep = eigvals > (1.0e-8 * scale)
        if not bool(torch.any(keep)):
            empty = eigvecs[:, :0]
            return p, eigvals[:0], empty, empty
        positive = eigvals[keep]
        basis = torch_canonicalize_modal_basis(eigvecs[:, keep])
        sqrt_lam = torch.sqrt(torch.clamp(positive, min=1.0e-10))
        l0 = basis * sqrt_lam.unsqueeze(0)
        return p, positive, basis, l0

    def modal_kernel_metric(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        basis: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:


        mode_count = int(basis.shape[1])
        if mode_count == 0:
            return torch.zeros((0, 0), dtype=node_features.dtype, device=node_features.device)
        node_input = node_features
        if self.hide_absolute_arc:
            node_input = node_features.clone()
            node_input[:, 16] = 0.0
        signed_mode_node = basis.transpose(0, 1) @ node_input
        localization_weights = torch.abs(basis.transpose(0, 1))
        localization_weights = localization_weights / torch.clamp(
            localization_weights.sum(dim=1, keepdim=True),
            min=1.0e-12,
        )
        abs_mode_node = localization_weights @ node_input
        scale = torch.clamp(torch.mean(torch.abs(eigenvalues)), min=1.0e-12)
        log_lambda = torch.log(torch.clamp(eigenvalues / scale, min=1.0e-8)).unsqueeze(-1)
        if mode_count == 1:
            mode_index = torch.zeros((mode_count, 1), dtype=node_features.dtype, device=node_features.device)
        else:
            mode_index = torch.linspace(
                0.0,
                1.0,
                mode_count,
                dtype=node_features.dtype,
                device=node_features.device,
            ).unsqueeze(-1)
        global_rows = global_features.unsqueeze(0).expand(mode_count, -1)
        encoded = self.modal_kernel_encoder(
            torch.cat([signed_mode_node, abs_mode_node, global_rows, log_lambda, mode_index], dim=-1)
        )
        left = encoded.unsqueeze(1).expand(mode_count, mode_count, -1)
        right = encoded.unsqueeze(0).expand(mode_count, mode_count, -1)
        log_left = log_lambda.unsqueeze(1).expand(mode_count, mode_count, -1)
        log_right = log_lambda.unsqueeze(0).expand(mode_count, mode_count, -1)
        index_left = mode_index.unsqueeze(1).expand(mode_count, mode_count, -1)
        index_right = mode_index.unsqueeze(0).expand(mode_count, mode_count, -1)
        delta = torch.abs(log_left - log_right)
        mean = 0.5 * (log_left + log_right)
        inv_delta = 1.0 / (1.0 + delta)
        eye_flag = torch.eye(mode_count, dtype=node_features.dtype, device=node_features.device).unsqueeze(-1)
        index_delta = torch.abs(index_left - index_right)
        index_mean = 0.5 * (index_left + index_right)
        pair = torch.cat([delta, mean, inv_delta, eye_flag, index_delta, index_mean], dim=-1)
        log_diag = self.log_diag_scale * torch.tanh(self.modal_kernel_log_diag_head(encoded).squeeze(-1))
        diagonal_factor = torch.diag(torch.exp(0.5 * log_diag))
        residual = self.modal_kernel_pair_head(torch.cat([left, right, pair], dim=-1)).squeeze(-1)
        residual = float(self.modal_kernel_scale) * residual / max(float(mode_count) ** 0.5, 1.0)
        eye = torch.eye(mode_count, dtype=node_features.dtype, device=node_features.device)
        residual = residual * (1.0 - eye)
        factor = diagonal_factor + residual
        return torch_symmetrize(factor @ factor.transpose(0, 1))

    def _reduced_correction_metric_from_basis(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        projector: torch.Tensor,
        basis: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:


        if self.metric_mode == "modal-kernel":
            c_reduced = self.modal_kernel_metric(node_features, global_features, basis, eigenvalues)
        else:
            c_full = self.correction_metric(node_features, global_features, rigid_body_modes)
            c_reduced = torch_symmetrize(basis.transpose(0, 1) @ projector @ c_full @ projector @ basis)
        if self.modal_adapter and float(self.modal_adapter_scale) != 0.0:
            adapter = self.modal_adapter_factor(node_features, global_features, basis, eigenvalues)
            c_reduced = torch_symmetrize(adapter @ c_reduced @ adapter.transpose(0, 1))
        return c_reduced

    def reduced_correction_metric_from_basis(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        projector: torch.Tensor,
        basis: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:







        return self._reduced_correction_metric_from_basis(
            node_features,
            global_features,
            rigid_body_modes,
            projector,
            basis,
            eigenvalues,
        )

    def reduced_correction_metric(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
    ) -> torch.Tensor:


        p, eigvals, basis, _ = self._s0_modal_basis(rigid_body_modes, s0)
        if int(basis.shape[1]) == 0:
            return torch.zeros((0, 0), dtype=s0.dtype, device=s0.device)
        return self._reduced_correction_metric_from_basis(
            node_features,
            global_features,
            rigid_body_modes,
            p,
            basis,
            eigvals,
        )

    def modal_adapter_factor(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        basis: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> torch.Tensor:


        mode_count = int(basis.shape[1])
        if mode_count == 0:
            return torch.zeros((0, 0), dtype=node_features.dtype, device=node_features.device)
        weights = torch.abs(basis.transpose(0, 1))
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0e-12)
        mode_node = weights @ node_features
        scale = torch.clamp(torch.mean(torch.abs(eigenvalues)), min=1.0e-12)
        log_lambda = torch.log(torch.clamp(eigenvalues / scale, min=1.0e-8)).unsqueeze(-1)
        if mode_count == 1:
            mode_index = torch.zeros((mode_count, 1), dtype=node_features.dtype, device=node_features.device)
        else:
            mode_index = torch.linspace(
                0.0,
                1.0,
                mode_count,
                dtype=node_features.dtype,
                device=node_features.device,
            ).unsqueeze(-1)
        global_rows = global_features.unsqueeze(0).expand(mode_count, -1)
        encoded = self.modal_encoder(torch.cat([mode_node, global_rows, log_lambda, mode_index], dim=-1))
        left = encoded.unsqueeze(1).expand(mode_count, mode_count, -1)
        right = encoded.unsqueeze(0).expand(mode_count, mode_count, -1)
        log_left = log_lambda.unsqueeze(1).expand(mode_count, mode_count, -1)
        log_right = log_lambda.unsqueeze(0).expand(mode_count, mode_count, -1)
        delta = torch.abs(log_left - log_right)
        mean = 0.5 * (log_left + log_right)
        inv_delta = 1.0 / (1.0 + delta)
        eye_flag = torch.eye(mode_count, dtype=node_features.dtype, device=node_features.device).unsqueeze(-1)
        pair = torch.cat([delta, mean, inv_delta, eye_flag], dim=-1)
        residual = self.modal_pair_head(torch.cat([left, right, pair], dim=-1)).squeeze(-1)
        residual = float(self.modal_adapter_scale) * residual / max(float(mode_count) ** 0.5, 1.0)
        eye = torch.eye(mode_count, dtype=node_features.dtype, device=node_features.device)
        return eye + residual

    def forward(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        rigid_body_modes: torch.Tensor,
        s0: torch.Tensor,
    ) -> torch.Tensor:
        p, eigvals, basis, l0 = self._s0_modal_basis(rigid_body_modes, s0)
        if int(basis.shape[1]) == 0:
            return torch.zeros_like(s0)
        c_reduced = self._reduced_correction_metric_from_basis(
            node_features,
            global_features,
            rigid_body_modes,
            p,
            basis,
            eigvals,
        )
        return torch_symmetrize(p @ l0 @ c_reduced @ l0.transpose(0, 1) @ p)

@dataclass(frozen=True)
class StructuredDeltaSchurSurrogate:
    model: StructuredDeltaSchurNet
    device: str = "cpu"
    metadata: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "StructuredDeltaSchurSurrogate":
        payload = torch.load(Path(path), map_location=device)
        model = StructuredDeltaSchurNet(
            node_feature_dim=int(payload["node_feature_dim"]),
            global_feature_dim=int(payload["global_feature_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            factor_rank=int(payload["factor_rank"]),
            factor_scale=float(payload.get("factor_scale", 0.05)),
            log_diag_scale=float(payload.get("log_diag_scale", 0.25)),
            metric_mode=str(payload.get("metric_mode", "low-rank")),
            dense_factor_scale=float(payload.get("dense_factor_scale", 2.0)),
            modal_kernel_scale=float(payload.get("modal_kernel_scale", 2.0)),
            boundary_kernel_scale=float(payload.get("boundary_kernel_scale", 1.0)),
            hide_absolute_arc=bool(payload.get("hide_absolute_arc", False)),
            modal_adapter=bool(payload.get("modal_adapter", False)),
            modal_adapter_scale=float(payload.get("modal_adapter_scale", 0.0)),
            shape_adapter=bool(payload.get("shape_adapter", False)),
            shape_adapter_scale=float(payload.get("shape_adapter_scale", 0.0)),
            shape_adapter_experts=int(payload.get("shape_adapter_experts", 4)),
            local_pair_adapter=bool(payload.get("local_pair_adapter", False)),
            local_pair_adapter_scale=float(payload.get("local_pair_adapter_scale", 0.0)),
            local_pair_adapter_edge_distance=int(payload.get("local_pair_adapter_edge_distance", 1)),
            rich_pair_features=bool(payload.get("rich_pair_features", False)),
        ).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=False)
        model.eval()
        metadata = dict(payload.get("metadata", {}))
        for key in (
            "s0_kind",
            "s0_scale",
            "s0_segments_per_edge",
            "s0_coefficients",
            "component_frame",
            "canonical_frame",
        ):
            if key in payload:
                metadata.setdefault(key, payload[key])
        return cls(model=model, device=device, metadata=metadata)

    def s0_from_polygon(
        self,
        vertices: np.ndarray,
        bubbles_per_edge: int,
        poisson: float,
        rigid_body_modes: np.ndarray,
        node_features: np.ndarray | None = None,
    ) -> np.ndarray:
        metadata = dict(self.metadata or {})
        s0_kind = str(metadata.get("s0_kind", "affine-vem"))
        if s0_kind == "affine-vem":
            coefficients = metadata.get("s0_coefficients")
            if coefficients is None:
                raise ValueError("affine-vem surrogate metadata is missing s0_coefficients")
            coeff = np.asarray(coefficients, dtype=np.float64).reshape(2)
            return direct_affine_vem_s0(
                vertices,
                int(bubbles_per_edge),
                float(poisson),
                rigid_body_modes,
                consistency_coefficient=float(coeff[0]),
                stabilization_coefficient=float(coeff[1]),
                segments_per_edge=int(metadata.get("s0_segments_per_edge", 8)),
            )
        if s0_kind == "graph-steklov":
            if node_features is None:
                from src.bnem.elastic_dataset import polygon_operator_features

                node_features, _ = polygon_operator_features(vertices, int(bubbles_per_edge), float(poisson))
            port_points = port_points_from_node_features(node_features)
            return direct_geometry_s0(port_points, rigid_body_modes) * float(metadata.get("s0_scale", 1.0))
        raise ValueError(f"unknown s0_kind {s0_kind!r}")

    def stiffness_from_polygon(
        self,
        vertices: np.ndarray,
        bubbles_per_edge: int,
        poisson: float,
        rigid_body_modes: np.ndarray,
    ) -> np.ndarray:
        metadata = dict(self.metadata or {})
        if str(metadata.get("component_frame", "global")) != "global":
            raise NotImplementedError("stiffness_from_polygon currently supports global component-frame experts")
        from src.bnem.elastic_dataset import polygon_operator_features

        node_features, global_features = polygon_operator_features(vertices, int(bubbles_per_edge), float(poisson))
        s0 = self.s0_from_polygon(
            vertices,
            int(bubbles_per_edge),
            float(poisson),
            rigid_body_modes,
            node_features=node_features,
        )
        return self.stiffness_from_features(node_features, global_features, rigid_body_modes, s0)

    def stiffness_from_features(
        self,
        node_features: np.ndarray,
        global_features: np.ndarray,
        rigid_body_modes: np.ndarray,
        s0: np.ndarray,
    ) -> np.ndarray:
        with torch.no_grad():
            pred = self.model(
                torch.as_tensor(node_features, dtype=torch.float32, device=self.device),
                torch.as_tensor(global_features, dtype=torch.float32, device=self.device),
                torch.as_tensor(rigid_body_modes, dtype=torch.float32, device=self.device),
                torch.as_tensor(s0, dtype=torch.float32, device=self.device),
            )
        p = rbm_projector(rigid_body_modes)
        return symmetrize(p @ np.asarray(pred.detach().cpu(), dtype=np.float64) @ p)

@dataclass(frozen=True)
class EdgeCountGatedDeltaSchurSurrogate:









    experts: dict[str, StructuredDeltaSchurSurrogate]
    edge_count_to_expert: dict[int, str]
    default_expert: str
    shape_rules: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] | None = None

    @classmethod
    def load(
        cls,
        expert_paths: dict[str, str | Path],
        *,
        edge_count_to_expert: dict[int, str],
        default_expert: str,
        shape_rules: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        device: str = "cpu",
        metadata: dict[str, Any] | None = None,
    ) -> "EdgeCountGatedDeltaSchurSurrogate":
        if default_expert not in expert_paths:
            raise ValueError(f"default expert {default_expert!r} is not in expert_paths")
        for edge_count, expert_name in edge_count_to_expert.items():
            if expert_name not in expert_paths:
                raise ValueError(f"edge_count {edge_count} maps to unknown expert {expert_name!r}")
        normalized_shape_rules: list[dict[str, Any]] = []
        for rule in shape_rules or ():
            normalized = dict(rule)
            expert_name = str(normalized.get("expert", ""))
            if expert_name not in expert_paths:
                raise ValueError(f"shape rule maps to unknown expert {expert_name!r}: {rule!r}")
            if "edge_count" in normalized:
                normalized["edge_count"] = int(normalized["edge_count"])
            normalized_shape_rules.append(normalized)
        experts = {
            name: StructuredDeltaSchurSurrogate.load(path, device=device)
            for name, path in expert_paths.items()
        }
        return cls(
            experts=experts,
            edge_count_to_expert={int(k): str(v) for k, v in edge_count_to_expert.items()},
            default_expert=str(default_expert),
            shape_rules=tuple(normalized_shape_rules),
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def edge_count_from_global_features(global_features: np.ndarray) -> int:
        features = np.asarray(global_features, dtype=np.float64).reshape(-1)
        if features.shape[0] < 2:
            raise ValueError("global_features must contain edge_count / 16 at index 1")
        return int(round(float(features[1]) * 16.0))

    def select_expert_name(self, global_features: np.ndarray) -> str:
        edge_count = self.edge_count_from_global_features(global_features)
        return self.edge_count_to_expert.get(edge_count, self.default_expert)

    @staticmethod
    def _shape_rule_matches(rule: dict[str, Any], edge_count: int, diagnostics: dict[str, float]) -> bool:
        if "edge_count" in rule and int(rule["edge_count"]) != int(edge_count):
            return False
        for key, value in diagnostics.items():
            if f"{key}_lt" in rule and not (float(value) < float(rule[f"{key}_lt"])):
                return False
            if f"{key}_le" in rule and not (float(value) <= float(rule[f"{key}_le"])):
                return False
            if f"{key}_gt" in rule and not (float(value) > float(rule[f"{key}_gt"])):
                return False
            if f"{key}_ge" in rule and not (float(value) >= float(rule[f"{key}_ge"])):
                return False
        return True

    def select_expert_name_for_polygon(self, vertices: np.ndarray) -> str:
        points = np.asarray(vertices, dtype=np.float64)
        edge_count = int(points.shape[0])
        if self.shape_rules:
            diagnostics = polygon_shape_diagnostics(points)
            for rule in self.shape_rules:
                if self._shape_rule_matches(rule, edge_count, diagnostics):
                    return str(rule["expert"])
        return self.edge_count_to_expert.get(edge_count, self.default_expert)

    def stiffness_from_features(
        self,
        node_features: np.ndarray,
        global_features: np.ndarray,
        rigid_body_modes: np.ndarray,
        s0: np.ndarray,
    ) -> np.ndarray:
        expert_name = self.select_expert_name(global_features)
        return self.experts[expert_name].stiffness_from_features(
            node_features,
            global_features,
            rigid_body_modes,
            s0,
        )

    def stiffness_from_polygon(
        self,
        vertices: np.ndarray,
        bubbles_per_edge: int,
        poisson: float,
        rigid_body_modes: np.ndarray,
    ) -> np.ndarray:
        expert_name = self.select_expert_name_for_polygon(vertices)
        return self.experts[expert_name].stiffness_from_polygon(
            vertices,
            int(bubbles_per_edge),
            float(poisson),
            rigid_body_modes,
        )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.bnem.convergence import validate_convergence_certificate
from src.bnem.dataset import find_patch_npz, patch_matches_filters
from src.bnem.polygon_geometry import (
    BOUNDARY_ARCLENGTH_SCHEMA,
    LEGACY_VERTEX_RMS_SCHEMA,
    boundary_arclength_centroid_rms,
    stable_unit_tangent_average,
    validate_geometry_normalization_schema,
)

@dataclass(frozen=True)
class ElasticPatchSample:
    path: Path
    eta: np.ndarray
    stiffness: np.ndarray
    stiffness_scale: float
    rigid_body_modes: np.ndarray
    port_dofs: int
    dataset_role: str
    convergence_certificate: Path
    convergence: dict[str, Any]

@dataclass(frozen=True)
class PolygonElasticSample:
    path: Path
    vertices: np.ndarray
    node_features: np.ndarray
    global_features: np.ndarray
    stiffness: np.ndarray
    stiffness_scale: float
    rigid_body_modes: np.ndarray
    edge_count: int
    bubbles_per_edge: int
    dataset_role: str
    convergence_certificate: Path
    convergence: dict[str, Any]

def load_elastic_patch_sample(path: str | Path) -> ElasticPatchSample:
    patch_path = Path(path)
    convergence = validate_convergence_certificate(patch_path)
    with np.load(patch_path, allow_pickle=False) as data:
        vertices = np.asarray(data["eta_vertices"], dtype=np.float64)
        width = float(np.linalg.norm(vertices[1] - vertices[0]))
        height = float(np.linalg.norm(vertices[3] - vertices[0]))
        length = float(np.sqrt(max(width * height, 1.0e-30)))
        young = float(np.asarray(data["eta_E"]).item())
        poisson = float(np.asarray(data["eta_nu"]).item())
        thickness = float(np.asarray(data["eta_thickness"]).item())
        scale = young * thickness
        stiffness = np.asarray(data["S"], dtype=np.float64) / scale
        rigid_body_modes = np.asarray(data["rigid_body_modes"], dtype=np.float64).T
    return ElasticPatchSample(
        path=patch_path,
        eta=np.asarray([width / length, height / length, poisson], dtype=np.float64),
        stiffness=stiffness,
        stiffness_scale=scale,
        rigid_body_modes=rigid_body_modes,
        port_dofs=int(stiffness.shape[0]),
        dataset_role=str(convergence["dataset_role"]),
        convergence_certificate=Path(str(convergence["certificate_path"])),
        convergence=convergence,
    )

def load_elastic_patch_samples(
    data_root: str | Path,
    *,
    allowed_roles: Iterable[str] | None = None,
    bubbles_per_edge: int = 2,
) -> list[ElasticPatchSample]:
    paths = [
        path
        for path in find_patch_npz(data_root)
        if patch_matches_filters(
            path,
            port_scheme="vertex_bubble_vector2",
            bubbles_per_edge=bubbles_per_edge,
            port_dofs=None,
        )
    ]
    samples = [load_elastic_patch_sample(path) for path in paths]
    if allowed_roles is not None:
        roles = set(allowed_roles)
        samples = [sample for sample in samples if sample.dataset_role in roles]
    return samples

def polygon_operator_features(
    vertices: np.ndarray,
    bubbles_per_edge: int,
    poisson: float,
    *,
    material_feature_mode: str = "raw-nu",
    geometry_normalization_schema: str = LEGACY_VERTEX_RMS_SCHEMA,
) -> tuple[np.ndarray, np.ndarray]:
    if material_feature_mode not in {"raw-nu", "lame-enhanced"}:
        raise ValueError("material_feature_mode must be 'raw-nu' or 'lame-enhanced'")
    if material_feature_mode == "lame-enhanced" and not (-0.999 < float(poisson) < 0.499):
        raise ValueError("lame-enhanced plane-strain features require -0.999 < poisson < 0.499")
    geometry_normalization_schema = validate_geometry_normalization_schema(
        geometry_normalization_schema
    )
    points = np.asarray(vertices, dtype=np.float64)
    if geometry_normalization_schema == BOUNDARY_ARCLENGTH_SCHEMA:
        center, scale = boundary_arclength_centroid_rms(points)
    else:
        center = np.mean(points, axis=0)
        scale = max(
            float(np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1)))),
            1.0e-12,
        )
    centered = points - center
    normalized = centered / scale
    edge_count = int(points.shape[0])
    shifted = np.roll(normalized, -1, axis=0)
    edge_vectors = shifted - normalized
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    perimeter = max(float(np.sum(edge_lengths)), 1.0e-12)
    mean_edge = max(perimeter / float(max(edge_count, 1)), 1.0e-12)
    edge_ratios = edge_lengths / mean_edge
    cumulative = np.concatenate([[0.0], np.cumsum(edge_lengths[:-1])]) / perimeter
    turn_sin = np.zeros(edge_count, dtype=np.float64)
    turn_cos = np.ones(edge_count, dtype=np.float64)
    for index in range(edge_count):
        incoming = normalized[index] - normalized[(index - 1) % edge_count]
        outgoing = normalized[(index + 1) % edge_count] - normalized[index]
        incoming /= max(float(np.linalg.norm(incoming)), 1.0e-12)
        outgoing /= max(float(np.linalg.norm(outgoing)), 1.0e-12)
        turn_sin[index] = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        turn_cos[index] = float(np.dot(incoming, outgoing))

    def scalar_row(
        xy: np.ndarray,
        tangent: np.ndarray,
        *,
        is_vertex: float,
        is_bubble: float,
        mode_value: float,
        edge_ratio: float,
        previous_ratio: float,
        next_ratio: float,
        turn_sin_value: float,
        turn_cos_value: float,
        arc_fraction: float,
    ) -> list[float]:
        tangent = np.asarray(tangent, dtype=np.float64)
        tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
        normal = np.asarray([tangent[1], -tangent[0]], dtype=np.float64)
        imbalance = abs(previous_ratio - next_ratio) / max(previous_ratio + next_ratio, 1.0e-12)
        return [
            float(xy[0]),
            float(xy[1]),
            float(tangent[0]),
            float(tangent[1]),
            float(normal[0]),
            float(normal[1]),
            float(is_vertex),
            float(is_bubble),
            float(mode_value),
            float(np.linalg.norm(xy)),
            float(edge_ratio),
            float(previous_ratio),
            float(next_ratio),
            float(imbalance),
            float(turn_sin_value),
            float(turn_cos_value),
            float(arc_fraction),
        ]

    scalar_rows = []
    for index in range(edge_count):
        previous = normalized[(index - 1) % edge_count]
        following = normalized[(index + 1) % edge_count]
        if geometry_normalization_schema == BOUNDARY_ARCLENGTH_SCHEMA:
            tangent = stable_unit_tangent_average(
                normalized[index] - previous,
                following - normalized[index],
            )
        else:
            tangent = following - previous
        previous_ratio = float(edge_ratios[(index - 1) % edge_count])
        next_ratio = float(edge_ratios[index])
        scalar_rows.append(
            scalar_row(
                normalized[index],
                tangent,
                is_vertex=1.0,
                is_bubble=0.0,
                mode_value=0.0,
                edge_ratio=0.5 * (previous_ratio + next_ratio),
                previous_ratio=previous_ratio,
                next_ratio=next_ratio,
                turn_sin_value=float(turn_sin[index]),
                turn_cos_value=float(turn_cos[index]),
                arc_fraction=float(cumulative[index]),
            )
        )
    for edge in range(edge_count):
        start = normalized[edge]
        end = normalized[(edge + 1) % edge_count]
        tangent = end - start
        midpoint = 0.5 * (start + end)
        edge_ratio = float(edge_ratios[edge])
        previous_ratio = float(edge_ratios[(edge - 1) % edge_count])
        next_ratio = float(edge_ratios[(edge + 1) % edge_count])
        midpoint_arc = (cumulative[edge] + 0.5 * edge_lengths[edge] / perimeter) % 1.0
        for mode in range(bubbles_per_edge):
            mode_value = float(mode + 1) / float(max(bubbles_per_edge, 1))
            scalar_rows.append(
                scalar_row(
                    midpoint,
                    tangent,
                    is_vertex=0.0,
                    is_bubble=1.0,
                    mode_value=mode_value,
                    edge_ratio=edge_ratio,
                    previous_ratio=previous_ratio,
                    next_ratio=next_ratio,
                    turn_sin_value=0.5 * float(turn_sin[edge] + turn_sin[(edge + 1) % edge_count]),
                    turn_cos_value=0.5 * float(turn_cos[edge] + turn_cos[(edge + 1) % edge_count]),
                    arc_fraction=float(midpoint_arc),
                )
            )
    scalar = np.asarray(scalar_rows, dtype=np.float64)
    node_features = np.vstack(
        [
            np.column_stack([scalar, np.ones(scalar.shape[0]), np.zeros(scalar.shape[0])]),
            np.column_stack([scalar, np.zeros(scalar.shape[0]), np.ones(scalar.shape[0])]),
        ]
    )
    area = 0.5 * abs(float(np.sum(normalized[:, 0] * shifted[:, 1] - shifted[:, 0] * normalized[:, 1])))
    edge_aspect = float(np.max(edge_ratios) / max(np.min(edge_ratios), 1.0e-12))
    radial = np.linalg.norm(normalized, axis=1)
    if geometry_normalization_schema == BOUNDARY_ARCLENGTH_SCHEMA:
        radial_summary = float(np.log(max(float(np.max(radial)), 1.0)))
        concavity = float(min(0.0, float(np.min(turn_sin))))
    else:
        radial_summary = float(
            np.log(float(np.max(radial) / max(np.min(radial), 1.0e-12)))
        )
        concavity = float(np.min(turn_sin))
    mean_abs_turn_or_lame = float(np.mean(np.abs(turn_sin)))
    if material_feature_mode == "lame-enhanced":
        mean_abs_turn_or_lame = float(poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson)))
    global_features = np.asarray(
        [
            float(poisson),
            float(edge_count) / 16.0,
            area / max(perimeter * perimeter, 1.0e-12),
            float(np.std(edge_ratios)),
            float(np.log(edge_aspect)),
            concavity,
            mean_abs_turn_or_lame,
            radial_summary,
        ],
        dtype=np.float64,
    )
    return node_features, global_features

def load_polygon_elastic_sample(
    path: str | Path,
    *,
    material_feature_mode: str = "raw-nu",
    geometry_normalization_schema: str = LEGACY_VERTEX_RMS_SCHEMA,
) -> PolygonElasticSample:
    patch_path = Path(path)
    convergence = validate_convergence_certificate(patch_path)
    with np.load(patch_path, allow_pickle=False) as data:
        vertices = np.asarray(data["eta_vertices"], dtype=np.float64)
        young = float(np.asarray(data["eta_E"]).item())
        poisson = float(np.asarray(data["eta_nu"]).item())
        thickness = float(np.asarray(data["eta_thickness"]).item())
        bubbles = int(np.asarray(data["bubbles_per_edge"]).item())
        scale = young * thickness
        stiffness = np.asarray(data["S"], dtype=np.float64) / scale
        rigid_body_modes = np.asarray(data["rigid_body_modes"], dtype=np.float64).T
    node_features, global_features = polygon_operator_features(
        vertices,
        bubbles,
        poisson,
        material_feature_mode=material_feature_mode,
        geometry_normalization_schema=geometry_normalization_schema,
    )
    if node_features.shape[0] != stiffness.shape[0]:
        raise ValueError(
            f"Polygon feature/port mismatch for {patch_path}: {node_features.shape[0]} != {stiffness.shape[0]}"
        )
    return PolygonElasticSample(
        path=patch_path,
        vertices=vertices,
        node_features=node_features,
        global_features=global_features,
        stiffness=stiffness,
        stiffness_scale=scale,
        rigid_body_modes=rigid_body_modes,
        edge_count=int(vertices.shape[0]),
        bubbles_per_edge=bubbles,
        dataset_role=str(convergence["dataset_role"]),
        convergence_certificate=Path(str(convergence["certificate_path"])),
        convergence=convergence,
    )

def load_polygon_elastic_samples(
    data_root: str | Path,
    *,
    allowed_roles: Iterable[str] | None = None,
    min_edge_count: int | None = None,
    max_edge_count: int | None = None,
    material_feature_mode: str = "raw-nu",
    geometry_normalization_schema: str = LEGACY_VERTEX_RMS_SCHEMA,
) -> list[PolygonElasticSample]:
    paths = [
        path
        for path in find_patch_npz(data_root)
        if patch_matches_filters(
            path,
            port_scheme="polygon_vertex_bubble_vector2",
            bubbles_per_edge=None,
            port_dofs=None,
        )
    ]
    samples = [
        load_polygon_elastic_sample(
            path,
            material_feature_mode=material_feature_mode,
            geometry_normalization_schema=geometry_normalization_schema,
        )
        for path in paths
    ]
    if allowed_roles is not None:
        roles = set(allowed_roles)
        samples = [sample for sample in samples if sample.dataset_role in roles]
    if min_edge_count is not None:
        samples = [sample for sample in samples if sample.edge_count >= int(min_edge_count)]
    if max_edge_count is not None:
        samples = [sample for sample in samples if sample.edge_count <= int(max_edge_count)]
    return samples

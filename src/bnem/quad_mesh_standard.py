

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

@dataclass(frozen=True)
class GeometryComplexity:
    hole_count: int
    maximum_simplified_vertices_per_hole: int
    maximum_hole_area: float
    maximum_hole_perimeter: float
    maximum_corner_turn_degrees: float

@dataclass(frozen=True)
class QuadMeshTier:
    name: str
    far_field_size: float
    hole_boundary_size: float

REGULAR = QuadMeshTier("regular", 0.18, 0.05)
ANGULAR = QuadMeshTier("angular", 0.14, 0.035)
INTRICATE = QuadMeshTier("intricate", 0.12, 0.028)

EDGE_MODE_FRACTION = 0.40
SIMPLIFY_TOLERANCE = 0.01

QUALITY_GATES = {
    "minimum_angle_degrees": 20.0,
    "maximum_angle_degrees": 162.0,
    "maximum_edge_aspect_by_tier": {
        "regular": 8.0,
        "angular": 8.0,
        "intricate": 10.0,
    },
    "minimum_scaled_jacobian": 0.30,
}

QUALITY_COMPARISON_TOLERANCES = {
    "angle_degrees": 0.01,
    "edge_aspect": 1.0e-9,
    "scaled_jacobian": 1.0e-9,
}

def geometry_complexity(
    domain: object,
    *,
    simplify_tolerance: float = SIMPLIFY_TOLERANCE,
) -> GeometryComplexity:


    interiors = list(getattr(domain, "interiors", ()))
    if not interiors:
        raise ValueError("mesh standard requires at least one internal cut-out")
    vertices_per_hole: list[int] = []
    areas: list[float] = []
    perimeters: list[float] = []
    corner_turns: list[float] = []
    for ring in interiors:
        simplified = ring.simplify(
            float(simplify_tolerance), preserve_topology=True
        )
        coordinates = np.asarray(simplified.coords[:-1], dtype=np.float64)
        if len(coordinates) < 3:
            raise ValueError("simplified cut-out boundary is degenerate")
        shifted = np.roll(coordinates, -1, axis=0)
        area = 0.5 * abs(
            float(
                np.sum(
                    coordinates[:, 0] * shifted[:, 1]
                    - shifted[:, 0] * coordinates[:, 1]
                )
            )
        )
        incoming = coordinates - np.roll(coordinates, 1, axis=0)
        outgoing = shifted - coordinates
        denominator = np.linalg.norm(incoming, axis=1) * np.linalg.norm(
            outgoing, axis=1
        )
        if np.any(denominator <= 1.0e-14):
            raise ValueError("cut-out boundary contains a zero-length segment")
        cosine = np.sum(incoming * outgoing, axis=1) / denominator
        turn = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        vertices_per_hole.append(int(len(coordinates)))
        areas.append(float(area))
        perimeters.append(float(simplified.length))
        corner_turns.append(float(np.max(turn)))
    return GeometryComplexity(
        hole_count=len(interiors),
        maximum_simplified_vertices_per_hole=max(vertices_per_hole),
        maximum_hole_area=max(areas),
        maximum_hole_perimeter=max(perimeters),
        maximum_corner_turn_degrees=max(corner_turns),
    )

def select_quad_mesh_tier(domain: object) -> tuple[QuadMeshTier, GeometryComplexity]:


    metrics = geometry_complexity(domain)
    if (
        metrics.hole_count >= 10
        or metrics.maximum_hole_perimeter >= 8.0
        or metrics.maximum_simplified_vertices_per_hole >= 128
    ):
        return INTRICATE, metrics
    if (
        metrics.hole_count >= 6
        or metrics.maximum_simplified_vertices_per_hole >= 64
        or (
            metrics.maximum_hole_area >= 0.18
            and metrics.maximum_corner_turn_degrees >= 85.0
        )
    ):
        return ANGULAR, metrics
    return REGULAR, metrics

def mesh_standard_record(domain: object) -> dict[str, object]:
    tier, metrics = select_quad_mesh_tier(domain)
    return {
        "tier": asdict(tier),
        "geometry_complexity": asdict(metrics),
        "edge_mode_fraction": EDGE_MODE_FRACTION,
        "simplify_tolerance": SIMPLIFY_TOLERANCE,
        "quality_gates": dict(QUALITY_GATES),
        "quality_comparison_tolerances": dict(QUALITY_COMPARISON_TOLERANCES),
    }

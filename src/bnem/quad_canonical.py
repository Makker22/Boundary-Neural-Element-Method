

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_D4_STEPS = np.asarray((1, 1, 1, 1, -1, -1, -1, -1), dtype=np.int64)
_D4_INDICES = np.asarray(
    [
        tuple((start + step * offset) % 4 for offset in range(4))
        for step in (1, -1)
        for start in range(4)
    ],
    dtype=np.int64,
)

@dataclass(frozen=True)
class CanonicalQuad:


    vertices: np.ndarray
    port_transform: np.ndarray
    coordinate_transform: np.ndarray
    input_indices_for_canonical: tuple[int, int, int, int]
    traversal_step: int
    scale: float

    def stiffness_to_canonical(self, input_stiffness: np.ndarray) -> np.ndarray:
        matrix = np.asarray(input_stiffness, dtype=np.float64)
        if matrix.shape != self.port_transform.shape:
            raise ValueError("input_stiffness has incompatible shape")
        result = self.port_transform @ matrix @ self.port_transform.T
        return 0.5 * (result + result.T)

    def stiffness_to_input(self, canonical_stiffness: np.ndarray) -> np.ndarray:
        matrix = np.asarray(canonical_stiffness, dtype=np.float64)
        if matrix.shape != self.port_transform.shape:
            raise ValueError("canonical_stiffness has incompatible shape")
        result = self.port_transform.T @ matrix @ self.port_transform
        return 0.5 * (result + result.T)

@dataclass(frozen=True)
class CanonicalQuadBatchArrays:


    vertices: np.ndarray
    port_transforms: np.ndarray | None
    coordinate_transforms: np.ndarray
    input_indices_for_canonical: np.ndarray
    traversal_steps: np.ndarray
    scales: np.ndarray

    def stiffness_to_canonical(self, input_stiffness: np.ndarray) -> np.ndarray:
        matrix = np.asarray(input_stiffness, dtype=np.float64)
        if matrix.shape != self.port_transform.shape:
            raise ValueError("input_stiffness has incompatible shape")

        result = self.port_transform @ matrix @ self.port_transform.T
        return 0.5 * (result + result.T)

def _signed_area(points: np.ndarray) -> float:
    shifted = np.roll(points, -1, axis=0)
    return 0.5 * float(
        np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1])
    )

def _scalar_port_permutation(
    indices: tuple[int, int, int, int],
    step: int,
    bubbles_per_edge: int,
) -> np.ndarray:
    scalar_dofs = 4 * (1 + bubbles_per_edge)
    transform = np.zeros((scalar_dofs, scalar_dofs), dtype=np.float64)
    for canonical, original in enumerate(indices):
        transform[canonical, original] = 1.0
    for canonical_edge in range(4):
        if step == 1:
            original_edge = indices[canonical_edge]
        else:
            original_edge = indices[(canonical_edge + 1) % 4]
        for mode in range(bubbles_per_edge):
            canonical_dof = 4 + canonical_edge * bubbles_per_edge + mode
            original_dof = 4 + original_edge * bubbles_per_edge + mode
            transform[canonical_dof, original_dof] = 1.0 if step == 1 else (-1.0) ** mode
    return transform

def _batch_scalar_port_permutation(
    indices: np.ndarray,
    steps: np.ndarray,
    bubbles_per_edge: int,
) -> np.ndarray:


    index_array = np.asarray(indices, dtype=np.int64)
    step_array = np.asarray(steps, dtype=np.int64)
    if index_array.ndim != 2 or index_array.shape[1] != 4:
        raise ValueError("indices must have shape (cell_count,4)")
    if step_array.shape != (len(index_array),):
        raise ValueError("steps must have one value per cell")
    bubbles = int(bubbles_per_edge)
    if bubbles < 0:
        raise ValueError("bubbles_per_edge must be non-negative")

    count = len(index_array)
    scalar_dofs = 4 * (1 + bubbles)
    transform = np.zeros((count, scalar_dofs, scalar_dofs), dtype=np.float64)
    rows = np.arange(count, dtype=np.int64)
    for canonical in range(4):
        transform[rows, canonical, index_array[:, canonical]] = 1.0
    for canonical_edge in range(4):
        original_edge = np.where(
            step_array == 1,
            index_array[:, canonical_edge],
            index_array[:, (canonical_edge + 1) % 4],
        )
        for mode in range(bubbles):
            canonical_dof = 4 + canonical_edge * bubbles + mode
            original_dof = 4 + original_edge * bubbles + mode
            signs = np.where(step_array == 1, 1.0, (-1.0) ** mode)
            transform[rows, canonical_dof, original_dof] = signs
    return transform

def canonicalize_quad(
    vertices: np.ndarray,
    *,
    bubbles_per_edge: int = 1,
    key_decimals: int = 12,
) -> CanonicalQuad:







    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError(f"vertices must be finite with shape (4,2), got {points.shape}")
    if abs(_signed_area(points)) <= 1.0e-14:
        raise ValueError("vertices must define a nondegenerate boundary ordering")
    bubbles = int(bubbles_per_edge)
    if bubbles < 0:
        raise ValueError("bubbles_per_edge must be non-negative")
    center = np.mean(points, axis=0)
    centered = points - center
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if scale <= 1.0e-14:
        raise ValueError("vertices have degenerate scale")

    candidates: list[tuple[tuple[float, ...], np.ndarray, np.ndarray, tuple[int, ...], int]] = []
    for step in (1, -1):
        for start in range(4):
            indices = tuple((start + step * offset) % 4 for offset in range(4))
            ordered = points[np.asarray(indices, dtype=np.int64)]
            direction = ordered[1] - ordered[0]
            length = float(np.linalg.norm(direction))
            if length <= 1.0e-14:
                continue
            tangent = direction / length
            if _signed_area(ordered) > 0.0:
                second = np.asarray([-tangent[1], tangent[0]])
            else:
                second = np.asarray([tangent[1], -tangent[0]])
            coordinate_transform = np.stack((tangent, second), axis=0)
            canonical = (ordered - center) @ coordinate_transform.T / scale
            if _signed_area(canonical) <= 0.0:
                raise RuntimeError("internal canonicalization orientation failure")
            key = tuple(float(value) for value in np.round(canonical.reshape(-1), key_decimals))
            candidates.append((key, canonical, coordinate_transform, indices, step))
    if not candidates:
        raise ValueError("could not construct a canonical frame")
    _key, canonical, coordinate_transform, raw_indices, step = min(
        candidates, key=lambda row: (row[0], row[4], row[3])
    )
    indices = tuple(int(value) for value in raw_indices)
    scalar = _scalar_port_permutation(indices, int(step), bubbles)
    scalar_dofs = scalar.shape[0]
    port = np.zeros((2 * scalar_dofs, 2 * scalar_dofs), dtype=np.float64)
    port[:scalar_dofs, :scalar_dofs] = coordinate_transform[0, 0] * scalar
    port[:scalar_dofs, scalar_dofs:] = coordinate_transform[0, 1] * scalar
    port[scalar_dofs:, :scalar_dofs] = coordinate_transform[1, 0] * scalar
    port[scalar_dofs:, scalar_dofs:] = coordinate_transform[1, 1] * scalar
    return CanonicalQuad(
        vertices=np.asarray(canonical, dtype=np.float64),
        port_transform=port,
        coordinate_transform=coordinate_transform,
        input_indices_for_canonical=indices,
        traversal_step=int(step),
        scale=scale,
    )

def canonicalize_quad_batch_arrays(
    vertices: np.ndarray,
    *,
    bubbles_per_edge: int = 1,
    key_decimals: int = 12,
    build_port_transforms: bool = True,
) -> CanonicalQuadBatchArrays:


    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (4, 2):
        raise ValueError(
            f"vertices must be finite with shape (cell_count,4,2), got {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("vertices must be finite")
    bubbles = int(bubbles_per_edge)
    if bubbles < 0:
        raise ValueError("bubbles_per_edge must be non-negative")
    if len(points) == 0:
        scalar_dofs = 4 * (1 + bubbles)
        return CanonicalQuadBatchArrays(
            vertices=np.empty((0, 4, 2), dtype=np.float64),
            port_transforms=(
                np.empty(
                    (0, 2 * scalar_dofs, 2 * scalar_dofs), dtype=np.float64
                )
                if build_port_transforms
                else None
            ),
            coordinate_transforms=np.empty((0, 2, 2), dtype=np.float64),
            input_indices_for_canonical=np.empty((0, 4), dtype=np.int64),
            traversal_steps=np.empty(0, dtype=np.int64),
            scales=np.empty(0, dtype=np.float64),
        )

    shifted = np.roll(points, -1, axis=1)
    signed_area = 0.5 * np.sum(
        points[:, :, 0] * shifted[:, :, 1]
        - shifted[:, :, 0] * points[:, :, 1],
        axis=1,
    )
    if np.any(np.abs(signed_area) <= 1.0e-14):
        raise ValueError("vertices must define nondegenerate boundary orderings")
    center = np.mean(points, axis=1)
    centered = points - center[:, None, :]
    scales = np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=1))
    if np.any(scales <= 1.0e-14):
        raise ValueError("vertices have degenerate scale")

    ordered = points[:, _D4_INDICES, :]
    directions = ordered[:, :, 1, :] - ordered[:, :, 0, :]
    lengths = np.linalg.norm(directions, axis=2)
    valid = lengths > 1.0e-14
    if np.any(~np.any(valid, axis=1)):
        raise ValueError("could not construct a canonical frame")
    safe_lengths = np.where(valid, lengths, 1.0)
    tangents = directions / safe_lengths[:, :, None]
    ordered_shifted = np.roll(ordered, -1, axis=2)
    candidate_areas = 0.5 * np.sum(
        ordered[:, :, :, 0] * ordered_shifted[:, :, :, 1]
        - ordered_shifted[:, :, :, 0] * ordered[:, :, :, 1],
        axis=2,
    )
    positive_second = np.stack((-tangents[:, :, 1], tangents[:, :, 0]), axis=2)
    negative_second = np.stack((tangents[:, :, 1], -tangents[:, :, 0]), axis=2)
    second = np.where(
        (candidate_areas > 0.0)[:, :, None], positive_second, negative_second
    )
    coordinate_transforms = np.stack((tangents, second), axis=2)
    candidate_vertices = np.matmul(
        ordered - center[:, None, None, :],
        np.swapaxes(coordinate_transforms, -1, -2),
    ) / scales[:, None, None, None]

    rounded = np.round(
        candidate_vertices.reshape(len(points), len(_D4_INDICES), -1),
        int(key_decimals),
    )
    rounded = np.where(valid[:, :, None], rounded, np.inf)
    count = len(points)
    steps = np.broadcast_to(_D4_STEPS[None, :], (count, len(_D4_STEPS)))
    candidate_indices = np.broadcast_to(
        _D4_INDICES[None, :, :], (count, *_D4_INDICES.shape)
    )
    comparison_fields = [rounded[:, :, index] for index in range(rounded.shape[2])]
    comparison_fields.append(steps)
    comparison_fields.extend(
        candidate_indices[:, :, index] for index in range(candidate_indices.shape[2])
    )
    selected = np.lexsort(tuple(reversed(comparison_fields)), axis=1)[:, 0]
    rows = np.arange(count, dtype=np.int64)
    canonical = candidate_vertices[rows, selected]
    rotations = coordinate_transforms[rows, selected]
    selected_indices = _D4_INDICES[selected]
    selected_steps = _D4_STEPS[selected]

    port = None
    if build_port_transforms:
        scalar = _batch_scalar_port_permutation(
            selected_indices, selected_steps, bubbles
        )
        scalar_dofs = scalar.shape[1]
        port = np.zeros(
            (count, 2 * scalar_dofs, 2 * scalar_dofs), dtype=np.float64
        )
        port[:, :scalar_dofs, :scalar_dofs] = (
            rotations[:, 0, 0, None, None] * scalar
        )
        port[:, :scalar_dofs, scalar_dofs:] = (
            rotations[:, 0, 1, None, None] * scalar
        )
        port[:, scalar_dofs:, :scalar_dofs] = (
            rotations[:, 1, 0, None, None] * scalar
        )
        port[:, scalar_dofs:, scalar_dofs:] = (
            rotations[:, 1, 1, None, None] * scalar
        )

    return CanonicalQuadBatchArrays(
        vertices=np.asarray(canonical, dtype=np.float64),
        port_transforms=port,
        coordinate_transforms=np.asarray(rotations, dtype=np.float64),
        input_indices_for_canonical=np.asarray(selected_indices, dtype=np.int64),
        traversal_steps=np.asarray(selected_steps, dtype=np.int64),
        scales=np.asarray(scales, dtype=np.float64),
    )

def canonicalize_quad_batch(
    vertices: np.ndarray,
    *,
    bubbles_per_edge: int = 1,
    key_decimals: int = 12,
) -> tuple[CanonicalQuad, ...]:


    batch = canonicalize_quad_batch_arrays(
        vertices,
        bubbles_per_edge=bubbles_per_edge,
        key_decimals=key_decimals,
    )
    if batch.port_transforms is None:
        raise RuntimeError("canonical batch unexpectedly omitted port transforms")
    return tuple(
        CanonicalQuad(
            vertices=batch.vertices[index],
            port_transform=batch.port_transforms[index],
            coordinate_transform=batch.coordinate_transforms[index],
            input_indices_for_canonical=tuple(
                int(value) for value in batch.input_indices_for_canonical[index]
            ),
            traversal_step=int(batch.traversal_steps[index]),
            scale=float(batch.scales[index]),
        )
        for index in range(len(batch.vertices))
    )

def canonical_quad_with_bubbles(item: CanonicalQuad, bubbles_per_edge: int) -> CanonicalQuad:


    bubbles = int(bubbles_per_edge)
    if bubbles < 0:
        raise ValueError("bubbles_per_edge must be non-negative")
    scalar = _scalar_port_permutation(
        item.input_indices_for_canonical, item.traversal_step, bubbles
    )
    scalar_dofs = scalar.shape[0]
    rotation = item.coordinate_transform
    port = np.zeros((2 * scalar_dofs, 2 * scalar_dofs), dtype=np.float64)
    port[:scalar_dofs, :scalar_dofs] = rotation[0, 0] * scalar
    port[:scalar_dofs, scalar_dofs:] = rotation[0, 1] * scalar
    port[scalar_dofs:, :scalar_dofs] = rotation[1, 0] * scalar
    port[scalar_dofs:, scalar_dofs:] = rotation[1, 1] * scalar
    return CanonicalQuad(
        vertices=item.vertices,
        port_transform=port,
        coordinate_transform=item.coordinate_transform,
        input_indices_for_canonical=item.input_indices_for_canonical,
        traversal_step=item.traversal_step,
        scale=item.scale,
    )

def scalar_quad_port_transform(
    item: CanonicalQuad, *, bubbles_per_edge: int
) -> np.ndarray:


    return _scalar_port_permutation(
        item.input_indices_for_canonical, item.traversal_step, int(bubbles_per_edge)
    )

def scalar_quad_port_transforms(
    items: tuple[CanonicalQuad, ...] | list[CanonicalQuad],
    *,
    bubbles_per_edge: int,
) -> np.ndarray:


    bubbles = int(bubbles_per_edge)
    scalar_dofs = 4 * (1 + bubbles)
    if not items:
        return np.empty((0, scalar_dofs, scalar_dofs), dtype=np.float64)
    indices = np.asarray(
        [item.input_indices_for_canonical for item in items], dtype=np.int64
    )
    steps = np.asarray([item.traversal_step for item in items], dtype=np.int64)
    return _batch_scalar_port_permutation(indices, steps, bubbles)

def scalar_quad_port_transforms_from_arrays(
    items: CanonicalQuadBatchArrays,
    *,
    bubbles_per_edge: int,
) -> np.ndarray:


    return _batch_scalar_port_permutation(
        items.input_indices_for_canonical,
        items.traversal_steps,
        int(bubbles_per_edge),
    )

def canonical_quad_features_from_canonical(
    canonical_vertices: np.ndarray, poisson: float
) -> np.ndarray:


    canonical = np.asarray(canonical_vertices, dtype=np.float64)
    if canonical.shape != (4, 2) or not np.all(np.isfinite(canonical)):
        raise ValueError("canonical_vertices must be finite with shape (4,2)")
    shifted = np.roll(canonical, -1, axis=0)
    edges = shifted - canonical
    lengths = np.linalg.norm(edges, axis=1)
    length_ratios = lengths / np.mean(lengths)

    natural = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    determinant: list[float] = []
    scaled: list[float] = []
    for xi, eta in natural:
        dxi = 0.25 * np.asarray([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)])
        deta = 0.25 * np.asarray([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi])
        first = dxi @ canonical
        second = deta @ canonical
        det = float(first[0] * second[1] - first[1] * second[0])
        determinant.append(det)
        scaled.append(det / (float(np.linalg.norm(first) * np.linalg.norm(second))))
    determinant_array = np.asarray(determinant)
    determinant_array /= float(np.mean(determinant_array))
    nu = float(poisson)
    if not (-0.999 < nu < 0.499):
        raise ValueError("plane strain requires -0.999 < poisson < 0.499")
    lame_ratio = nu / (1.0 - 2.0 * nu)
    return np.concatenate(
        (
            canonical.reshape(-1),
            length_ratios,
            determinant_array,
            np.asarray(scaled),
            np.asarray([nu, lame_ratio]),
        )
    ).astype(np.float64)

def canonical_quad_features(vertices: np.ndarray, poisson: float) -> np.ndarray:


    canonical = canonicalize_quad(vertices, bubbles_per_edge=1).vertices
    return canonical_quad_features_from_canonical(canonical, poisson)

__all__ = [
    "CanonicalQuad",
    "CanonicalQuadBatchArrays",
    "canonical_quad_with_bubbles",
    "scalar_quad_port_transform",
    "canonical_quad_features",
    "canonical_quad_features_from_canonical",
    "canonicalize_quad",
    "canonicalize_quad_batch",
    "canonicalize_quad_batch_arrays",
    "scalar_quad_port_transforms",
    "scalar_quad_port_transforms_from_arrays",
]



from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.bnem.assembly import ConformingPolygonMeshAssembly
from src.bnem.quad_partition_certification import (
    HierarchicalPortIndicator,
    hierarchical_port_indicator,
)

def u1_and_next_mode_indices(
    bubbles_per_edge: int = 2,
) -> tuple[np.ndarray, np.ndarray]:


    bubbles = int(bubbles_per_edge)
    if bubbles < 2:
        raise ValueError("hierarchical enrichment requires at least two bubbles per edge")
    scalar_dofs = 4 * (1 + bubbles)
    scalar_u1 = np.asarray(
        [0, 1, 2, 3] + [4 + edge * bubbles for edge in range(4)], dtype=np.int64
    )
    scalar_extra = np.asarray(
        [
            4 + edge * bubbles + mode
            for edge in range(4)
            for mode in range(1, bubbles)
        ],
        dtype=np.int64,
    )
    u1 = np.concatenate((scalar_u1, scalar_dofs + scalar_u1))
    extra = np.concatenate((scalar_extra, scalar_dofs + scalar_extra))
    return u1, extra

def extract_u1_hierarchical_blocks(
    u2_stiffness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


    matrix = np.asarray(u2_stiffness, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 8 != 0:
        raise ValueError("hierarchical stiffness must be square with a valid quad port size")
    bubbles = matrix.shape[0] // 8 - 1
    if bubbles < 2:
        raise ValueError("hierarchical stiffness must contain at least two bubbles per edge")
    u1, extra = u1_and_next_mode_indices(bubbles)
    return (
        matrix[np.ix_(u1, u1)],
        matrix[np.ix_(extra, u1)],
        matrix[np.ix_(extra, extra)],
    )

@dataclass(frozen=True)
class PartitionHierarchicalResidual:
    indicator: HierarchicalPortIndicator
    global_residual: np.ndarray
    inverse_diagonal_metric: np.ndarray
    per_cell_residual_norm: np.ndarray
    excluded_global_modes: tuple[int, ...]

def assemble_partition_hierarchical_residual(
    mesh: ConformingPolygonMeshAssembly,
    u1_solution: np.ndarray,
    coupling_blocks: Sequence[np.ndarray] | np.ndarray,
    omitted_stiffness_blocks: Sequence[np.ndarray] | np.ndarray,
    *,
    reference_energy: float,
    excluded_edges: Iterable[tuple[int, int]] = (),
) -> PartitionHierarchicalResidual:







    if mesh.bubbles_per_edge != 1 or any(len(cell) != 4 for cell in mesh.cells):
        raise ValueError("hierarchical residual requires a U1 quadrilateral mesh")
    solution = np.asarray(u1_solution, dtype=np.float64)
    if solution.shape != (mesh.vector_dofs,):
        raise ValueError("u1_solution has incompatible shape")
    coupling = np.asarray(coupling_blocks, dtype=np.float64)
    omitted = np.asarray(omitted_stiffness_blocks, dtype=np.float64)
    count = len(mesh.cells)
    if coupling.ndim != 3 or coupling.shape[0] != count or coupling.shape[2] != 16:
        raise ValueError("coupling blocks must have shape [C,enrichment,16]")
    enrichment_dofs = int(coupling.shape[1])
    if enrichment_dofs % 8 != 0 or omitted.shape != (count, enrichment_dofs, enrichment_dofs):
        raise ValueError("omitted blocks have an incompatible enrichment size")
    extra_modes_per_edge = enrichment_dofs // 8

    edge_count = mesh.edge_count
    scalar_enrichment = edge_count * extra_modes_per_edge
    residual = np.zeros(2 * scalar_enrichment, dtype=np.float64)
    diagonal = np.zeros(2 * scalar_enrichment, dtype=np.float64)
    per_cell = np.zeros(count, dtype=np.float64)
    for cell_index, cell in enumerate(mesh.cells):
        local_map = mesh.local_vector_dof_map(cell_index)
        local_u1 = np.asarray(
            [sign * solution[dof] for dof, sign in local_map], dtype=np.float64
        )
        local_residual = -(coupling[cell_index] @ local_u1)
        per_cell[cell_index] = float(np.linalg.norm(local_residual))
        edge_map: list[tuple[int, float]] = []
        for local_edge in range(4):
            start = cell[local_edge]
            end = cell[(local_edge + 1) % 4]
            edge = (min(start, end), max(start, end))
            reversed_orientation = start > end
            serial = mesh.edge_serials[edge]
            for omitted_mode in range(extra_modes_per_edge):
                actual_legendre_mode = omitted_mode + 1
                sign = (
                    (-1.0) ** actual_legendre_mode if reversed_orientation else 1.0
                )
                edge_map.append(
                    (serial * extra_modes_per_edge + omitted_mode, float(sign))
                )
        vector_map = edge_map + [
            (scalar_enrichment + dof, sign) for dof, sign in edge_map
        ]
        for local, (global_mode, sign) in enumerate(vector_map):
            residual[global_mode] += sign * local_residual[local]
            diagonal[global_mode] += omitted[cell_index, local, local]

    excluded: set[int] = set()
    for raw_edge in excluded_edges:
        edge = (min(raw_edge), max(raw_edge))
        if edge not in mesh.edge_serials:
            raise ValueError(f"excluded edge {edge} is not in the mesh")
        scalar = mesh.edge_serials[edge]
        for mode in range(extra_modes_per_edge):
            scalar_mode = scalar * extra_modes_per_edge + mode
            excluded.update((scalar_mode, scalar_enrichment + scalar_mode))
    active = np.ones(2 * scalar_enrichment, dtype=bool)
    if excluded:
        active[np.asarray(sorted(excluded), dtype=np.int64)] = False
    if not np.any(active):

        indicator = HierarchicalPortIndicator(0.0, 0.0, float(reference_energy), ())
        inverse = np.zeros_like(diagonal)
    else:
        if np.any(diagonal[active] <= 0.0) or not np.all(np.isfinite(diagonal[active])):
            raise ValueError("assembled omitted diagonal must be finite and positive")
        inverse = np.zeros_like(diagonal)
        inverse[active] = 1.0 / diagonal[active]
        active_indicator = hierarchical_port_indicator(
            residual[active], inverse[active], reference_energy=reference_energy
        )
        contributions = np.zeros_like(residual)
        contributions[active] = np.asarray(active_indicator.mode_contributions)
        indicator = HierarchicalPortIndicator(
            value=active_indicator.value,
            residual_energy=active_indicator.residual_energy,
            reference_energy=active_indicator.reference_energy,
            mode_contributions=tuple(float(value) for value in contributions),
        )
    return PartitionHierarchicalResidual(
        indicator=indicator,
        global_residual=residual,
        inverse_diagonal_metric=inverse,
        per_cell_residual_norm=per_cell,
        excluded_global_modes=tuple(sorted(excluded)),
    )

__all__ = [
    "PartitionHierarchicalResidual",
    "assemble_partition_hierarchical_residual",
    "extract_u1_hierarchical_blocks",
    "u1_and_next_mode_indices",
]

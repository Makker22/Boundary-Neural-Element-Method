from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.bnem.ports import vertex_bubble_matrix

@dataclass
class DofBook:
    vertex_dofs: dict[tuple[float, float], int] = field(default_factory=dict)
    edge_bubbles: dict[str, list[int]] = field(default_factory=dict)
    edge_coords: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)

    def add_vertex(self, x: float, y: float) -> int:
        key = (float(x), float(y))
        if key not in self.vertex_dofs:
            self.vertex_dofs[key] = self.count
        return self.vertex_dofs[key]

    def add_edge(self, key: str, x0: float, y0: float, x1: float, y1: float, bubbles: int) -> list[int]:
        if key not in self.edge_bubbles:
            self.edge_bubbles[key] = [self.count + mode for mode in range(bubbles)]
            self.edge_coords[key] = (float(x0), float(y0), float(x1), float(y1))
        return self.edge_bubbles[key]

    @property
    def count(self) -> int:
        return len(self.vertex_dofs) + sum(len(indices) for indices in self.edge_bubbles.values())

def bubble_signs(bubbles: int, reversed_orientation: bool = False) -> np.ndarray:
    signs = np.ones(bubbles, dtype=np.float64)
    if reversed_orientation:
        signs = np.array([1.0 if mode % 2 == 0 else -1.0 for mode in range(bubbles)], dtype=np.float64)
    return signs

def edge_trace_matrix(xis: np.ndarray, bubbles: int) -> np.ndarray:
    endpoints = np.column_stack([0.5 * (1.0 - xis), 0.5 * (1.0 + xis)])
    return np.column_stack([endpoints, vertex_bubble_matrix(xis, bubbles)])

def add_edge_bubbles_to_transform(
    transform: np.ndarray,
    *,
    local_start: int,
    global_indices: list[int],
    reversed_orientation: bool,
) -> None:
    signs = bubble_signs(len(global_indices), reversed_orientation=reversed_orientation)
    for mode, global_dof in enumerate(global_indices):
        transform[local_start + mode, global_dof] = signs[mode]

def build_book_coarse_mortar(small_count: int, bubbles: int, mortar_bubbles: int) -> DofBook:
    book = DofBook()
    height = float(small_count)
    width = float(small_count + 1)
    book.add_vertex(0.0, 0.0)
    book.add_vertex(0.0, height)
    book.add_vertex(float(small_count), 0.0)
    book.add_vertex(float(small_count), height)
    for j in range(small_count + 1):
        book.add_vertex(width, float(j))

    book.add_edge("big_bottom", 0.0, 0.0, float(small_count), 0.0, bubbles)
    book.add_edge("big_top", 0.0, height, float(small_count), height, bubbles)
    book.add_edge("big_left", 0.0, 0.0, 0.0, height, bubbles)
    book.add_edge("coarse_interface", float(small_count), 0.0, float(small_count), height, mortar_bubbles)
    for j in range(small_count):
        book.add_edge(f"right_{j}", width, float(j), width, float(j + 1), bubbles)
    for j in range(small_count + 1):
        book.add_edge(f"small_h_{j}", float(small_count), float(j), width, float(j), bubbles)
    return book

def coarse_basis_at_y(y: np.ndarray, small_count: int, mortar_bubbles: int) -> np.ndarray:
    xi = 2.0 * y / float(small_count) - 1.0
    return edge_trace_matrix(xi, mortar_bubbles)

def project_coarse_trace_to_segment(
    *,
    segment: int,
    small_count: int,
    bubbles: int,
    mortar_bubbles: int,
    coarse_dofs: list[int],
    total_dofs: int,
) -> np.ndarray:

    local_xi = np.linspace(-1.0, 1.0, 33)
    y = float(segment) + 0.5 * (local_xi + 1.0)
    coarse_values = coarse_basis_at_y(y, small_count, mortar_bubbles)
    bottom_values = coarse_basis_at_y(np.asarray([float(segment)]), small_count, mortar_bubbles)[0]
    top_values = coarse_basis_at_y(np.asarray([float(segment + 1)]), small_count, mortar_bubbles)[0]
    local_linear = np.outer(0.5 * (1.0 - local_xi), bottom_values) + np.outer(
        0.5 * (1.0 + local_xi),
        top_values,
    )
    residual = coarse_values - local_linear
    local_bubbles = vertex_bubble_matrix(local_xi, bubbles)
    bubble_coeffs, *_ = np.linalg.lstsq(local_bubbles, residual, rcond=None)

    projection = np.zeros((2 + bubbles, total_dofs), dtype=np.float64)
    for col, dof in enumerate(coarse_dofs):
        projection[0, dof] = bottom_values[col]
        projection[1, dof] = top_values[col]
        for mode in range(bubbles):
            projection[2 + mode, dof] = bubble_coeffs[mode, col]
    return projection

def build_nonmatching_transforms_coarse_mortar(
    *,
    book: DofBook,
    small_count: int,
    bubbles: int,
    mortar_bubbles: int,
    local_dofs: int,
) -> tuple[list[np.ndarray], list[int]]:
    total = book.count
    transforms: list[np.ndarray] = []
    coarse_dofs = [
        book.vertex_dofs[(float(small_count), 0.0)],
        book.vertex_dofs[(float(small_count), float(small_count))],
        *book.edge_bubbles["coarse_interface"],
    ]

    big = np.zeros((local_dofs, total), dtype=np.float64)
    big[0, book.vertex_dofs[(0.0, 0.0)]] = 1.0
    big[1, coarse_dofs[0]] = 1.0
    big[2, coarse_dofs[1]] = 1.0
    big[3, book.vertex_dofs[(0.0, float(small_count))]] = 1.0
    add_edge_bubbles_to_transform(big, local_start=4, global_indices=book.edge_bubbles["big_bottom"], reversed_orientation=False)
    for mode, global_dof in enumerate(book.edge_bubbles["coarse_interface"]):
        big[4 + bubbles + mode, global_dof] = 1.0
    add_edge_bubbles_to_transform(big, local_start=4 + 2 * bubbles, global_indices=book.edge_bubbles["big_top"], reversed_orientation=True)
    add_edge_bubbles_to_transform(big, local_start=4 + 3 * bubbles, global_indices=book.edge_bubbles["big_left"], reversed_orientation=True)
    transforms.append(big)

    for segment in range(small_count):
        y0 = float(segment)
        y1 = float(segment + 1)
        left_projection = project_coarse_trace_to_segment(
            segment=segment,
            small_count=small_count,
            bubbles=bubbles,
            mortar_bubbles=mortar_bubbles,
            coarse_dofs=coarse_dofs,
            total_dofs=total,
        )
        cell = np.zeros((local_dofs, total), dtype=np.float64)
        cell[0, :] = left_projection[0]
        cell[1, book.vertex_dofs[(float(small_count + 1), y0)]] = 1.0
        cell[2, book.vertex_dofs[(float(small_count + 1), y1)]] = 1.0
        cell[3, :] = left_projection[1]
        add_edge_bubbles_to_transform(cell, local_start=4, global_indices=book.edge_bubbles[f"small_h_{segment}"], reversed_orientation=False)
        add_edge_bubbles_to_transform(cell, local_start=4 + bubbles, global_indices=book.edge_bubbles[f"right_{segment}"], reversed_orientation=False)
        add_edge_bubbles_to_transform(cell, local_start=4 + 2 * bubbles, global_indices=book.edge_bubbles[f"small_h_{segment + 1}"], reversed_orientation=True)
        signs = bubble_signs(bubbles, reversed_orientation=True)
        for mode in range(bubbles):
            cell[4 + 3 * bubbles + mode, :] = signs[mode] * left_projection[2 + mode]
        transforms.append(cell)

    return transforms, coarse_dofs

def assemble_from_transforms(stiffness: np.ndarray, transforms: list[np.ndarray], total_dofs: int) -> np.ndarray:
    k_global = np.zeros((total_dofs, total_dofs), dtype=np.float64)
    for transform in transforms:
        k_global += transform.T @ stiffness @ transform
    return k_global

def apply_mortar_penalty(
    k_global: np.ndarray,
    *,
    coarse_dofs: list[int],
    mortar_bubbles: int,
    filter_order: int | None,
    penalty: float,
) -> tuple[list[int], float]:
    if filter_order is None or penalty <= 0.0 or filter_order >= mortar_bubbles:
        return [], 0.0
    if filter_order < 0:
        raise ValueError("--mortar-filter-order must be nonnegative")

    high_mode_dofs = coarse_dofs[2 + filter_order :]
    positive_diag = np.diag(k_global)
    positive_diag = positive_diag[positive_diag > 0.0]
    reference = float(np.mean(positive_diag)) if positive_diag.size else 1.0
    penalty_value = penalty * reference
    for dof in high_mode_dofs:
        k_global[dof, dof] += penalty_value
    return high_mode_dofs, penalty_value

def mortar_high_mode_ratio(
    q: np.ndarray,
    *,
    coarse_dofs: list[int],
    mortar_bubbles: int,
    filter_order: int | None,
) -> float:
    if filter_order is None or filter_order >= mortar_bubbles:
        return 0.0
    bubble_coeffs = np.asarray([q[dof] for dof in coarse_dofs[2:]], dtype=np.float64)
    high_coeffs = np.asarray([q[dof] for dof in coarse_dofs[2 + filter_order :]], dtype=np.float64)
    return float(np.linalg.norm(high_coeffs) / (np.linalg.norm(bubble_coeffs) + 1e-12))

@dataclass(frozen=True)
class StructuredGridAssembly:
    nx: int
    ny: int
    modes_per_edge: int
    stiffness: np.ndarray
    cell_width: float = 1.0
    cell_height: float = 1.0

    @property
    def horizontal_edges(self) -> int:
        return self.nx * (self.ny + 1)

    @property
    def vertical_edges(self) -> int:
        return (self.nx + 1) * self.ny

    @property
    def edge_count(self) -> int:
        return self.horizontal_edges + self.vertical_edges

    @property
    def dofs(self) -> int:
        return self.edge_count * self.modes_per_edge

    def horizontal_edge(self, i: int, j: int) -> int:
        return j * self.nx + i

    def vertical_edge(self, i: int, j: int) -> int:
        return self.horizontal_edges + j * (self.nx + 1) + i

    def edge_dof(self, edge_id: int, mode: int) -> int:
        return edge_id * self.modes_per_edge + mode

    def reversed_signs(self) -> np.ndarray:
        return np.array(
            [1.0 if mode % 2 == 0 else -1.0 for mode in range(self.modes_per_edge)],
            dtype=np.float64,
        )

    def local_edge_map(self, cell_i: int, cell_j: int) -> list[tuple[int, np.ndarray]]:
        same = np.ones(self.modes_per_edge, dtype=np.float64)
        rev = self.reversed_signs()
        return [
            (self.horizontal_edge(cell_i, cell_j), same),
            (self.vertical_edge(cell_i + 1, cell_j), same),
            (self.horizontal_edge(cell_i, cell_j + 1), rev),
            (self.vertical_edge(cell_i, cell_j), rev),
        ]

    def assemble_matrix(self) -> np.ndarray:
        k_global = np.zeros((self.dofs, self.dofs), dtype=np.float64)
        m = self.modes_per_edge
        for cell_j in range(self.ny):
            for cell_i in range(self.nx):
                edge_map = self.local_edge_map(cell_i, cell_j)
                for local_edge_a, (edge_a, signs_a) in enumerate(edge_map):
                    for local_mode_a in range(m):
                        local_a = local_edge_a * m + local_mode_a
                        global_a = self.edge_dof(edge_a, local_mode_a)
                        sign_a = signs_a[local_mode_a]
                        for local_edge_b, (edge_b, signs_b) in enumerate(edge_map):
                            for local_mode_b in range(m):
                                local_b = local_edge_b * m + local_mode_b
                                global_b = self.edge_dof(edge_b, local_mode_b)
                                sign_b = signs_b[local_mode_b]
                                k_global[global_a, global_b] += (
                                    sign_a * self.stiffness[local_a, local_b] * sign_b
                                )
        return k_global

    def dirichlet_vector(
        self,
        *,
        left: float,
        right: float,
        bottom: float,
        top: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        fixed = np.zeros(self.dofs, dtype=bool)
        values = np.zeros(self.dofs, dtype=np.float64)

        for i in range(self.nx):
            for j, value in [(0, bottom), (self.ny, top)]:
                edge = self.horizontal_edge(i, j)
                fixed[self.edge_dof(edge, 0)] = True
                values[self.edge_dof(edge, 0)] = value
                for mode in range(1, self.modes_per_edge):
                    fixed[self.edge_dof(edge, mode)] = True

        for j in range(self.ny):
            for i, value in [(0, left), (self.nx, right)]:
                edge = self.vertical_edge(i, j)
                fixed[self.edge_dof(edge, 0)] = True
                values[self.edge_dof(edge, 0)] = value
                for mode in range(1, self.modes_per_edge):
                    fixed[self.edge_dof(edge, mode)] = True

        return fixed, values

    def dirichlet_vector_linear_x(self) -> tuple[np.ndarray, np.ndarray]:





        fixed = np.zeros(self.dofs, dtype=bool)
        values = np.zeros(self.dofs, dtype=np.float64)
        total_width = self.nx * self.cell_width

        def set_edge(edge: int, coeffs: list[float]) -> None:
            for mode in range(self.modes_per_edge):
                fixed[self.edge_dof(edge, mode)] = True
                values[self.edge_dof(edge, mode)] = (
                    coeffs[mode] if mode < len(coeffs) else 0.0
                )

        for i in range(self.nx):
            x0 = i * self.cell_width
            x1 = (i + 1) * self.cell_width
            u0 = 1.0 - x0 / total_width
            u1 = 1.0 - x1 / total_width
            coeffs = [0.5 * (u0 + u1), 0.5 * (u1 - u0)]
            set_edge(self.horizontal_edge(i, 0), coeffs)
            set_edge(self.horizontal_edge(i, self.ny), coeffs)

        for j in range(self.ny):
            set_edge(self.vertical_edge(0, j), [1.0])
            set_edge(self.vertical_edge(self.nx, j), [0.0])

        return fixed, values

@dataclass(frozen=True)
class CompatibleStructuredGridAssembly:
    nx: int
    ny: int
    bubbles_per_edge: int
    stiffness: np.ndarray
    cell_width: float = 1.0
    cell_height: float = 1.0

    @property
    def vertex_count(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def horizontal_edges(self) -> int:
        return self.nx * (self.ny + 1)

    @property
    def vertical_edges(self) -> int:
        return (self.nx + 1) * self.ny

    @property
    def edge_count(self) -> int:
        return self.horizontal_edges + self.vertical_edges

    @property
    def dofs(self) -> int:
        return self.vertex_count + self.edge_count * self.bubbles_per_edge

    @property
    def local_dofs(self) -> int:
        return 4 + 4 * self.bubbles_per_edge

    def vertex(self, i: int, j: int) -> int:
        return j * (self.nx + 1) + i

    def horizontal_edge(self, i: int, j: int) -> int:
        return j * self.nx + i

    def vertical_edge(self, i: int, j: int) -> int:
        return self.horizontal_edges + j * (self.nx + 1) + i

    def edge_bubble_dof(self, edge_id: int, mode: int) -> int:
        return self.vertex_count + edge_id * self.bubbles_per_edge + mode

    def reversed_bubble_signs(self) -> np.ndarray:
        return np.array(
            [1.0 if mode % 2 == 0 else -1.0 for mode in range(self.bubbles_per_edge)],
            dtype=np.float64,
        )

    def local_dof_map(self, cell_i: int, cell_j: int) -> list[tuple[int, float]]:
        same = np.ones(self.bubbles_per_edge, dtype=np.float64)
        rev = self.reversed_bubble_signs()
        pairs: list[tuple[int, float]] = [
            (self.vertex(cell_i, cell_j), 1.0),
            (self.vertex(cell_i + 1, cell_j), 1.0),
            (self.vertex(cell_i + 1, cell_j + 1), 1.0),
            (self.vertex(cell_i, cell_j + 1), 1.0),
        ]
        edges = [
            (self.horizontal_edge(cell_i, cell_j), same),
            (self.vertical_edge(cell_i + 1, cell_j), same),
            (self.horizontal_edge(cell_i, cell_j + 1), rev),
            (self.vertical_edge(cell_i, cell_j), rev),
        ]
        for edge_id, signs in edges:
            for mode in range(self.bubbles_per_edge):
                pairs.append((self.edge_bubble_dof(edge_id, mode), float(signs[mode])))
        return pairs

    def assemble_matrix(self) -> np.ndarray:
        if self.stiffness.shape != (self.local_dofs, self.local_dofs):
            raise ValueError(
                f"local stiffness shape {self.stiffness.shape} does not match {self.local_dofs}"
            )
        k_global = np.zeros((self.dofs, self.dofs), dtype=np.float64)
        for cell_j in range(self.ny):
            for cell_i in range(self.nx):
                local_map = self.local_dof_map(cell_i, cell_j)
                for local_a, (global_a, sign_a) in enumerate(local_map):
                    for local_b, (global_b, sign_b) in enumerate(local_map):
                        k_global[global_a, global_b] += (
                            sign_a * self.stiffness[local_a, local_b] * sign_b
                        )
        return k_global

    def dirichlet_vector_linear_x(self) -> tuple[np.ndarray, np.ndarray]:
        fixed = np.zeros(self.dofs, dtype=bool)
        values = np.zeros(self.dofs, dtype=np.float64)
        total_width = self.nx * self.cell_width

        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                if i in (0, self.nx) or j in (0, self.ny):
                    dof = self.vertex(i, j)
                    x = i * self.cell_width
                    fixed[dof] = True
                    values[dof] = 1.0 - x / total_width

        def fix_edge_bubbles(edge_id: int) -> None:
            for mode in range(self.bubbles_per_edge):
                fixed[self.edge_bubble_dof(edge_id, mode)] = True

        for i in range(self.nx):
            fix_edge_bubbles(self.horizontal_edge(i, 0))
            fix_edge_bubbles(self.horizontal_edge(i, self.ny))
        for j in range(self.ny):
            fix_edge_bubbles(self.vertical_edge(0, j))
            fix_edge_bubbles(self.vertical_edge(self.nx, j))

        return fixed, values

    def dirichlet_vector_top_sine(self) -> tuple[np.ndarray, np.ndarray]:
        fixed = np.zeros(self.dofs, dtype=bool)
        values = np.zeros(self.dofs, dtype=np.float64)
        total_width = self.nx * self.cell_width
        top_j = self.ny

        def boundary_value(i: int, j: int) -> float:
            if j == top_j:
                x = i * self.cell_width
                return float(np.sin(np.pi * x / total_width))
            return 0.0

        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                if i in (0, self.nx) or j in (0, self.ny):
                    dof = self.vertex(i, j)
                    fixed[dof] = True
                    values[dof] = boundary_value(i, j)

        def fix_edge_bubbles_zero(edge_id: int) -> None:
            for mode in range(self.bubbles_per_edge):
                fixed[self.edge_bubble_dof(edge_id, mode)] = True

        for i in range(self.nx):
            fix_edge_bubbles_zero(self.horizontal_edge(i, 0))
        for j in range(self.ny):
            fix_edge_bubbles_zero(self.vertical_edge(0, j))
            fix_edge_bubbles_zero(self.vertical_edge(self.nx, j))

        xis = np.linspace(-1.0, 1.0, 25)
        bubble = (
            vertex_bubble_matrix(xis, self.bubbles_per_edge)
            if self.bubbles_per_edge > 0
            else np.zeros((xis.size, 0), dtype=np.float64)
        )
        for i in range(self.nx):
            edge = self.horizontal_edge(i, self.ny)
            x_left = i * self.cell_width
            x_right = (i + 1) * self.cell_width
            v_left = float(np.sin(np.pi * x_left / total_width))
            v_right = float(np.sin(np.pi * x_right / total_width))
            xs = x_left + 0.5 * (xis + 1.0) * self.cell_width
            target = np.sin(np.pi * xs / total_width)
            linear = 0.5 * (1.0 - xis) * v_left + 0.5 * (1.0 + xis) * v_right
            coeffs, *_ = np.linalg.lstsq(bubble, target - linear, rcond=None)
            for mode in range(self.bubbles_per_edge):
                dof = self.edge_bubble_dof(edge, mode)
                fixed[dof] = True
                values[dof] = coeffs[mode]

        return fixed, values

@dataclass
class MaskedCompatibleGridAssembly:
    nx: int
    ny: int
    active_cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]
    bubbles_per_edge: int
    stiffness: np.ndarray
    cell_width: float = 1.0
    cell_height: float = 1.0
    vertex_dofs: dict[tuple[int, int], int] = field(init=False, default_factory=dict)
    horizontal_edge_serials: dict[tuple[int, int], int] = field(init=False, default_factory=dict)
    vertical_edge_serials: dict[tuple[int, int], int] = field(init=False, default_factory=dict)
    edge_use_count: dict[tuple[str, int, int], int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        active = sorted({(int(i), int(j)) for i, j in self.active_cells})
        if not active:
            raise ValueError("active_cells must not be empty")
        for i, j in active:
            if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
                raise ValueError(f"active cell {(i, j)} outside grid {self.nx}x{self.ny}")
        self.active_cells = active

        vertices: set[tuple[int, int]] = set()
        horizontal: set[tuple[int, int]] = set()
        vertical: set[tuple[int, int]] = set()
        edge_use_count: dict[tuple[str, int, int], int] = {}
        for i, j in active:
            vertices.update(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))
            cell_edges = (
                ("h", i, j),
                ("v", i + 1, j),
                ("h", i, j + 1),
                ("v", i, j),
            )
            horizontal.update(((i, j), (i, j + 1)))
            vertical.update(((i, j), (i + 1, j)))
            for edge in cell_edges:
                edge_use_count[edge] = edge_use_count.get(edge, 0) + 1

        self.vertex_dofs = {key: idx for idx, key in enumerate(sorted(vertices))}
        serial = 0
        for key in sorted(horizontal):
            self.horizontal_edge_serials[key] = serial
            serial += 1
        for key in sorted(vertical):
            self.vertical_edge_serials[key] = serial
            serial += 1
        self.edge_use_count = edge_use_count

    @property
    def vertex_count(self) -> int:
        return len(self.vertex_dofs)

    @property
    def edge_count(self) -> int:
        return len(self.horizontal_edge_serials) + len(self.vertical_edge_serials)

    @property
    def dofs(self) -> int:
        return self.vertex_count + self.edge_count * self.bubbles_per_edge

    @property
    def local_dofs(self) -> int:
        return 4 + 4 * self.bubbles_per_edge

    @property
    def active_cell_count(self) -> int:
        return len(self.active_cells)

    def vertex(self, i: int, j: int) -> int:
        return self.vertex_dofs[(i, j)]

    def edge_serial(self, kind: str, i: int, j: int) -> int:
        if kind == "h":
            return self.horizontal_edge_serials[(i, j)]
        if kind == "v":
            return self.vertical_edge_serials[(i, j)]
        raise ValueError(f"Unsupported edge kind: {kind}")

    def edge_bubble_dof(self, kind: str, i: int, j: int, mode: int) -> int:
        return self.vertex_count + self.edge_serial(kind, i, j) * self.bubbles_per_edge + mode

    def edge_coords(self, kind: str, i: int, j: int) -> tuple[float, float, float, float]:
        if kind == "h":
            return (
                i * self.cell_width,
                j * self.cell_height,
                (i + 1) * self.cell_width,
                j * self.cell_height,
            )
        if kind == "v":
            return (
                i * self.cell_width,
                j * self.cell_height,
                i * self.cell_width,
                (j + 1) * self.cell_height,
            )
        raise ValueError(f"Unsupported edge kind: {kind}")

    def is_active_cell(self, i: int, j: int) -> bool:
        return (i, j) in set(self.active_cells)

    def is_outer_vertex(self, i: int, j: int) -> bool:
        return i in (0, self.nx) or j in (0, self.ny)

    def is_outer_edge(self, kind: str, i: int, j: int) -> bool:
        if kind == "h":
            return j in (0, self.ny)
        if kind == "v":
            return i in (0, self.nx)
        raise ValueError(f"Unsupported edge kind: {kind}")

    def boundary_edges(self) -> list[tuple[str, int, int]]:
        return sorted(edge for edge, count in self.edge_use_count.items() if count == 1)

    def outer_boundary_edges(self) -> list[tuple[str, int, int]]:
        return [edge for edge in self.boundary_edges() if self.is_outer_edge(*edge)]

    def hole_boundary_edges(self) -> list[tuple[str, int, int]]:
        return [edge for edge in self.boundary_edges() if not self.is_outer_edge(*edge)]

    def reversed_bubble_signs(self) -> np.ndarray:
        return bubble_signs(self.bubbles_per_edge, reversed_orientation=True)

    def local_dof_map(self, cell_i: int, cell_j: int) -> list[tuple[int, float]]:
        same = np.ones(self.bubbles_per_edge, dtype=np.float64)
        rev = self.reversed_bubble_signs()
        pairs: list[tuple[int, float]] = [
            (self.vertex(cell_i, cell_j), 1.0),
            (self.vertex(cell_i + 1, cell_j), 1.0),
            (self.vertex(cell_i + 1, cell_j + 1), 1.0),
            (self.vertex(cell_i, cell_j + 1), 1.0),
        ]
        edges = [
            ("h", cell_i, cell_j, same),
            ("v", cell_i + 1, cell_j, same),
            ("h", cell_i, cell_j + 1, rev),
            ("v", cell_i, cell_j, rev),
        ]
        for kind, i, j, signs in edges:
            for mode in range(self.bubbles_per_edge):
                pairs.append((self.edge_bubble_dof(kind, i, j, mode), float(signs[mode])))
        return pairs

    def assemble_matrix(self) -> np.ndarray:
        if self.stiffness.shape != (self.local_dofs, self.local_dofs):
            raise ValueError(
                f"local stiffness shape {self.stiffness.shape} does not match {self.local_dofs}"
            )
        k_global = np.zeros((self.dofs, self.dofs), dtype=np.float64)
        for cell_i, cell_j in self.active_cells:
            local_map = self.local_dof_map(cell_i, cell_j)
            for local_a, (global_a, sign_a) in enumerate(local_map):
                for local_b, (global_b, sign_b) in enumerate(local_map):
                    k_global[global_a, global_b] += (
                        sign_a * self.stiffness[local_a, local_b] * sign_b
                    )
        return k_global

    def assemble_sparse_matrix(self):
        from scipy.sparse import coo_matrix

        if self.stiffness.shape != (self.local_dofs, self.local_dofs):
            raise ValueError(
                f"local stiffness shape {self.stiffness.shape} does not match {self.local_dofs}"
            )
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for cell_i, cell_j in self.active_cells:
            local_map = self.local_dof_map(cell_i, cell_j)
            for local_a, (global_a, sign_a) in enumerate(local_map):
                for local_b, (global_b, sign_b) in enumerate(local_map):
                    value = sign_a * self.stiffness[local_a, local_b] * sign_b
                    if value != 0.0:
                        rows.append(global_a)
                        cols.append(global_b)
                        data.append(float(value))
        return coo_matrix((data, (rows, cols)), shape=(self.dofs, self.dofs)).tocsr()

    def assemble_sparse_matrix_with_cell_scales(self, cell_scales: dict[tuple[int, int], float]):
        from scipy.sparse import coo_matrix

        if self.stiffness.shape != (self.local_dofs, self.local_dofs):
            raise ValueError(
                f"local stiffness shape {self.stiffness.shape} does not match {self.local_dofs}"
            )
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for cell_i, cell_j in self.active_cells:
            scale = float(cell_scales.get((cell_i, cell_j), 1.0))
            local_map = self.local_dof_map(cell_i, cell_j)
            for local_a, (global_a, sign_a) in enumerate(local_map):
                for local_b, (global_b, sign_b) in enumerate(local_map):
                    value = scale * sign_a * self.stiffness[local_a, local_b] * sign_b
                    if value != 0.0:
                        rows.append(global_a)
                        cols.append(global_b)
                        data.append(float(value))
        return coo_matrix((data, (rows, cols)), shape=(self.dofs, self.dofs)).tocsr()

    def _set_edge_bubbles(self, fixed: np.ndarray, values: np.ndarray, edge: tuple[str, int, int], coeffs: np.ndarray) -> None:
        kind, i, j = edge
        for mode in range(self.bubbles_per_edge):
            dof = self.edge_bubble_dof(kind, i, j, mode)
            fixed[dof] = True
            values[dof] = float(coeffs[mode]) if mode < coeffs.size else 0.0

    def dirichlet_vector_linear_x(self) -> tuple[np.ndarray, np.ndarray]:
        fixed = np.zeros(self.dofs, dtype=bool)
        values = np.zeros(self.dofs, dtype=np.float64)
        total_width = self.nx * self.cell_width
        for (i, j), dof in self.vertex_dofs.items():
            if self.is_outer_vertex(i, j):
                fixed[dof] = True
                values[dof] = 1.0 - (i * self.cell_width) / total_width
        zeros = np.zeros(self.bubbles_per_edge, dtype=np.float64)
        for edge in self.outer_boundary_edges():
            self._set_edge_bubbles(fixed, values, edge, zeros)
        return fixed, values

    def dirichlet_vector_top_sine(self) -> tuple[np.ndarray, np.ndarray]:
        fixed = np.zeros(self.dofs, dtype=bool)
        values = np.zeros(self.dofs, dtype=np.float64)
        total_width = self.nx * self.cell_width
        top_y = self.ny * self.cell_height
        for (i, j), dof in self.vertex_dofs.items():
            if self.is_outer_vertex(i, j):
                x = i * self.cell_width
                y = j * self.cell_height
                fixed[dof] = True
                values[dof] = float(np.sin(np.pi * x / total_width)) if abs(y - top_y) <= 1e-12 else 0.0

        zeros = np.zeros(self.bubbles_per_edge, dtype=np.float64)
        xis = np.linspace(-1.0, 1.0, 25)
        bubble = (
            vertex_bubble_matrix(xis, self.bubbles_per_edge)
            if self.bubbles_per_edge > 0
            else np.zeros((xis.size, 0), dtype=np.float64)
        )
        for edge in self.outer_boundary_edges():
            kind, i, j = edge
            if kind == "h" and j == self.ny:
                x_left = i * self.cell_width
                x_right = (i + 1) * self.cell_width
                v_left = float(np.sin(np.pi * x_left / total_width))
                v_right = float(np.sin(np.pi * x_right / total_width))
                xs = x_left + 0.5 * (xis + 1.0) * self.cell_width
                target = np.sin(np.pi * xs / total_width)
                linear = 0.5 * (1.0 - xis) * v_left + 0.5 * (1.0 + xis) * v_right
                if self.bubbles_per_edge > 0:
                    coeffs, *_ = np.linalg.lstsq(bubble, target - linear, rcond=None)
                    self._set_edge_bubbles(fixed, values, edge, coeffs)
            else:
                self._set_edge_bubbles(fixed, values, edge, zeros)
        return fixed, values

def solve_dirichlet_system(
    k_global: np.ndarray,
    fixed: np.ndarray,
    values: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    free = ~fixed
    q = values.copy()
    k_ff = k_global[np.ix_(free, free)]
    k_fc = k_global[np.ix_(free, fixed)]
    rhs = -k_fc @ values[fixed]
    q[free] = np.linalg.solve(k_ff, rhs)
    residual = k_global @ q
    free_residual = residual[free]
    energy = 0.5 * float(q @ k_global @ q)
    return {
        "q": q,
        "residual": residual,
        "energy": energy,
        "free_dofs": int(np.count_nonzero(free)),
        "fixed_dofs": int(np.count_nonzero(fixed)),
        "free_residual_norm": float(np.linalg.norm(free_residual)),
        "free_residual_relative": float(
            np.linalg.norm(free_residual) / (np.linalg.norm(k_global @ values) + 1e-12)
        ),
        "condition_number_free": float(np.linalg.cond(k_ff)),
    }

def solve_dirichlet_system_sparse(
    k_global,
    fixed: np.ndarray,
    values: np.ndarray,
) -> dict[str, np.ndarray | float | int | None]:
    from scipy.sparse.linalg import spsolve

    free = ~fixed
    q = values.copy()
    k_ff = k_global[free, :][:, free]
    k_fc = k_global[free, :][:, fixed]
    rhs = -(k_fc @ values[fixed])
    q[free] = spsolve(k_ff, rhs)
    residual = k_global @ q
    free_residual = residual[free]
    energy = 0.5 * float(q @ (k_global @ q))
    return {
        "q": q,
        "residual": residual,
        "energy": energy,
        "free_dofs": int(np.count_nonzero(free)),
        "fixed_dofs": int(np.count_nonzero(fixed)),
        "free_residual_norm": float(np.linalg.norm(free_residual)),
        "free_residual_relative": float(
            np.linalg.norm(free_residual) / (np.linalg.norm(k_global @ values) + 1e-12)
        ),
        "condition_number_free": None,
    }

@dataclass
class ConformingPolygonMeshAssembly:









    vertices: np.ndarray
    cells: list[list[int]] | tuple[tuple[int, ...], ...]
    bubbles_per_edge: int
    edge_serials: dict[tuple[int, int], int] = field(init=False, default_factory=dict)
    edge_use_count: dict[tuple[int, int], int] = field(init=False, default_factory=dict)

    @classmethod
    def from_quads(
        cls,
        *,
        vertices: np.ndarray,
        cells: np.ndarray | list[list[int]] | tuple[tuple[int, ...], ...],
        bubbles_per_edge: int,
    ) -> "ConformingPolygonMeshAssembly":








        points = np.asarray(vertices, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"vertices must have shape [V,2], got {points.shape}")
        if not np.all(np.isfinite(points)):
            raise ValueError("vertices contain non-finite values")
        bubbles = int(bubbles_per_edge)
        if bubbles < 0:
            raise ValueError("bubbles_per_edge must be nonnegative")
        cell_array = np.asarray(cells, dtype=np.int64)
        if cell_array.ndim != 2 or cell_array.shape[1] != 4:
            raise ValueError(f"quadrilateral cells must have shape [N,4], got {cell_array.shape}")
        if len(cell_array) == 0:
            raise ValueError("cells must not be empty")
        if np.any(cell_array < 0) or np.any(cell_array >= len(points)):
            raise ValueError(f"cells reference a vertex outside [0,{len(points)})")
        sorted_vertices = np.sort(cell_array, axis=1)
        if np.any(np.diff(sorted_vertices, axis=1) == 0):
            raise ValueError("a quadrilateral cell repeats a vertex")

        polygons = points[cell_array]
        shifted = np.roll(polygons, -1, axis=1)
        signed_areas = 0.5 * np.sum(
            polygons[:, :, 0] * shifted[:, :, 1]
            - shifted[:, :, 0] * polygons[:, :, 1],
            axis=1,
        )
        invalid = np.flatnonzero(signed_areas <= 1.0e-14)
        if len(invalid):
            raise ValueError(
                f"cell {int(invalid[0])} must be nondegenerate and counter-clockwise"
            )

        next_vertices = np.roll(cell_array, -1, axis=1)
        edges = np.sort(
            np.stack((cell_array, next_vertices), axis=2).reshape(-1, 2),
            axis=1,
        )
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        non_manifold = np.flatnonzero(counts > 2)
        if len(non_manifold):
            edge = tuple(int(value) for value in unique_edges[non_manifold[0]])
            raise ValueError(f"non-manifold edge {edge} is used by more than two cells")

        instance = cls.__new__(cls)
        instance.vertices = points
        instance.cells = [tuple(int(value) for value in row) for row in cell_array]
        instance.bubbles_per_edge = bubbles
        instance.edge_use_count = {
            tuple(int(value) for value in edge): int(count)
            for edge, count in zip(unique_edges, counts)
        }
        instance.edge_serials = {
            edge: serial for serial, edge in enumerate(instance.edge_use_count)
        }
        return instance

    def __post_init__(self) -> None:
        points = np.asarray(self.vertices, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"vertices must have shape [V,2], got {points.shape}")
        if not np.all(np.isfinite(points)):
            raise ValueError("vertices contain non-finite values")
        if self.bubbles_per_edge < 0:
            raise ValueError("bubbles_per_edge must be nonnegative")
        normalized_cells: list[tuple[int, ...]] = []
        edge_use_count: dict[tuple[int, int], int] = {}
        for cell_index, raw_cell in enumerate(self.cells):
            cell = tuple(int(value) for value in raw_cell)
            if not (3 <= len(cell) <= 8):
                raise ValueError(f"cell {cell_index} must have 3--8 vertices, got {len(cell)}")
            if len(set(cell)) != len(cell):
                raise ValueError(f"cell {cell_index} repeats a vertex")
            if min(cell) < 0 or max(cell) >= points.shape[0]:
                raise ValueError(f"cell {cell_index} references a vertex outside [0,{points.shape[0]})")
            polygon = points[np.asarray(cell, dtype=np.int64)]
            shifted = np.roll(polygon, -1, axis=0)
            signed_area = 0.5 * float(
                np.sum(polygon[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon[:, 1])
            )
            if signed_area <= 1.0e-14:
                raise ValueError(f"cell {cell_index} must be nondegenerate and counter-clockwise")
            for local_edge in range(len(cell)):
                start = cell[local_edge]
                end = cell[(local_edge + 1) % len(cell)]
                edge = (min(start, end), max(start, end))
                edge_use_count[edge] = edge_use_count.get(edge, 0) + 1
                if edge_use_count[edge] > 2:
                    raise ValueError(f"non-manifold edge {edge} is used by more than two cells")
            normalized_cells.append(cell)
        if not normalized_cells:
            raise ValueError("cells must not be empty")
        self.vertices = points
        self.cells = normalized_cells
        self.edge_use_count = edge_use_count
        self.edge_serials = {edge: serial for serial, edge in enumerate(sorted(edge_use_count))}

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def edge_count(self) -> int:
        return len(self.edge_serials)

    @property
    def scalar_dofs(self) -> int:
        return self.vertex_count + self.edge_count * self.bubbles_per_edge

    @property
    def vector_dofs(self) -> int:
        return 2 * self.scalar_dofs

    def edge_bubble_dof(self, edge: tuple[int, int], mode: int) -> int:
        canonical = (min(edge), max(edge))
        if canonical not in self.edge_serials:
            raise KeyError(f"unknown mesh edge {edge}")
        if not (0 <= mode < self.bubbles_per_edge):
            raise IndexError(f"bubble mode {mode} outside [0,{self.bubbles_per_edge})")
        return self.vertex_count + self.edge_serials[canonical] * self.bubbles_per_edge + mode

    def boundary_edges(self) -> list[tuple[int, int]]:
        return sorted(edge for edge, count in self.edge_use_count.items() if count == 1)

    def boundary_vertices(
        self,
        edges: list[tuple[int, int]] | None = None,
    ) -> list[int]:
        selected = self.boundary_edges() if edges is None else edges
        return sorted({vertex for edge in selected for vertex in edge})

    def select_boundary_edges(
        self,
        edge_selector=None,
        selected_edges: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        edges = self.boundary_edges()
        if selected_edges is not None:
            if edge_selector is not None:
                raise ValueError("Provide either edge_selector or selected_edges, not both")
            boundary_set = set(edges)
            normalized = sorted({(min(edge), max(edge)) for edge in selected_edges})
            unknown = [edge for edge in normalized if edge not in boundary_set]
            if unknown:
                raise ValueError(f"selected_edges contains non-boundary edges: {unknown[:8]}")
            return normalized
        if edge_selector is None:
            return edges
        selected: list[tuple[int, int]] = []
        for edge in edges:
            start, end = edge
            if edge_selector(self.vertices[start].copy(), self.vertices[end].copy()):
                selected.append(edge)
        return selected

    def scalar_dirichlet_values(
        self,
        value_function,
        *,
        sample_count: int = 25,
        edge_selector=None,
        selected_edges: list[tuple[int, int]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:


        if sample_count < max(3, self.bubbles_per_edge + 2):
            raise ValueError("sample_count is too small for the requested bubble projection")
        fixed = np.zeros(self.scalar_dofs, dtype=bool)
        values = np.zeros(self.scalar_dofs, dtype=np.float64)
        active_edges = self.select_boundary_edges(edge_selector, selected_edges)
        for vertex in self.boundary_vertices(active_edges):
            x, y = self.vertices[vertex]
            fixed[vertex] = True
            values[vertex] = float(value_function(float(x), float(y)))
        if self.bubbles_per_edge <= 0:
            return fixed, values
        xis = np.linspace(-1.0, 1.0, sample_count)
        bubble = vertex_bubble_matrix(xis, self.bubbles_per_edge)
        for edge in active_edges:
            start, end = edge
            start_xy = self.vertices[start]
            end_xy = self.vertices[end]
            t = 0.5 * (xis + 1.0)
            points = (1.0 - t[:, None]) * start_xy[None, :] + t[:, None] * end_xy[None, :]
            target = np.asarray(
                [value_function(float(x), float(y)) for x, y in points],
                dtype=np.float64,
            )
            linear = 0.5 * (1.0 - xis) * values[start] + 0.5 * (1.0 + xis) * values[end]
            coefficients, *_ = np.linalg.lstsq(bubble, target - linear, rcond=None)
            for mode, coefficient in enumerate(coefficients):
                dof = self.edge_bubble_dof(edge, mode)
                fixed[dof] = True
                values[dof] = float(coefficient)
        return fixed, values

    def vector_dirichlet_values(
        self,
        value_function,
        *,
        sample_count: int = 25,
        edge_selector=None,
        selected_edges: list[tuple[int, int]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:


        component_data = []
        for component in range(2):
            component_data.append(
                self.scalar_dirichlet_values(
                    lambda x, y, component=component: value_function(x, y)[component],
                    sample_count=sample_count,
                    edge_selector=edge_selector,
                    selected_edges=selected_edges,
                )
            )
        fixed = np.concatenate([component_data[0][0], component_data[1][0]])
        values = np.concatenate([component_data[0][1], component_data[1][1]])
        return fixed, values

    def local_scalar_dof_map(self, cell_index: int) -> list[tuple[int, float]]:
        cell = self.cells[cell_index]
        mapping: list[tuple[int, float]] = [(vertex, 1.0) for vertex in cell]
        for local_edge in range(len(cell)):
            start = cell[local_edge]
            end = cell[(local_edge + 1) % len(cell)]
            reversed_orientation = start > end
            signs = bubble_signs(
                self.bubbles_per_edge,
                reversed_orientation=reversed_orientation,
            )
            for mode in range(self.bubbles_per_edge):
                mapping.append((self.edge_bubble_dof((start, end), mode), float(signs[mode])))
        return mapping

    def local_vector_dof_map(self, cell_index: int) -> list[tuple[int, float]]:
        scalar = self.local_scalar_dof_map(cell_index)
        return scalar + [(self.scalar_dofs + dof, sign) for dof, sign in scalar]

    def _assemble(
        self,
        local_stiffnesses: list[np.ndarray] | tuple[np.ndarray, ...],
        *,
        vector: bool,
        sparse: bool,
    ):
        if len(local_stiffnesses) != len(self.cells):
            raise ValueError(
                f"expected {len(self.cells)} local stiffness matrices, got {len(local_stiffnesses)}"
            )
        total_dofs = self.vector_dofs if vector else self.scalar_dofs
        prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for cell_index, raw_stiffness in enumerate(local_stiffnesses):
            mapping = (
                self.local_vector_dof_map(cell_index)
                if vector
                else self.local_scalar_dof_map(cell_index)
            )
            stiffness = np.asarray(raw_stiffness, dtype=np.float64)
            expected = len(mapping)
            if stiffness.shape != (expected, expected):
                raise ValueError(
                    f"cell {cell_index} stiffness shape {stiffness.shape} does not match {(expected, expected)}"
                )
            global_map = np.fromiter(
                (int(dof) for dof, _sign in mapping),
                dtype=np.int64,
                count=expected,
            )
            signs = np.fromiter(
                (float(sign) for _dof, sign in mapping),
                dtype=np.float64,
                count=expected,
            )
            prepared.append((global_map, signs, stiffness))
        if not sparse:
            dense = np.zeros((total_dofs, total_dofs), dtype=np.float64)
            for global_map, signs, stiffness in prepared:
                transformed = signs[:, None] * stiffness * signs[None, :]
                np.add.at(
                    dense,
                    (global_map[:, None], global_map[None, :]),
                    transformed,
                )
            return dense
        from scipy.sparse import coo_matrix

        entry_count = sum(len(global_map) ** 2 for global_map, _signs, _stiffness in prepared)
        rows = np.empty(entry_count, dtype=np.int64)
        cols = np.empty(entry_count, dtype=np.int64)
        data = np.empty(entry_count, dtype=np.float64)
        offset = 0
        for global_map, signs, stiffness in prepared:
            local_dofs = len(global_map)
            count = local_dofs * local_dofs
            block = slice(offset, offset + count)
            rows[block] = np.repeat(global_map, local_dofs)
            cols[block] = np.tile(global_map, local_dofs)
            data[block] = (
                signs[:, None] * stiffness * signs[None, :]
            ).reshape(-1)
            offset += count
        return coo_matrix(
            (data, (rows, cols)), shape=(total_dofs, total_dofs)
        ).tocsr()

    def assemble_scalar(
        self,
        local_stiffnesses: list[np.ndarray] | tuple[np.ndarray, ...],
        *,
        sparse: bool = False,
    ):
        return self._assemble(local_stiffnesses, vector=False, sparse=sparse)

    def assemble_scalar_quads(
        self,
        local_stiffnesses: np.ndarray,
    ):







        from scipy.sparse import coo_matrix

        cells = np.asarray(self.cells, dtype=np.int64)
        matrices = np.asarray(local_stiffnesses, dtype=np.float64)
        local_dofs = 4 * (1 + int(self.bubbles_per_edge))
        if cells.ndim != 2 or cells.shape[1] != 4:
            raise ValueError("vectorized scalar assembly requires four-node cells")
        if matrices.shape != (len(cells), local_dofs, local_dofs):
            raise ValueError(
                f"expected {(len(cells), local_dofs, local_dofs)} local matrices, "
                f"got {matrices.shape}"
            )

        next_vertices = np.roll(cells, -1, axis=1)
        canonical_edges = np.sort(
            np.stack((cells, next_vertices), axis=2), axis=2
        )
        edge_vertices = np.asarray(tuple(self.edge_serials), dtype=np.int64)
        edge_codes = edge_vertices[:, 0] * self.vertex_count + edge_vertices[:, 1]
        local_codes = (
            canonical_edges[:, :, 0] * self.vertex_count
            + canonical_edges[:, :, 1]
        )
        edge_serials = np.searchsorted(edge_codes, local_codes)
        if (
            np.any(edge_serials >= self.edge_count)
            or not np.array_equal(edge_codes[edge_serials], local_codes)
        ):
            raise RuntimeError("vectorized scalar edge map disagrees with topology")

        modes = int(self.bubbles_per_edge)
        bubble_dofs = (
            self.vertex_count
            + edge_serials[:, :, None] * modes
            + np.arange(modes, dtype=np.int64)[None, None, :]
        ).reshape(len(cells), -1)
        global_map = np.concatenate((cells, bubble_dofs), axis=1)
        reversed_edges = cells > next_vertices
        bubble_sign = np.ones((len(cells), 4, modes), dtype=np.float64)
        if modes > 1:
            bubble_sign[:, :, 1::2] = np.where(
                reversed_edges[:, :, None], -1.0, 1.0
            )
        signs = np.concatenate(
            (np.ones((len(cells), 4), dtype=np.float64), bubble_sign.reshape(len(cells), -1)),
            axis=1,
        )
        transformed = signs[:, :, None] * matrices * signs[:, None, :]
        rows = np.broadcast_to(
            global_map[:, :, None], (len(cells), local_dofs, local_dofs)
        ).reshape(-1)
        columns = np.broadcast_to(
            global_map[:, None, :], (len(cells), local_dofs, local_dofs)
        ).reshape(-1)
        return coo_matrix(
            (transformed.reshape(-1), (rows, columns)),
            shape=(self.scalar_dofs, self.scalar_dofs),
        ).tocsr()

    def assemble_vector(
        self,
        local_stiffnesses: list[np.ndarray] | tuple[np.ndarray, ...],
        *,
        sparse: bool = False,
    ):
        return self._assemble(local_stiffnesses, vector=True, sparse=sparse)

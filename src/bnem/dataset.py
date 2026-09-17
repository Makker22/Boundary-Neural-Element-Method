from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.bnem.convergence import (
    certified_manifest_artifact_paths,
    validate_convergence_certificate,
)

@dataclass(frozen=True)
class PatchSample:
    path: Path
    eta: np.ndarray
    stiffness: np.ndarray
    port_scheme: str
    feature_mode: str
    stiffness_scale: float
    bubbles_per_edge: int | None
    port_dofs: int
    dataset_role: str | None
    convergence_certificate: Path | None
    convergence: dict[str, Any] | None

def rectangle_width_height(data: np.lib.npyio.NpzFile) -> tuple[float, float]:
    vertices = np.asarray(data["eta_vertices"], dtype=np.float64)
    width = float(np.linalg.norm(vertices[1] - vertices[0]))
    height = float(np.linalg.norm(vertices[3] - vertices[0]))
    return width, height

def stiffness_scale_from_npz(data: np.lib.npyio.NpzFile, feature_mode: str = "legacy_dimensional") -> float:
    conductivity = float(np.asarray(data["eta_k"]).item())
    if feature_mode == "legacy_dimensional":
        return 1.0
    if feature_mode == "scalar2d_dimless_v1":
        return conductivity
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")

def eta_from_npz(data: np.lib.npyio.NpzFile, feature_mode: str = "legacy_dimensional") -> np.ndarray:
    width, height = rectangle_width_height(data)
    conductivity = float(np.asarray(data["eta_k"]).item())
    if feature_mode == "legacy_dimensional":
        return np.array([width, height, conductivity], dtype=np.float64)
    if feature_mode == "scalar2d_dimless_v1":
        area_length = float(np.sqrt(width * height))
        return np.array([width / area_length, height / area_length], dtype=np.float64)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")

def load_patch_sample(
    path: str | Path,
    feature_mode: str = "legacy_dimensional",
    *,
    require_converged: bool = True,
    allowed_roles: Iterable[str] | None = None,
) -> PatchSample:
    patch_path = Path(path)
    convergence = None
    if require_converged:
        convergence = validate_convergence_certificate(patch_path, allowed_roles=allowed_roles)
    data = np.load(patch_path, allow_pickle=False)
    eta = eta_from_npz(data, feature_mode=feature_mode)
    stiffness_scale = stiffness_scale_from_npz(data, feature_mode=feature_mode)
    stiffness = np.asarray(data["S"], dtype=np.float64) / stiffness_scale
    port_scheme = str(np.asarray(data["port_scheme"]).item()) if "port_scheme" in data.files else "edge_legendre"
    bubbles_per_edge = (
        int(np.asarray(data["bubbles_per_edge"]).item())
        if "bubbles_per_edge" in data.files
        else None
    )
    return PatchSample(
        path=patch_path,
        eta=eta,
        stiffness=stiffness,
        port_scheme=port_scheme,
        feature_mode=feature_mode,
        stiffness_scale=stiffness_scale,
        bubbles_per_edge=bubbles_per_edge,
        port_dofs=int(stiffness.shape[0]),
        dataset_role=None if convergence is None else str(convergence["dataset_role"]),
        convergence_certificate=(
            None if convergence is None else Path(str(convergence["certificate_path"]))
        ),
        convergence=convergence,
    )

def find_patch_npz(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    manifested = certified_manifest_artifact_paths(root)
    if manifested is not None:
        return sorted(
            path
            for path in manifested
            if path.is_file() and path.name.startswith("patch_") and path.suffix == ".npz"
        )
    return sorted(path for path in root.rglob("patch_*.npz") if path.is_file())

def patch_matches_filters(
    path: Path,
    *,
    port_scheme: str | None,
    bubbles_per_edge: int | None,
    port_dofs: int | None,
) -> bool:
    with np.load(path, allow_pickle=False) as data:
        sample_port_scheme = (
            str(np.asarray(data["port_scheme"]).item())
            if "port_scheme" in data.files
            else "edge_legendre"
        )
        sample_bubbles = (
            int(np.asarray(data["bubbles_per_edge"]).item())
            if "bubbles_per_edge" in data.files
            else None
        )
        sample_port_dofs = int(np.asarray(data["S"]).shape[0])
    return bool(
        (port_scheme is None or sample_port_scheme == port_scheme)
        and (bubbles_per_edge is None or sample_bubbles == bubbles_per_edge)
        and (port_dofs is None or sample_port_dofs == port_dofs)
    )

def load_patch_samples(
    data_dir: str | Path,
    port_scheme: str | None = None,
    feature_mode: str = "legacy_dimensional",
    bubbles_per_edge: int | None = None,
    port_dofs: int | None = None,
    require_converged: bool = True,
    allowed_roles: Iterable[str] | None = None,
) -> list[PatchSample]:
    paths = [
        path
        for path in find_patch_npz(data_dir)
        if patch_matches_filters(
            path,
            port_scheme=port_scheme,
            bubbles_per_edge=bubbles_per_edge,
            port_dofs=port_dofs,
        )
    ]
    samples = [
        load_patch_sample(
            path,
            feature_mode=feature_mode,
            require_converged=require_converged,
            allowed_roles=None,
        )
        for path in paths
    ]
    if allowed_roles is not None:
        role_set = set(allowed_roles)
        samples = [sample for sample in samples if sample.dataset_role in role_set]
    return samples
